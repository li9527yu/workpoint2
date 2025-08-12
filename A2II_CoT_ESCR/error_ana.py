# 错误分析、例子分析：把预测错误的例子保存下来，查看对应的生成的知识
# 细粒度分析：计算每个类别具体的性能分数

import logging
import torch
import torch.utils.data as Data
from torch.optim import AdamW
import argparse
import os
import numpy as np
from tqdm import tqdm, trange
import random
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from CoT_Model import MyFlanT5
from T5_model_v2 import MyFlanT5
from DataProcessor_first_rel import MyDataset
from sklearn.metrics import precision_recall_fscore_support
import wandb
from torch import nn
from tool import warmup_linear,parse_sequences,compute_metrics
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/annotation/data", type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ESCR/witoutImage/42", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=2, type=int)
    parser.add_argument('--LEARNING_RATE', default=1e-5, type=float)
    parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%  of training.")
    parser.add_argument("--max_seq_len",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--run_name",
                        default="pytorch_model",
                        type=str,
                        help="运行的名称")
    parser.add_argument("--output_model_name",
                        default="pytorch_model",
                        type=str,
                        help="保存权重的名称")
    parser.add_argument("--output_log_name",
                        default="log",
                        type=str,
                        help="日志的名称")  
    opt = parser.parse_args()

    return opt

def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label,relation_label=batch
      
    input_ids = input_ids.clone().detach().long().to(device)
    input_attention_mask = input_attention_mask.clone().detach().float().to(device)
    
    labels=labels.to(device).long()
    senti_label=senti_label.to(device).long()
    relation_label=relation_label.to(device).long()

    input_pooler_outputs = input_pooler_outputs.clone().detach().to(input_ids.device)
    input_hidden_states = input_hidden_states.clone().detach().to(input_ids.device)
      
    return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label,relation_label

def evaluate(args,model,test_dataloader,logger):
    model.eval()
    pred_sequence,senti_labels=[],[]
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label,relation_label= post_dataloader(batch)
        with torch.no_grad():
            outputs= model.generate(input_ids=input_ids,
                            attention_mask=input_attention_mask,              
                            input_hidden_states    = input_hidden_states,
                            labels=labels,
                            relation=relation_label)
            pred_sequence.extend(outputs)
            senti_labels.extend(senti_label.detach().cpu().numpy())

    senti_preds=parse_sequences(pred_sequence)
    senti_result = compute_metrics(senti_preds, senti_labels,logger)
    result = {     
        f'senti_acc':senti_result['acc'],    
        f'senti_f1':senti_result['f1'],  
   
    }
    
    return result,senti_preds,senti_labels

def main(args):
    model_path='/data/lzy1211/code/model/flan-t5-base/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_model_file = os.path.join(args.output_dir, f"{args.output_model_name}.bin")
    output_logger_file=os.path.join(args.output_dir,f'{args.output_log_name}.txt')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO,
                filename=output_logger_file)
    logger = logging.getLogger(__name__)

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    eos_token_id = tokenizer.eos_token_id  
    t5_model=T5ForConditionalGeneration.from_pretrained(model_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(args)
    
    emotion_data=f"/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/text_emotion/twitter2017/train.json"
    test_data=f"{args.data_dir}/test.json"
    test_img_data=f"{args.img_feat_dir}/twitter2017/train.pkl"
    test_dataset=MyDataset(args,test_data,test_img_data,emotion_data,tokenizer)

    # model
    model = MyFlanT5(model_path=t5_model,tokenizer=tokenizer)
    model.to(device)
    
    # Test
    model.eval()
    logger.info("***** Running evaluation on Test Set*****")
    logger.info("  Num examples = %d", test_dataset.number) #len(eval_examples)
    logger.info("  Batch size = %d", args.BATCH_SIZE)
    test_dataloader= Data.DataLoader(dataset=test_dataset,shuffle=False, batch_size=args.BATCH_SIZE,num_workers=0)
    args.output_test_path=os.path.join(args.output_dir,f'ana_pred.json')


    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(args.output_model_file)
    model.load_state_dict(model_state_dict)
    
    
    test_results,senti_preds,senti_labels=evaluate(args,model,test_dataloader,logger)
    logger.info("***** Test Eval results *****")
    for key in sorted(test_results.keys()):
        logger.info("  %s = %s", key, str(test_results[key]))

    # # 保存预测的结果
    with open(test_data,'r',encoding='utf-8') as f:
        test_list=json.load(f)

    # sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
    sentiment_map = {'0': 'negative', '1': 'neutral', '2': 'positive'}
    pred_res=[]
    for pred,gold,item in zip(senti_preds,senti_labels,test_list):
        item['pred']=sentiment_map[str(pred)]
        item['gold']=sentiment_map[str(gold)]
        if item['aspect']=="Barbara Hepworth":
            print("A")
        pred_res.append(item)
    with open(args.output_test_path,'w',encoding='utf-8') as f:
        json.dump(pred_res,f)
    
    # 6. 详细错误类型分析
    errors = []
    for item in pred_res:
        if item['pred'] != item['gold']:
            errors.append(item)

    # 7. 错误统计
    error_types = {'neutral_to_negative': 0, 'neutral_to_positive': 0,
                'negative_to_neutral': 0, 'negative_to_positive': 0,
                'positive_to_neutral': 0, 'positive_to_negative': 0}
    for e in errors:
        key = f"{e['gold']}_to_{e['pred']}"
        if key in error_types:
            error_types[key] += 1

    logger.info('\nError Type Statistics:')
    for et, count in error_types.items():
        logger.info(f"{et}: {count}")

    # 8. 输出错误样本
    with open(os.path.join(args.output_dir,'error_samples.json'), 'w') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    logger.info(f"\nTotal Errors: {len(errors)} (saved to 'error_samples.json')")



if __name__ == "__main__":
    args=get_parser()
    main(args)