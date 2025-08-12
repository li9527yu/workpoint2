import logging
import torch
import torch.utils.data as Data
from torch.optim import AdamW
import argparse
import os
from datasets import Dataset
import pickle
import json
import numpy as np
from tqdm import tqdm, trange
import random
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration,DataCollatorForSeq2Seq
from T5_Model import MyFlanT5
from DataProcessor_Re import ReDataset
from DataProcessor_first_rel import MyDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import wandb
from transformers import RobertaTokenizer, RobertaModel
from itertools import cycle
from torch import nn
from tool import warmup_linear,parse_sequences,compute_metrics

# 标签映射
label_map = {"positive": 2, "negative": 0, "neutral": 1}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/annotation/data", type=str)
    parser.add_argument('--Re_data_dir', default="/data/lzy1211/code/A2II/instructBLIP/relation_data", type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/A2II-ESCR_original/42/", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=1, type=int)
    parser.add_argument('--LEARNING_RATE', default=1e-5, type=float)
    parser.add_argument('--Rel_LEARNING_RATE', default=1e-6, type=float)
    parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%  of training.")
    parser.add_argument("--max_seq_length",
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
    input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,label_index=batch    
        
    input_re_ids = input_re_ids.clone().detach().long().to(device)
    input_re_attention_mask = input_re_attention_mask.clone().detach().float().to(device)
    input_ir_ids = input_ir_ids.clone().detach().long().to(device)
    input_ir_attention_mask = input_ir_attention_mask.clone().detach().float().to(device)
    
    labels=labels.to(device).long()
    label_index=label_index.to(device).long()
    input_pooler_outputs = input_pooler_outputs.clone().detach().to(input_re_ids.device)
    input_hidden_states = input_hidden_states.clone().detach().to(input_re_ids.device)

            
    return input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,label_index



def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def evaluate(model,test_dataloader, logger):
    model.eval()
    pred_sequence,senti_labels=[],[]
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,label_index= post_dataloader(batch)
        with torch.no_grad():
            outputs= model.generate(input_ids=input_re_ids,
                                attention_mask     = input_re_attention_mask, 
                                input_ir_ids   = input_ir_ids, 
                                input_ir_attention_mask     = input_ir_attention_mask,                 
                                input_hidden_state    = input_hidden_states,
                                input_pooler_output=input_pooler_outputs,
                                labels=labels)
            pred_sequence.extend(outputs)
            senti_labels.extend(label_index.detach().cpu().numpy())

    senti_preds=parse_sequences(pred_sequence)
    senti_result = compute_metrics(senti_preds, senti_labels)
    result = {     
        f'senti_acc':senti_result['acc'],    
        f'senti_f1':senti_result['f1'],  
   
    }   
    return result


# 获取相关性标签
def get_rel_data(dataset):
    path1=f'/data/lzy1211/code/gpt=api/{dataset}/test_process.json'
    path2=f'/data/lzy1211/code/gpt=api/{dataset}/train_process.json'

    
    with open(path1,'r') as f:
        rel1=json.load(f)
    f.close()
    with open(path2,'r') as f:
        rel2=json.load(f)
    f.close()

    rel=rel1+rel2
    return rel

def main(args):

    model_path='/data/lzy1211/code/model/flan-t5-base/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_model_file = os.path.join(args.output_dir, f"{args.output_model_name}.bin")
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
    rel_data=get_rel_data(args.dataset)
    # rel_data=[]
    train_data=f"{args.data_dir}/train.json"
    val_data=f"{args.data_dir}/dev.json"
    test_data=f"{args.data_dir}/test.json"
    img_data=f"{args.img_feat_dir}/twitter2017/train.pkl"
    train_dataset=MyDataset(train_data,tokenizer,rel_data,img_data,max_seq_len= args.max_seq_length)
    val_dataset=MyDataset(val_data,tokenizer,rel_data,img_data,max_seq_len= args.max_seq_length)
    test_dataset=MyDataset(test_data,tokenizer,rel_data,img_data,max_seq_len= args.max_seq_length)

    train_dataloader= Data.DataLoader(dataset=train_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    val_dataloader= Data.DataLoader(dataset=val_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    test_dataloader= Data.DataLoader(dataset=test_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)

    train_number=train_dataset.number
    num_train_steps = int( train_number / args.BATCH_SIZE * args.EPOCHS)

    model = MyFlanT5(model_path=t5_model,tokenizer=tokenizer)
    model.to(device)

    # Prepare optimizer
    # optimizer BertAdam
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer_t5 = AdamW(optimizer_grouped_parameters,
                            lr=args.LEARNING_RATE)
    max_senti_acc = 0.0
    best_epoch=-1
    global_step = 0

    # # train
    logger.info("*************** Running training ***************")
    for train_idx in trange(int(args.EPOCHS), desc="Epoch"):
        logger.info("************************************************** Epoch: "+ str(train_idx) + " *************************************************************")
        logger.info("  Num examples = %d",  train_number) 
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        logger.info("  Num steps = %d", num_train_steps)
        
        ### train
        model.train()
        senti_l,rel_l=0,0
        for step,data in enumerate(tqdm(train_dataloader,desc="Iteration")):
            batch=data
            # Sentiment
            input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,label_index= post_dataloader(batch)
            outputs= model(input_ids=input_re_ids,
                                attention_mask     = input_re_attention_mask, 
                                input_ir_ids   = input_ir_ids, 
                                input_ir_attention_mask     = input_ir_attention_mask,                 
                                input_hidden_state    = input_hidden_states,
                                input_pooler_output=input_pooler_outputs,
                                labels=labels)
            output_loss=outputs['loss']
            output_loss.backward()
            lr_this_step = args.LEARNING_RATE * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
            for param_group in optimizer_t5.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_t5.step()
            optimizer_t5.zero_grad()
            global_step += 1
            senti_l+=output_loss.item()

        senti_l=senti_l/global_step
        logger.info("sentiment_loss:%s",senti_l)


        model.eval()
        logger.info("***** Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", val_dataset.number) #len(eval_examples)
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        dev_results=evaluate(model,val_dataloader,logger)
        logger.info("***** Dev Eval results *****")
        for key in sorted(dev_results.keys()):
            logger.info("  %s = %s", key, str(dev_results[key]))


        # save model
        if dev_results['senti_acc'] >= max_senti_acc:
            max_senti_acc=dev_results['senti_acc']
            best_epoch=train_idx
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("best_epoch: %d",best_epoch)

    logger.info("best_epoch: %d",best_epoch)

    model.eval()
    logger.info("***** Running evaluation on Test Set*****")
    logger.info("  Num examples = %d", test_dataset.number) #len(eval_examples)
    logger.info("  Batch size = %d", args.BATCH_SIZE)
    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model.load_state_dict(model_state_dict)
    test_results=evaluate(model,test_dataloader,logger)
    logger.info("***** Test Eval results *****")
    for key in sorted(test_results.keys()):
        logger.info("  %s = %s", key, str(test_results[key]))




if __name__ == "__main__":

    args=get_parser()
    main(args)