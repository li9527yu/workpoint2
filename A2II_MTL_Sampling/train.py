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
from sklearn.metrics import f1_score
import re

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--Re_data_dir', default="/data/lzy1211/code/A2II/instructBLIP/relation_data", type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_feat", type=str)
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/A2II-MTL", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=6, type=int)
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

def post_Rel_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_ids,input_mask,img_id,img_feat,relation_label=batch    
    
    input_ids = input_ids.clone().detach().long().to(device)
    input_mask = input_mask.clone().detach().float().to(device)

    
    relation_label=relation_label.to(device).long()

    img_feat = img_feat.clone().detach().to(input_ids.device)

            
    return input_ids,input_mask,img_id,img_feat,relation_label


def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label,semantic_rel_label,emotion_rel_label=batch
      
    input_ids = input_ids.clone().detach().long().to(device)
    input_attention_mask = input_attention_mask.clone().detach().float().to(device)
    
    labels=labels.to(device).long()
    senti_label=senti_label.to(device).long()
    semantic_rel_label=semantic_rel_label.to(device).long()
    emotion_rel_label=emotion_rel_label.to(device).long()
    input_pooler_outputs = input_pooler_outputs.clone().detach().to(input_ids.device)
    input_hidden_states = input_hidden_states.clone().detach().to(input_ids.device)
      
    return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label,semantic_rel_label,emotion_rel_label

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def evaluate(model,test_dataloader,logger):
    pred_sequence=[]
    senti_labels,semantic_rel_labels,emotion_rel_labels=[],[],[]
    model.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label,semantic_rel_label,emotion_rel_label= post_dataloader(batch)
            with torch.no_grad():
                outputs= model.generate(input_ids=input_ids,
                            input_attention_mask=input_attention_mask,             
                            input_hidden_states    = input_hidden_states,
                            labels=labels)
                # 获取预测结果
                pred_sequence.extend(outputs)
                senti_labels.extend(senti_label.detach().cpu().numpy())
                semantic_rel_labels.extend(semantic_rel_label.detach().cpu().numpy())
                emotion_rel_labels.extend(emotion_rel_label.detach().cpu().numpy())

    senti_preds,semantic_rel_preds,emotion_rel_preds=parse_sequences(pred_sequence)
    senti_result = compute_metrics(senti_preds, senti_labels)
    semantic_rel_result = compute_metrics(semantic_rel_preds, semantic_rel_labels)
    emotion_rel_result = compute_metrics(emotion_rel_preds, emotion_rel_labels)

    result = {     
        'senti_acc':senti_result['acc'],    
        'senti_f1':senti_result['f1'],  
        'semantic_rel_acc':semantic_rel_result['acc'],     
        'semantic_rel_f1':semantic_rel_result['f1'],  
        'emotion_rel_acc':emotion_rel_result['acc'],    
        'emotion_rel_f1':emotion_rel_result['f1'],      
        }      
    return result


def simple_accuracy(preds, labels):
    return (preds == labels).mean()
def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }

def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)


def parse_sequences(pred_sequences):
    senti_preds,srel_preds,erel_preds= [],[],[]
    for seq in pred_sequences:
        seq = seq.lower().replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
       

        sentiment,semantic_rel,emotion_rel=seq.split('.')[0],seq.split('.')[1].split(',')[0],seq.split('.')[1].split(',')[1]

        # sentiment = sentiment_match.group(1).lower() if sentiment_match else "neutral"
        if 'negative' in sentiment:
            pred = 0
        elif 'positive' in sentiment:
            pred = 2
        else:
            pred = 1
        
        if 'irrelevant' in semantic_rel:
            srel_pred = 0
        else:
            srel_pred = 1
        
        if 'irrelevant' in emotion_rel:
            erel_pred = 0
        else:
            erel_pred = 1
        senti_preds.append(pred)
        srel_preds.append(srel_pred)
        erel_preds.append(erel_pred)

    return np.array(senti_preds),np.array(srel_preds),np.array(erel_preds)

# 获取相关性标签
def get_rel_data(dataset):
    path1=f'/data/lzy1211/code/A2II/instructBLIP/sentiment_relation_twitter_output/{dataset}/test_result.json'
    path2=f'/data/lzy1211/code/A2II/instructBLIP/sentiment_relation_twitter_output/{dataset}/train_result.json'

    
    with open(path1,'r') as f:
        rel1=json.load(f)
    f.close()
    with open(path2,'r') as f:
        rel2=json.load(f)
    f.close()

    rel=rel1+rel2
    return rel

def main(args):
    # wandb初始化
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="A2II-our-sweep",
    #     config=args,
    # )
    # args.BATCH_SIZE=wandb.config.BATCH_SIZE
    # args.LEARNING_RATE=wandb.config.LEARNING_RATE
    # # args.EPOCHS=wandb.config.EPOCHS
    # args.output_dir=f'{args.output_dir}/A2II-Our-{args.BATCH_SIZE}-{args.LEARNING_RATE}-{args.EPOCHS}/'

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
    train_data=f"{args.data_dir}/{args.dataset}/train.pkl"
    val_data=f"{args.data_dir}/{args.dataset}/val.pkl"
    test_data=f"{args.data_dir}/{args.dataset}/test.pkl"
    train_dataset=MyDataset(train_data,tokenizer,rel_data,max_seq_len= args.max_seq_length)
    val_dataset=MyDataset(val_data,tokenizer,rel_data,max_seq_len= args.max_seq_length)
    test_dataset=MyDataset(test_data,tokenizer,rel_data,max_seq_len= args.max_seq_length)

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
    Rel_global_step,global_step = 0,0

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
            input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label,semantic_rel_label,emotion_rel_label= post_dataloader(batch)
            outputs= model(input_ids=input_ids,
                            input_attention_mask=input_attention_mask,              
                            input_hidden_states    = input_hidden_states,
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
        # wandb.log(
        #     {
        #         "train_loss":senti_l
        #     })
        
        ### dev
        logger.info("***** Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", val_dataset.number) #len(eval_examples)
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        dev_result=evaluate(model,val_dataloader, logger)
        for key in sorted(dev_result.keys()):
                logger.info("  %s = %s", key, str(dev_result[key]))
        # wandb.log(
        #     {
        #     'senti_acc':dev_result['senti_acc'],    
        #     'senti_f1':dev_result['senti_f1'],  
        #     'semantic_rel_acc':dev_result['semantic_rel_acc'],     
        #     'semantic_rel_f1':dev_result['semantic_rel_f1'],  
        #     'emotion_rel_acc':dev_result['emotion_rel_acc'],    
        #     'emotion_rel_f1':dev_result['emotion_rel_f1'],     
        # })

        # 保存 模型
        # save model
        if dev_result['senti_acc'] >= max_senti_acc:
            max_senti_acc=dev_result['senti_acc']
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
    result=evaluate(model,test_dataloader, logger)
    for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    # wandb.log(
    #         {
    #         'senti_acc':result['senti_acc'],    
    #         'senti_f1':result['senti_f1'],  
    #         'semantic_rel_acc':result['semantic_rel_acc'],     
    #         'semantic_rel_f1':result['semantic_rel_f1'],  
    #         'emotion_rel_acc':result['emotion_rel_acc'],    
    #         'emotion_rel_f1':result['emotion_rel_f1'],     
    #     })




if __name__ == "__main__":
    # args=get_parser()
    # sweep_configuration = {
    #     "method": "grid",
    #     "name": "A2II-our-sweep",
    #     "metric": {"goal": "maximize", "name": "test_senti_acc"},
    #     "parameters": {
    #        "BATCH_SIZE": {"values": [8,16,32]},
    #         "LEARNING_RATE": {"values": [2e-5,5e-5,3e-5]},
    #     },
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="A2II-our-sweep")
    # wandb.agent(sweep_id, function=main(args), count=10)
    args=get_parser()
    main(args)