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
from DataProcessor import MyDataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
# import wandb
from transformers import RobertaTokenizer, RobertaModel
from itertools import cycle
from torch import nn


# 标签映射
label_map = {"positive": 2, "negative": 0, "neutral": 1}
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2015', type=str, help=', ')
    parser.add_argument('--data_dir', default="/public/home/ghfu/lzy/code/instructBLIP/img_data", type=str)
    parser.add_argument('--Re_data_dir', default="/public/home/ghfu/lzy/code/instructBLIP/relation_data", type=str)
    parser.add_argument('--img_feat_dir', default="/public/home/ghfu/lzy/code/instructBLIP/img_feat", type=str)
    parser.add_argument('--output_dir', default="/public/home/ghfu/lzy/code/instructBLIP/results/twitter2015-a2ii-first_work-v2/", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=10, type=int)
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
    input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
    relation_input_mask,img_feat,relation_label=batch
    
    # RR
    relation_input_ids = relation_input_ids.clone().detach().long().to(device)
    relation_input_mask = relation_input_mask.clone().detach().float().to(device)
    
    relation_label=relation_label.to(device).long()

    img_feat = img_feat.clone().detach().to(relation_input_ids.device)

    # ss
        
    input_re_ids = input_re_ids.clone().detach().long().to(device)
    input_re_attention_mask = input_re_attention_mask.clone().detach().float().to(device)
    input_ir_ids = input_ir_ids.clone().detach().long().to(device)
    input_ir_attention_mask = input_ir_attention_mask.clone().detach().float().to(device)
    
    labels=labels.to(device).long()

    input_pooler_outputs = input_pooler_outputs.clone().detach().to(input_re_ids.device)
    input_hidden_states = input_hidden_states.clone().detach().to(input_re_ids.device)

            
    return input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
    relation_input_mask,img_feat,relation_label

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

# 手动裁剪每个序列，直到第一个 </s> 之前的位置
def crop_sequences(ids, eos_token_id):
    cropped_ids = []
    for seq in ids:
        eos_index = (seq == eos_token_id).nonzero(as_tuple=True)[0][0]  # 找到 </s> token 的位置
        if eos_index.numel() > 0:  # 如果找到了 </s> token
            cropped_ids.append(seq[:eos_index.item()])
        else:
            cropped_ids.append(seq)  # 如果没有找到 eos token，返回原始序列
    return torch.stack(cropped_ids)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def evaluate_on_test(model,test_dataloader,tokenizer,eos_token_id, logger):
    model.eval()
    sa_examples = 0
    test_senti_acc=0
    senti_true_label_list = []
    senti_pred_label_list = []
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
            relation_input_mask,img_feat,relation_label= post_dataloader(batch)
        with torch.no_grad():
            outputs= model(rel_inputs_id=relation_input_ids,
                        rel_inputs_mask=relation_input_mask,
                        img_feat=img_feat,
                        rel_label=relation_label,
                        input_ids=input_re_ids,
                        attention_mask     = input_re_attention_mask, 
                        input_ir_ids   = input_ir_ids, 
                        input_ir_attention_mask     = input_ir_attention_mask,                 
                        input_hidden_state    = input_hidden_states,
                        input_pooler_output=input_pooler_outputs,
                        labels=labels)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        cropped_predicted_ids = crop_sequences(predicted_ids, eos_token_id)
        cropped_labels = crop_sequences(labels, eos_token_id)
        predicted_labels = tokenizer.batch_decode(cropped_predicted_ids, skip_special_tokens=True)
        true_labels = tokenizer.batch_decode(cropped_labels, skip_special_tokens=True)
        senti_true_label_list.append(true_labels)
        senti_pred_label_list.append(predicted_labels)
        # 计算准确率
        test_senti_acc += sum([1 for pred, true in zip(predicted_labels, true_labels) if pred == true])
        sa_examples += labels.size(0)  

    test_senti_acc=test_senti_acc/sa_examples
    senti_true_label = np.concatenate(senti_true_label_list)
    senti_pred_outputs = np.concatenate(senti_pred_label_list)
    true_labels_numeric = [label_map[label] for label in senti_true_label]
    predictions_numeric = [label_map[label] for label in senti_pred_outputs]
    test_senti_precision, test_senti_recall, test_senti_F_score = macro_f1(true_labels_numeric, predictions_numeric)

    result = {
        'Test_senti_acc':test_senti_acc,
        'Test_senti_precision':test_senti_precision,
        'Test_senti_recall':test_senti_recall,
        'Test_senti_F_score':test_senti_F_score}
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    # wandb.log(
    #     {
    #         "test_acc": test_senti_acc,
    #         "test_macro_f1": test_senti_F_score,
    #         'Test_senti_precision':test_senti_precision,
    #         'Test_senti_recall':test_senti_recall,
    #     })


# 在相关性test上评测
def evaluate_on_Rel_test(model,Re_test_dataloader, logger):
    model.eval()
    rel_examples = 0
    rel_acc=0
    true_label_list = []
    pred_label_list = []
    # RELation
    for batch in tqdm(Re_test_dataloader, desc="Evaluating-Relation"):
        input_ids,input_mask,img_id,img_feat,relation_label=post_Rel_dataloader(batch)
        with torch.no_grad():
            Rel_score=model(rel_inputs_id=input_ids,
                            rel_inputs_mask=input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label)
            Rel_score=Rel_score.detach().cpu().numpy()
            relation_pred = np.argmax(Rel_score, axis=1)
            tmp_rel_accuracy=np.sum(relation_pred == relation_label.cpu().numpy()) 
            true_label_list.append(relation_label)
            pred_label_list.append(relation_pred)
            rel_acc += tmp_rel_accuracy
            rel_examples+= input_ids.size()[0]

    rel_acc = rel_acc/ rel_examples
    rel_true_label = np.concatenate(true_label_list)
    rel_pred_outputs = np.concatenate(pred_label_list)
    test_rel_precision, test_rel_recall, test_rel_F_score = macro_f1(rel_true_label, rel_pred_outputs)

    result = {
        'Test_rel_acc':rel_acc,
        'Test_rel_precision':test_rel_precision,
        'Test_rel_recall':test_rel_recall,
        'Test_rel_F_score':test_rel_F_score}
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

# q-former 相关性测试
def evaluate_on_Rel_test_QFormer(model,Re_test_dataloader, logger):
    model.eval()
    rel_examples = 0
    rel_acc=0
    true_label_list = []
    pred_label_list = []
    # RELation
    for batch in tqdm(Re_test_dataloader, desc="Evaluating-Relation"):
        input_ids,input_mask,img_id,img_feat,relation_label=post_Rel_dataloader(batch)
        with torch.no_grad():
            Rel_score=model(rel_inputs_id=input_ids,
                            rel_inputs_mask=input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label)
            Rel_score=Rel_score.detach().cpu().numpy()
            relation_pred = np.argmax(Rel_score, axis=1)
            tmp_rel_accuracy=np.sum(relation_pred == relation_label.cpu().numpy()) 
            true_label_list.append(relation_label)
            pred_label_list.append(relation_pred)
            rel_acc += tmp_rel_accuracy
            rel_examples+= input_ids.size()[0]

    rel_acc = rel_acc/ rel_examples
    rel_true_label = np.concatenate(true_label_list)
    rel_pred_outputs = np.concatenate(pred_label_list)
    test_rel_precision, test_rel_recall, test_rel_F_score = macro_f1(rel_true_label, rel_pred_outputs)

    result = {
        'Test_rel_acc':rel_acc,
        'Test_rel_precision':test_rel_precision,
        'Test_rel_recall':test_rel_recall,
        'Test_rel_F_score':test_rel_F_score}
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        

def main(args):
    # wandb初始化
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="A2II-RUN",
    #     config=args,
    #     name=args.run_name
    # )
    # args.BATCH_SIZE=wandb.config.BATCH_SIZE
    # args.LEARNING_RATE=wandb.config.LEARNING_RATE
    # args.EPOCHS=wandb.config.EPOCHS
    # args.output_dir=f'{args.output_dir}/A2II-RUN-{args.BATCH_SIZE}-{args.LEARNING_RATE}-{args.EPOCHS}/'

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

    # 处理数据
    # 相关性数据
    img_dir=f"/data/lzy1211/code/A2II/instructBLIP/img_feat/twitter2017.pkl"
    img_feat={}
    try:
        with open(img_dir, 'rb') as f:  # 以二进制模式打开文件
            img_feat = pickle.load(f)  # 反序列化
    except Exception as e:
        print(f"读取 .pkl 文件时出错: {e}")

    Re_train_data=f"{args.Re_data_dir}/train.json"
    Re_val_data=f"{args.Re_data_dir}/dev.json"
    # Re_test_data=f"{args.Re_data_dir}/test.json"
    Re_train_dataset=ReDataset(Re_train_data,tokenizer,img_feat, max_seq_len= args.max_seq_length)
    Re_val_dataset=ReDataset(Re_val_data,tokenizer,img_feat, max_seq_len= args.max_seq_length)
    # Re_test_dataset=ReDataset(Re_test_data,tokenizer,img_dir, max_seq_len= args.max_seq_length)

    Re_train_dataloader= Data.DataLoader(dataset=Re_train_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    Re_val_dataloader= Data.DataLoader(dataset=Re_val_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    # Re_test_dataloader= Data.DataLoader(dataset=Re_test_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)


    # sentiment数据
    img_dir=f"/data/lzy1211/code/A2II/instructBLIP/img_feat/{args.dataset}.pkl"
    img_feat={}
    try:
        with open(img_dir, 'rb') as f:  # 以二进制模式打开文件
            img_feat = pickle.load(f)  # 反序列化
    except Exception as e:
        print(f"读取 .pkl 文件时出错: {e}")
    train_data=f"{args.data_dir}/{args.dataset}/train.pkl"
    val_data=f"{args.data_dir}/{args.dataset}/val.pkl"
    test_data=f"{args.data_dir}/{args.dataset}/test.pkl"
    train_dataset=MyDataset(train_data,tokenizer,img_feat,max_seq_len= args.max_seq_length)
    val_dataset=MyDataset(val_data,tokenizer,img_feat,max_seq_len= args.max_seq_length)
    test_dataset=MyDataset(test_data,tokenizer,img_feat,max_seq_len= args.max_seq_length)

    train_dataloader= Data.DataLoader(dataset=train_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    val_dataloader= Data.DataLoader(dataset=val_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    test_dataloader= Data.DataLoader(dataset=test_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)

    train_number=train_dataset.number
    num_train_steps = int( train_number / args.BATCH_SIZE * args.EPOCHS)

    model = MyFlanT5(model_path=t5_model)
    model.to(device)

    # Prepare optimizer
    # optimizer BertAdam
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
   
    optimizer_rel = AdamW(optimizer_grouped_parameters,
                            lr=args.Rel_LEARNING_RATE)
    optimizer_t5 = AdamW(optimizer_grouped_parameters,
                            lr=args.LEARNING_RATE)
    max_senti_acc = 0.0
    best_epoch=-1
    Rel_global_step,global_step = 0,0

    # train
    logger.info("*************** Running training ***************")
    for train_idx in trange(int(args.EPOCHS), desc="Epoch"):
        logger.info("************************************************** Epoch: "+ str(train_idx) + " *************************************************************")
        logger.info("  Num examples = %d",  train_number) 
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        logger.info("  Num steps = %d", num_train_steps)
        
        ### train
        model.train()
        senti_l,rel_l=0,0
        for step,data in enumerate(tqdm(zip(cycle(Re_train_dataloader),train_dataloader),desc="Iteration")):
            Rel_batch,batch=data

            # RELation
            input_ids,input_mask,img_id,img_feat,relation_label=post_Rel_dataloader(Rel_batch)
            Rel_score=model(rel_inputs_id=input_ids,
                            rel_inputs_mask=input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label)
            Rel_loss_func=nn.CrossEntropyLoss()
            Rel_loss=Rel_loss_func(Rel_score,relation_label.long())
            Rel_loss.backward()    
            lr_this_step = args.Rel_LEARNING_RATE * warmup_linear(Rel_global_step/num_train_steps, args.warmup_proportion)
            for param_group in optimizer_rel.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_rel.step()
            optimizer_rel.zero_grad()
            Rel_global_step+=1
            rel_l+=Rel_loss.item()


            # Sentiment
            input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
            relation_input_mask,img_feat,relation_label= post_dataloader(batch)
            outputs= model(rel_inputs_id=relation_input_ids,
                            rel_inputs_mask=relation_input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label,
                            input_ids=input_re_ids,
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

        rel_l=rel_l/Rel_global_step
        senti_l=senti_l/global_step
        logger.info("relation_loss:%s",rel_l)
        logger.info("sentiment_loss:%s",senti_l)

        model.eval()
        logger.info("***** Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", val_dataset.number) #len(eval_examples)
        logger.info("  Batch size = %d", args.BATCH_SIZE)


        rel_examples,sa_examples = 0,0
        rel_acc,senti_acc=0,0

        # RELation
        for batch in tqdm(Re_val_dataloader, desc="Evaluating-Relation"):
            input_ids,input_mask,img_id,img_feat,relation_label=post_Rel_dataloader(batch)
            with torch.no_grad():
                Rel_score=model(rel_inputs_id=input_ids,
                                rel_inputs_mask=input_mask,
                                img_feat=img_feat,
                                rel_label=relation_label)
                Rel_score=Rel_score.detach().cpu().numpy()
                relation_pred = np.argmax(Rel_score, axis=1)
                tmp_rel_accuracy=np.sum(relation_pred == relation_label.cpu().numpy()) 
                rel_acc += tmp_rel_accuracy
                rel_examples+= input_ids.size()[0]

        rel_acc = rel_acc/ rel_examples

        # Sentiment
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
            relation_input_mask,img_feat,relation_label= post_dataloader(batch)
            with torch.no_grad():
                outputs= model(rel_inputs_id=relation_input_ids,
                            rel_inputs_mask=relation_input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label,
                            input_ids=input_re_ids,
                            attention_mask     = input_re_attention_mask, 
                            input_ir_ids   = input_ir_ids, 
                            input_ir_attention_mask     = input_ir_attention_mask,                 
                            input_hidden_state    = input_hidden_states,
                            input_pooler_output=input_pooler_outputs,
                            labels=labels)
                 # 获取预测结果
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                cropped_predicted_ids = crop_sequences(predicted_ids, eos_token_id)
                cropped_labels = crop_sequences(labels, eos_token_id)
                predicted_labels = tokenizer.batch_decode(cropped_predicted_ids, skip_special_tokens=True)
                true_labels = tokenizer.batch_decode(cropped_labels, skip_special_tokens=True)
                # 计算准确率
                senti_acc += sum([1 for pred, true in zip(predicted_labels, true_labels) if pred == true])
                sa_examples += labels.size(0)

                
        senti_acc=senti_acc/sa_examples
        result = {
            'rel_examples':rel_examples,         
            'Dev_rel_acc':rel_acc, 
            'sa_examples':sa_examples,         
            'Dev_senti_acc':senti_acc,             
            }        
        logger.info("***** Dev Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        # wandb.log(
        #     {
        #         "val_acc": senti_acc
        #     })
        # 保存 模型
                # save model
        if senti_acc >= max_senti_acc:
            max_senti_acc=senti_acc
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
    evaluate_on_test(model,test_dataloader,tokenizer,eos_token_id, logger)




if __name__ == "__main__":
    # args=get_parser()
    # sweep_configuration = {
    #     "method": "grid",
    #     "name": "A2II-sweep",
    #     "metric": {"goal": "maximize", "name": "test_senti_F_score"},
    #     "parameters": {
    #         "BATCH_SIZE": {"values": [8,16,4]},
    #         "LEARNING_RATE": {"values": [1e-4,1e-5,1e-3]},
    #         "EPOCHS": {"values": [10,20]}
    #     },
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="A2II-sweep")
    # wandb.agent(sweep_id, function=main(args), count=10)
    args=get_parser()
    main(args)