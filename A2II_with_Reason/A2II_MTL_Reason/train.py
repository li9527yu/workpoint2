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
from T5_Model import MyFlanT5
from DataProcessor_first_rel import MyDataset,collate_fn
import wandb
from tool import warmup_linear,get_rel_data,parse_sequences,compute_metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--Re_data_dir', default="/data/lzy1211/code/A2II/instructBLIP/relation_data", type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_feat", type=str)
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/A2II-curriculum-aspectContext", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', type=str, default='cuda:7', 
                      help='指定GPU设备，例如 cuda:0, cuda:1')
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


        
def evaluate(model,test_dataloader,logger):
    pred_sequence=[]
    senti_labels,semantic_rel_labels,emotion_rel_labels=[],[],[]
    model.eval()
    device = torch.device(args.device)
    for batch in tqdm(test_dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_hidden_states", "input_ids","input_hidden_states", "attention_mask", "labels"]}
            senti_label = batch["senti_label"]  # 字符串列表，比如 ['sentiment_reason', 'alignment', ...]
            # Sentiment
            # input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label= post_dataloader(batch)
            
            with torch.no_grad():
                outputs= model.generate(**inputs)
                pred_sequence.extend(outputs)
                senti_labels.extend(senti_label.detach().cpu().numpy())
               

    senti_preds=parse_sequences(pred_sequence)
    senti_result = compute_metrics(senti_preds, senti_labels)
    result = {     
        f'senti_acc':senti_result['acc'],    
        f'senti_f1':senti_result['f1'],  
   
    }      
    return result


def train_multitask(
    args,
    model,
    train_dataset,
    val_dataset,
    optimizer_t5,
    logger,
    curriculum_schedule=None):
    
    # 0. 参数设置
    max_senti_acc = 0.0
    best_epoch=-1
    Rel_global_step,global_step = 0,0

    # 1. 准备 dataloader
    device = torch.device(args.device)
    train_number=train_dataset.number
    num_train_steps = int( train_number / args.BATCH_SIZE * args.EPOCHS)
    train_dataloader= Data.DataLoader(dataset=train_dataset,shuffle=True, batch_size= args.BATCH_SIZE,collate_fn=collate_fn)  # 需要自己定义 collate_fn，处理不同长度
    val_dataloader= Data.DataLoader(dataset=val_dataset,shuffle=True, batch_size= args.BATCH_SIZE,collate_fn=collate_fn)
    # # train
    logger.info("*************** Running training ***************")
    for train_idx in trange(args.EPOCHS, desc="Epoch"):
        logger.info("************************************************** Epoch: "+ str(train_idx) + " *************************************************************")
        logger.info("  Num examples = %d",  train_number) 
        logger.info("  Batch size = %d",  args.BATCH_SIZE)
        logger.info("  Num steps = %d", num_train_steps)
        total_loss = 0.0

        # # 用来单独记录每个任务的 loss
        # task_losses = {"sentiment": 0.0, "relevance": 0.0, "reason": 0.0}
        # task_counts = {"sentiment": 0, "relevance": 0, "reason": 0}

        # 2. curriculum 动态调整权重
        if curriculum_schedule and train_idx in curriculum_schedule:
            new_weights = curriculum_schedule[train_idx]
            logger.info(f"📈 Updating task sampling weights: {new_weights}")
            train_dataset.update_task_weights(new_weights)

        ### train
        model.train()
        for step,data in enumerate(tqdm(train_dataloader,desc="Iteration")):
            batch=data
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_hidden_states", "input_ids","input_hidden_states", "attention_mask", "labels"]}
            task_types = batch["task_type"]  # 字符串列表，比如 ['sentiment_reason', 'alignment', ...]

            outputs= model(**inputs, task_type=task_types)
            output_loss=outputs['loss']
            output_loss.backward()
            lr_this_step = args.LEARNING_RATE * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
            for param_group in optimizer_t5.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_t5.step()
            optimizer_t5.zero_grad()
            global_step += 1
            total_loss += output_loss.item()

            # 单独记录不同任务 loss
            # for task in task_types:
            #     task_losses[task] += output_loss.item()
            #     task_counts[task] += 1
        
        # 4. 每个任务的平均 loss
        # avg_losses = {task: task_losses[task]/max(task_counts[task],1) for task in task_losses}

        logger.info(f"🔥 Epoch {train_idx+1} Losses:")
        logger.info(f"total loss: {total_loss/global_step:.4f}")
        # for task, avg_loss in avg_losses.items():
        #     logger.info(f"   {task}: {avg_loss:.4f}") 
        
        ### dev
        logger.info("***** Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", val_dataset.number) #len(eval_examples)
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        dev_result=evaluate(model,val_dataloader, logger)
        for key in sorted(dev_result.keys()):
                logger.info("  %s = %s", key, str(dev_result[key]))

        # 保存 模型
        # save model
        if dev_result['senti_acc'] >= max_senti_acc:
            max_senti_acc=dev_result['senti_acc']
            best_epoch=train_idx
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  
            torch.save(model_to_save.state_dict(), args.output_model_file)
            logger.info("best_epoch: %d",best_epoch)

    logger.info("best_epoch: %d",best_epoch)


def main(args):

    model_path='/data/lzy1211/code/model/flan-t5-base/'
    device = torch.device(args.device)
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
    t5_model=T5ForConditionalGeneration.from_pretrained(model_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(args)
    task_weights={
            "sentiment":1,
            "reason": 0,
            "relevance": 0
        }
    val_task_weights={
            "sentiment":1,
            "reason": 0,
            "relevance": 0
        }
    rel_data=get_rel_data(args.dataset)
    train_data=f"{args.data_dir}/{args.dataset}/train.pkl"
    val_data=f"{args.data_dir}/{args.dataset}/val.pkl"
    test_data=f"{args.data_dir}/{args.dataset}/test.pkl"
    train_dataset=MyDataset(train_data,f'/data/lzy1211/code/A2II/instructBLIP/reason_data/data/{args.dataset}/train_cause.json',rel_data,task_weights,tokenizer,max_seq_len= args.max_seq_length)
    val_dataset=MyDataset(val_data,f'/data/lzy1211/code/A2II/instructBLIP/reason_data/data/{args.dataset}/dev_cause.json',rel_data,val_task_weights,tokenizer,max_seq_len= args.max_seq_length)
    test_dataset=MyDataset(test_data,f'/data/lzy1211/code/A2II/instructBLIP/reason_data/data/{args.dataset}/test_cause.json',rel_data,val_task_weights,tokenizer,max_seq_len= args.max_seq_length)
    
    test_dataloader= Data.DataLoader(dataset=test_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)

    model = MyFlanT5(model_path=t5_model,tokenizer=tokenizer)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer_t5 = AdamW(optimizer_grouped_parameters,
                            lr=args.LEARNING_RATE)

    curriculum_schedule = {
        0: {"sentiment": 1.0},
        3: {"sentiment": 0.4, "relevance": 0.4, "reason": 0.2 },
        6: {"sentiment": 0.2, "relevance": 0.3, "reason": 0.5},
    }
    # #  {
    #     0: {"sentiment": 0.7, "relevance": 0.2, "reason": 0.1},
    #     3: {"sentiment": 0.4, "relevance": 0.3, "reason": 0.3},
    #     6: {"sentiment": 0.2, "relevance": 0.3, "reason": 0.5},
    # }
    # # train
    train_multitask(args,model,train_dataset,val_dataset,optimizer_t5,logger,curriculum_schedule)
    
    model.eval()
    logger.info("***** Running evaluation on Test Set*****")
    logger.info("  Num examples = %d", test_dataset.number) #len(eval_examples)
    logger.info("  Batch size = %d", args.BATCH_SIZE)
    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(args.output_model_file)
    model.load_state_dict(model_state_dict)
    result=evaluate(model,test_dataloader, logger)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__":
    args=get_parser()
    main(args)