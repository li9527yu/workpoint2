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
from dataset_multiTask import MyDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration,DataCollatorForSeq2Seq
from T5_MultiTask import MyFlanT5
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import wandb
from itertools import cycle
from torch import nn
from trainer_multiTask import train, evaluate


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2015', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--caption_data_dir', default="/data/lzy1211/code/A2II/instructBLIP/A2II_multiTask", type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_feat", type=str)
    parser.add_argument('--pretrained_model_dir', default="/data/lzy1211/code/model/flan-t5-base", type=str)
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/multi_task/", type=str)
    parser.add_argument("--output_model_name",
                        default="pytorch_model",
                        type=str,
                        help="保存权重的名称")
    parser.add_argument("--output_log_name",
                        default="log",
                        type=str,
                        help="日志的名称")  
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=4, type=int)
    parser.add_argument('--cuda_id', default=6, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=5, type=int)
    parser.add_argument('--LEARNING_RATE', default=1e-5, type=float)
    parser.add_argument('--lamda', default=0.2, type=float)
    parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%  of training.")
    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--max_grad_norm",
                    default=1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%  of training.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_output_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--run_name",
                        default="pytorch_model",
                        type=str,
                        help="运行的名称")

    opt = parser.parse_args()

    return opt

def main(args):

    #set args
    args.output_dir=f'{args.output_dir}/test-{args.dataset}/'
    model_path='/data/lzy1211/code/model/flan-t5-base/'

    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_model_file = os.path.join(args.output_dir, f"{args.output_model_name}.bin")
    args.output_logger_file=os.path.join(args.output_dir,f'{args.output_log_name}.txt')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO,
                filename=args.output_logger_file)
    logger = logging.getLogger(__name__)

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    # 添加额外字符
    tokenizer.add_tokens(
        ['<image>', '</image>', '<relation>', '</relation>', '<emotion>', '</emotion>',
         'qa: ', 'qsra: ', 'qra: '])
    args.tokenizer = tokenizer
    # logger.info(args)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset=MyDataset(args, split='train')
    val_dataset=MyDataset(args, split='val')
    test_dataset=MyDataset(args, split='test')


    # Build Model
    model = MyFlanT5(args)
    model.to(args.device)

    # Train Model
    _, _, all_eval_results, best_model = train(args, train_dataset, model, val_dataset)
    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x['acc'])
        for key in sorted(best_eval_result.keys()):
            logger.info("  %s = %s", key, str(best_eval_result[key]))

    # Test
    test_results, _ = evaluate(args, test_dataset, best_model, True)
    logger.info("***** Test Results *****")
    for key in test_results.keys():
        logger.info("  %s = %s", key, str(test_results[key]))
    readme_path = os.path.join(args.output_dir, 'readme.txt')
    with open(readme_path, 'a+') as writer:
        writer.write('***** Test Results *****')
        writer.write('acc={}, f1={}'.format(test_results['acc'], test_results['f1']))
        writer.write('\n')


if __name__ == "__main__":

    args=get_parser()
    main(args)