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
from collections import defaultdict


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
        sentiment=seq.split('.')[0]
        if 'negative' in sentiment:
            pred = 0
        elif 'positive' in sentiment:
            pred = 2
        else:
            pred = 1
        

        senti_preds.append(pred)


    return np.array(senti_preds)



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


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x