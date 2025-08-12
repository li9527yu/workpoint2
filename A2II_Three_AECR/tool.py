import json
import numpy as np
# pip install accelerate
from sklearn.metrics import f1_score

from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

def acc_and_f1_detailed(preds, labels, target_names,logger):
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

    # # 每类的 precision, recall, f1, support
    # precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)

    # # 打印整体指标
    # logger.info(f"Accuracy: {acc:.4f}")
    # logger.info(f"Macro F1: {macro_f1:.4f}\n")

    # # 打印每一类的指标
    # for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
    #     class_name = f"class_{i}" if target_names is None else target_names[i]
    #     logger.info(f"{class_name} - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}, Support: {s}")

    # 可选返回字典（如果你需要后续处理）
    result = {
        "acc": acc,
        "f1": macro_f1,
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


def compute_metrics(preds, labels,logger):
    # target_names = ["neutral","positive","negative", ]
    target_names = ["negative","neutral","positive" ]
    return acc_and_f1_detailed(preds, labels,target_names,logger)
    # return acc_and_f1(preds, labels)

 
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