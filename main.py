import imp
import torch
import argparse
import os
from datasets import Dataset
from process_data.dataprocesser import MyDataset
import pickle
import json
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration,DataCollatorForSeq2Seq
from MyModel import MyFlanT5
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score


# 加载模型和分词器
model_path='/public/home/ghfu/lzy/model/flan-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = MyFlanT5.from_pretrained(model_path, device_map="auto")


def compute_metrics(p):
    predictions, labels = p
    # 解码生成的预测值
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 计算准确率
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2015', type=str, help=', ')
    parser.add_argument('--data_dir', default="/public/home/ghfu/lzy/code/instructBLIP/img_data", type=str)
    parser.add_argument('--img_feat_dir', default="/public/home/ghfu/lzy/code/instructBLIP/img_data", type=str)
    parser.add_argument('--output_dir', default="/public/home/ghfu/lzy/code/qwen2-VL/result", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=32, type=int)
    parser.add_argument('--seed', default=32, type=int)
    parser.add_argument('--EPOCHS', default=20, type=int)
    parser.add_argument('--LEARNING_RATE', default=5e-5, type=float)
    opt = parser.parse_args()

    return opt


def process_func(example):
    """
    将数据集进行预处理
    """
    max_input_len = 150
    max_label_len= 150
    # input_ids, attention_mask, labels = [], [], []
    # 相关和无关的instruction
    instruction_related="Definition:Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
    instruction_irrelvant="Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."

    input_aspect = example["aspect"]
    input_text = example["text"]
    input_hidden_state=example["hidden_state"]
    input_pooler_output=example["pooler_output"]
    output_label=example["sentiment"]

    input_re=f'{instruction_related} Sentence: {input_text} aspect: {input_aspect} OPTIONS: -positive -neutral -negative output:'
    input_ir=f'{instruction_irrelvant} Sentence: {input_text} aspect: {input_aspect} OPTIONS: -positive -neutral -negative output:'

    model_inputs = tokenizer(input_re,padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt").to("cuda")
    # input_re_ids=input_re['input_ids']
    # input_re_attention_mask=input_re['attention_mask']
    
    input_ir = tokenizer(input_ir,padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt").to("cuda")
    model_inputs['input_ir_ids']=input_ir['input_ids']
    model_inputs['input_ir_attention_mask']=input_ir['attention_mask']

    # input_ir_ids=input_ir['input_ids']
    # input_ir_attention_mask=input_ir['attention_mask']

    # input_query=input_query.to('cuda')
    model_inputs['input_hidden_state']=torch.tensor(input_hidden_state)
    model_inputs['input_pooler_output']=torch.tensor(input_pooler_output)
    # input_hidden_state=torch.tensor(input_hidden_state)
    # input_pooler_output=torch.tensor(input_pooler_output)
    model_inputs['label']= tokenizer(output_label,padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")['input_ids'].to("cuda")
    # model_inputs['labels'] = [
    #     -100 if label == tokenizer.pad_token_id else label
    #     for label in model_inputs['labels']
    # ]
    return model_inputs
    # output_label= tokenizer(output_label,padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")['input_ids'].to("cuda")
    # output_label[output_label == tokenizer.pad_token_id] = -100
    # return {"input_ids": input_re_ids,"attention_mask": input_re_attention_mask, \
    #         "input_ir_ids": input_ir_ids,"input_ir_attention_mask": input_ir_attention_mask, \
    #         "labels": output_label,
    #         "hidden_state": input_hidden_state,
    #         "pooler_output": input_pooler_output}

def process_func2(examples):
    """
    将数据集进行预处理，适用于批量处理
    """
    max_input_len = 150
    max_label_len = 150
    instruction_related = "Definition: Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
    instruction_irrelevant = "Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."

    # 从批量数据中提取信息
    input_texts = examples["text"]
    input_aspects = examples["aspect"]
    input_hidden_states = examples["hidden_state"]
    input_pooler_outputs = examples["pooler_output"]
    output_labels = examples["sentiment"]

    # 创建输入
    input_re = [f'{instruction_related} Sentence: {text} aspect: {aspect} OPTIONS: -positive -neutral -negative output:' for text, aspect in zip(input_texts, input_aspects)]
    input_ir = [f'{instruction_irrelevant} Sentence: {text} aspect: {aspect} OPTIONS: -positive -neutral -negative output:' for text, aspect in zip(input_texts, input_aspects)]

    # 对相关和无关的输入进行tokenize
    model_inputs_re = tokenizer(input_re, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
    model_inputs_ir = tokenizer(input_ir, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")

    # 获取 tokenized 的 input_ids 和 attention_mask
    input_re_ids = model_inputs_re["input_ids"]
    input_re_attention_mask = model_inputs_re["attention_mask"]
    input_ir_ids = model_inputs_ir["input_ids"]
    input_ir_attention_mask = model_inputs_ir["attention_mask"]

    # 将输入的 hidden_state 和 pooler_output 转换为 tensor
    input_hidden_states = torch.tensor(input_hidden_states).to(input_re_ids.device)  # 确保在同一个设备上
    input_pooler_outputs = torch.tensor(input_pooler_outputs).to(input_re_ids.device)  # 确保在同一个设备上

    # 处理标签
    model_inputs_labels = tokenizer(output_labels, padding='max_length', truncation=True, max_length=max_label_len, return_tensors="pt")
    labels = model_inputs_labels["input_ids"]

    # 将所有信息保存到字典中
    model_inputs = {
        "input_ids": input_re_ids,
        "attention_mask": input_re_attention_mask,
        "input_ir_ids": input_ir_ids,
        "input_ir_attention_mask": input_ir_attention_mask,
        "input_hidden_state": input_hidden_states,
        "input_pooler_output": input_pooler_outputs,
        "labels": labels  # label 可能需要处理成 -100, 见下文
    }

    # 标签处理 - 由于 Flan-T5 中，通常使用 -100 来标记 padding token，避免对其计算损失
    # model_inputs["labels"] = [
    #     [-100 if label == tokenizer.pad_token_id else label for label in batch] for batch in model_inputs["labels"]
    # ]
    model_inputs["labels"] = torch.tensor([
        [-100 if label == tokenizer.pad_token_id else label for label in batch]
        for batch in model_inputs["labels"]
    ], dtype=torch.long).to(input_re_ids.device)  # 确保在同一个设备上
    return model_inputs

def get_datasets(train_data):
    with open(train_data, "rb") as f:
        train_dataset = pickle.load(f)
    f.close()
    train_dataset = Dataset.from_dict({key: [entry[key] for entry in train_dataset] for key in train_dataset[0]})
    return train_dataset

def main(args):
    # 处理数据
    train_data=f"{args.data_dir}/{args.dataset}/train.pkl"
    val_data=f"{args.data_dir}/{args.dataset}/val.pkl"
    test_data=f"{args.data_dir}/{args.dataset}/test.pkl"

    train_dataset=get_datasets(train_data)
    val_dataset=get_datasets(val_data)
    # test_dataset=get_datasets(test_data)

    train_dataset = train_dataset.map(process_func2,batched=True)
    val_dataset = val_dataset.map(process_func2,batched=True)
    # test_dataset = test_dataset.map(process_func2,batched=True)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",                  # 输出目录
        num_train_epochs=1,  # 设置训练的轮次
        per_device_train_batch_size=2,  # 每个 GPU 上的 batch size
        gradient_accumulation_steps=8,  # 梯度累积
        evaluation_strategy="steps",  
        logging_dir='./logs',  # 日志文件夹
        logging_steps=100,  # 每隔多少步记录一次日志
        save_strategy="steps",
        save_steps=100,  # 每隔多少步保存一次模型
        load_best_model_at_end=True,  # 训练结束后加载最好的模型
        metric_for_best_model="accuracy",  # 评估最好的模型               # 如果度量越大越好（例如准确率）
        remove_unused_columns=False
    )

    # 假设我们已经有了预处理过的数据集
    # 创建 Trainer 实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        compute_metrics=compute_metrics,  # 计算指标
        
    )

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    args=get_parser()
    main(args)