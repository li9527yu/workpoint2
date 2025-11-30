import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import random
import os
import logging
import torch
import re
import json
class MyDataset(Data.Dataset):
    def __init__(self,data_dir,reason_path,rel_data,task_weights,tokenizer,max_seq_len=128):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.examples=self.creat_examples(data_dir)
        # 
        dataset=data_dir.split('/')[-2]
        data_type=data_dir.split('/')[-1].split('.')[0]
        self.imgeExamples=self.creat_images_examples(os.path.join('/data/lzy1211/code/A2II/instructBLIP/aspect_context_imgFeat',dataset,f'{data_type}.pkl'))
        self.rel_data=rel_data
        self.reason_examples=self.create_reason_examples(reason_path)
        self.task_weights = task_weights or {
            "sentiment": 0.5,
            "reason": 0.3,
            "relevance": 0.2
        }
        self.task_types = list(self.task_weights.keys())
        self.task_probs = [self.task_weights[t] for t in self.task_types]
        total = sum(self.task_probs)
        self.task_probs = [p / total for p in self.task_probs]
        self.number = len(self.examples)

    def __len__(self):
        return self.number
    def get_rel_labels(self,input_img,input_aspects):
        relation=''
        for item in self.rel_data:
            img_index=item['conversations'][0]['value']
            item_aspect=item['conversations'][0]['aspect']
            rel=item['conversations'][0]['relation']
            # 使用正则表达式提取数字
            # match = re.search(r'/(\d+)\.jpg', img_index)
            # match = re.search(r'(\d{2}_\d{2}_\d{4})\.jpg', img_index)
            # if match:
            #     img_index=match.group(1)
            path_part = img_index.split("<|vision_start|>/")[1]
            # 第二次分割获取文件名（含扩展名）
            filename_with_ext = path_part.split("/")[-1]
            # 第三次分割去除扩展名
            img_index = filename_with_ext.split(".")[0]
            if img_index==input_img and input_aspects==item_aspect:
                relation=rel
                break
        return relation
    def __getitem__(self,index):
        line=self.examples[index]
        reason=self.reason_examples[index]
        imges=self.imgeExamples[index]
        return self.transform(line,reason,imges)   
    
    def creat_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples
    
    def creat_images_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatImageExample"):
            examples.append(item)
        return examples
    
    def create_reason_examples(self,data_dir):
        with open(data_dir,"r") as f:
            dict=json.load(f)
        examples=[]
        for item in tqdm(dict,desc="create_reason_examples"):
            examples.append(item)
        return examples
    def update_task_weights(self, new_task_weights):
        """更新任务采样权重"""
        self.task_weights = new_task_weights
        self.task_types = list(self.task_weights.keys())
        self.task_probs = [self.task_weights[t] for t in self.task_types]
        total = sum(self.task_probs)
        self.task_probs = [p / total for p in self.task_probs]
    
    def transform(self,line,reason,imges):
        max_input_len =self.max_seq_len
        value=line
        responses=reason
        imgs=imges
        input_texts = value["text"]
        input_aspects = value["aspect"]
        input_hidden_states = value["hidden_state"]
        # input_pooler_outputs = value["pooler_output"]
        output_labels = value["sentiment"]
        rc=responses['response']
        filename_with_extension = value["image"].split('/')[-1]
        # 使用 '.' 分割，并取第一部分来去除扩展名
        input_img = filename_with_extension.split('.')[0]
        relation_label= self.get_rel_labels(input_img,input_aspects)
        semantic_rel,emotional_rel=relation_label.split(',')
        # 动态选择任务
        task_type = random.choices(self.task_types, weights=self.task_probs)[0]
        # === 🧠 构造任务 ===
        if task_type == "sentiment":
            input_prompt =  (
                f"Analyze the informations below and images.\n"
                f"Text: '{input_texts}'\n"
                f"Aspect Information: '{input_texts}'\n"
                f"What is the sentiment of aspect {input_aspects} in this Text?\n"
            )
            target = f"The sentiment of this aspect is {output_labels}."

        elif task_type == "reason":
            input_prompt =  (
                f"Analyze the informations below and images.\n"
                f"Text: '{input_texts}'\n"
                f"What is the sentiment of aspect {input_aspects} in this Text?\n"
                f"Explain the reasoning behind the sentiment."
            )
            target = (
                f"The sentiment of this aspect is {output_labels}.\n"
                f"{rc}"
            )

        elif task_type == "relevance":
            input_prompt = (
                f"Analyze the informations below and images.\n"
                f"Text: '{input_texts}'\n"
                f"What is the sentiment of aspect {input_aspects} in this Text?\n"
                f"Explain the semantic relevance and emotional relevance of aspect {input_aspects}."
            )
            target = (
                f"The sentiment of this aspect is {output_labels}.\n"
                f"{relation_label}"
            )
        else:
            raise ValueError(f"Unsupported task: {task_type}")



        if 'irrelevant' in semantic_rel:
            semantic_rel_label=0
        else:
            semantic_rel_label=1
        if 'irrelevant' in emotional_rel:
            emotion_rel_label=0
        else:
            emotion_rel_label=1
        if 'positive' in output_labels:
            senti_label=2
        elif 'neutral' in output_labels:
            senti_label=1
        else:
            senti_label=0
        inputs = self.tokenizer(input_prompt, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        

        # # 获取 tokenized 的 input_ids 和 attention_mask
        # input_ids = model_inputs["input_ids"].squeeze(0)
        # input_attention_mask = model_inputs["attention_mask"].squeeze(0)

        # 处理标签
        model_inputs_labels = self.tokenizer(target, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["input_hidden_states"] = input_hidden_states
        inputs["labels"] = labels
        inputs["senti_label"] =torch.tensor(senti_label)
        inputs["semantic_rel_label"] =torch.tensor(semantic_rel_label)
        inputs["emotion_rel_label"] =torch.tensor(emotion_rel_label)
        inputs["task_type"] = task_type  # 可用于 loss logging

        return inputs
        # return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,task_type,labels,senti_label,semantic_rel_label,emotion_rel_label


def collate_fn(batch):
    batch_inputs = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "input_hidden_states": torch.stack([item["input_hidden_states"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "task_type": [item["task_type"] for item in batch],
        "senti_label": torch.stack([item["senti_label"] for item in batch]),
    }
    return batch_inputs