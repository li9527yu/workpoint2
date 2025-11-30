import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import logging
import torch
import re
class MyDataset(Data.Dataset):
    def __init__(self,data_dir,tokenizer,rel_data,max_seq_len=64):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.rel_data=rel_data
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
        return self.transform(line)   
    
    def creat_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples


    
    def transform(self,line):
        max_input_len =self.max_seq_len
        value=line
        instruction_related = "Definition: Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
        instruction_irrelevant = "Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."

        input_texts = value["text"]
        input_aspects = value["aspect"]
        input_hidden_states = value["hidden_state"]
        input_pooler_outputs = value["pooler_output"]
        output_labels = value["sentiment"]

        filename_with_extension = value["image"].split('/')[-1]
        # 使用 '.' 分割，并取第一部分来去除扩展名
        input_img = filename_with_extension.split('.')[0]
        relation_label= self.get_rel_labels(input_img,input_aspects)
        semantic_rel,emotional_rel=relation_label.split(',')
        # inputs = f'{instruction_related} Sentence: {input_texts} aspect: {input_aspects} OPTIONS: -positive -neutral -negative output:' 
        input_prompt = (
            f"Analyze the informations below and images.\n"
            f"Text: '{input_texts}'\n"
            f"What is the sentiment of aspect {input_aspects} in this Text?\n"
            f" What is the semantic relevance and emotional relevance of aspect {input_aspects}?"
        )
        target = (
            f"The sentiment of this aspect is {output_labels}."
            f"{relation_label}"
        )
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
        model_inputs = self.tokenizer(input_prompt, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        

        # 获取 tokenized 的 input_ids 和 attention_mask
        input_ids = model_inputs["input_ids"].squeeze(0)
        input_attention_mask = model_inputs["attention_mask"].squeeze(0)

        # 处理标签
        model_inputs_labels = self.tokenizer(target, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,senti_label,semantic_rel_label,emotion_rel_label
