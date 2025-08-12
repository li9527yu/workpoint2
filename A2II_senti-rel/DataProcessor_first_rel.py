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
        instruction_sentiment_related = "Definition: Determine the sentiment of a specific aspect mentioned in the sentence by integrating information from both the image and the sentence. The image provides essential emotional cues that help in accurately identifying the sentiment."
        instruction_irrelevant = "Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."

        input_texts = value["text"]
        input_aspects = value["aspect"]
        input_hidden_states = value["hidden_state"]
        input_pooler_outputs = value["pooler_output"]
        output_labels = value["sentiment"]

        filename_with_extension = value["image"].split('/')[-1]
        # 使用 '.' 分割，并取第一部分来去除扩展名
        input_img = filename_with_extension.split('.')[0]

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
        # inputs = f'{instruction_related} Sentence: {input_texts} aspect: {input_aspects} OPTIONS: -positive -neutral -negative output:' 
        
        inputs=''
        if relation=='the semantic relevance is relevant, the emotional relevance is irrelevant':  # 语义相关但情感无关
            inputs = f'{instruction_related} Sentence: {input_texts} Aspect: {input_aspects} OPTIONS: -positive -neutral -negative Output:' 
        elif relation=='the semantic relevance is relevant, the emotional relevance is relevant': # 语义相关且情感相关
            inputs = f'{instruction_sentiment_related} Sentence: {input_texts} Aspect: {input_aspects} OPTIONS: -positive -neutral -negative Output:'
        elif relation=='the semantic relevance is irrelevant, the emotional relevance is irrelevant': # 图文无关
            inputs = f'{instruction_irrelevant} Sentence: {input_texts} Aspect: {input_aspects} OPTIONS: -positive -neutral -negative Output:'


        

        model_inputs = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        

        # 获取 tokenized 的 input_ids 和 attention_mask
        input_ids = model_inputs["input_ids"].squeeze(0)
        input_attention_mask = model_inputs["attention_mask"].squeeze(0)

        # 处理标签
        model_inputs_labels = self.tokenizer(output_labels, padding='max_length', truncation=True, max_length=5, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels
