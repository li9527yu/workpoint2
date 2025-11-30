import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import json
import numpy as np
import os
import logging
import torch
import re
class MyDataset(Data.Dataset):
    def __init__(self,data_dir,tokenizer,rel_data,img_data,max_seq_len=64):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.img_examples=self.creat_Imgexamples(img_data)
        self.rel_data=rel_data
        self.number = len(self.examples)

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        return self.transform(line)   

    def creat_examples(self,data_dir):
        with open(data_dir,"r") as f:
            dict=json.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples

    def creat_Imgexamples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples
    
    # 找到对应的图像特征
    def search_img(self,aspect,img):
        for item in self.img_examples:
            item_img=item['image'].split('/')[-1]
            if item['aspect']==aspect and item_img==img:
                return item['hidden_state']
        return None


    def search_relation(self,aspect,img):
        for item in self.rel_data:
            img_index=item['conversations'][0]['value']
            item_aspect=item['conversations'][0]['aspect']
            rel=item['conversations'][0]['relation']
            path_part = img_index.split("<|vision_start|>/")[1]
            
            # 第二次分割获取文件名（含扩展名）
            filename_with_ext = path_part.split("/")[-1]
            filename_with_ext=filename_with_ext.split("<|vision_end|>")[0]
            if filename_with_ext==img and aspect==item_aspect:
                return rel
        return None
    

    def transform(self,line):
        max_input_len =self.max_seq_len
        value=line
        instruction_related = "Definition: Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
        instruction_irrelevant = "Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."

        input_texts = value["text"]
        
        input_aspects = value["aspect"]
        input_texts=input_texts.replace('$T$',input_aspects)
        
        input_pooler_outputs = 0
        output_labels = value["sentiment"]
        label_map = {'2':"positive", '0':"negative", '1':"neutral"}
        label_index=int(output_labels)
        output_labels=label_map[output_labels]
        filename_with_extension = value["ImageID"].split('/')[-1]
        input_hidden_states = self.search_img(input_aspects,filename_with_extension)
        # text_clue = value["text_clue"]
        # image_emtoion = value["image_emtoion"]
        imagetext_meaning = value["imagetext_meaning"]
        # relation=self.search_relation(input_aspects,filename_with_extension)
        # inputs=''
        # if 'irrelvant' in relation:
        #     inputs = f'{instruction_irrelevant} Sentence: {input_texts} aspect: {input_aspects} OPTIONS: -positive -neutral -negative output:'
        # else:
        #     inputs = f'{instruction_related} Sentence: {input_texts} aspect: {input_aspects} OPTIONS: -positive -neutral -negative output:' 
        inputs_multi= f"{instruction_related} Image-Text Description: {imagetext_meaning} Sentence: {input_texts} aspect: {input_aspects}  OPTIONS: -positive -neutral -negative output:"
        model_inputs = self.tokenizer(inputs_multi, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        

        # 获取 tokenized 的 input_ids 和 attention_mask
        input_ids = model_inputs["input_ids"].squeeze(0)
        input_attention_mask = model_inputs["attention_mask"].squeeze(0)

        # 处理标签
        output_desc=f'the sentiment of {input_aspects} in the sentence is {output_labels}'
        model_inputs_labels = self.tokenizer(output_desc, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,label_index

    def transform_gold(self,line):
        max_input_len =self.max_seq_len
        value=line
        instruction_related = "Definition: Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
        instruction_irrelevant = "Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."

        input_texts = value["text"]
        
        input_aspects = value["aspect"]
        input_texts=input_texts.replace('$T$',input_aspects)
        input_relation = value["relation_s"]
        input_pooler_outputs = 0
        output_labels = value["sentiment"]
        label_map = {2:"positive", 0:"negative", 1:"neutral"}
        label_index=int(output_labels)
        output_labels=label_map[label_index]
        filename_with_extension = value["ImageID"].split('/')[-1]
        input_hidden_states = self.search_img(input_aspects,filename_with_extension)
        # relation=self.search_relation(input_aspects,filename_with_extension)
        inputs=''
        if input_relation == '0' and input_relation == '3':
            inputs = f'{instruction_irrelevant} Sentence: {input_texts} aspect: {input_aspects} OPTIONS: -positive -neutral -negative output:'
        else:
            inputs = f'{instruction_related} Sentence: {input_texts} aspect: {input_aspects} OPTIONS: -positive -neutral -negative output:' 
            

        model_inputs = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        

        # 获取 tokenized 的 input_ids 和 attention_mask
        input_ids = model_inputs["input_ids"].squeeze(0)
        input_attention_mask = model_inputs["attention_mask"].squeeze(0)

        # 处理标签
        model_inputs_labels = self.tokenizer(output_labels, padding='max_length', truncation=True, max_length=5, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,label_index
