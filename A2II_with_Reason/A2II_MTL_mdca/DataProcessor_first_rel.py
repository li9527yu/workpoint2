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
from torch.nn.utils.rnn import pad_sequence
import json
class MyDataset(Data.Dataset):
    def __init__(self,data_dir,reason_path,rel_data,tokenizer,max_seq_len=128):
        dataset=data_dir.split('/')[-2]
        data_type=data_dir.split('/')[-1].split('.')[0]
        self.imgeExamples=self.creat_images_examples(os.path.join('/data/lzy1211/code/A2II/instructBLIP/aspect_context_imgFeat',dataset,f'{data_type}.pkl'))
        
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.rel_data=rel_data
        self.reason_examples=self.create_reason_examples(reason_path)
        
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

    def create_reason_examples(self,data_dir):
        with open(data_dir,"r") as f:
            dict=json.load(f)
        examples=[]
        for item in tqdm(dict,desc="create_reason_examples"):
            examples.append(item)
        return examples
    
    def creat_images_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatImageExample"):
            examples.append(item)
        return examples
    def get_input_sentence(self, sentence, aspect):
        a_input_sentence = (
                f"Analyze the informations below and images.\n"
                f"Text: '{sentence}'\n"
                f"What is the sentiment of aspect {aspect} in this Text?\n"
            )
        ea_input_sentence =  (
                f"Analyze the informations below and images.\n"
                f"Text: '{sentence}'\n"
                f"What is the sentiment of aspect {aspect} in this Text?\n"
                f"Explain the reasoning behind the sentiment."
            )
        iea_input_sentence = (
                f"Analyze the informations below and images.\n"
                f"Text: '{sentence}'\n"
                f"What is the sentiment of aspect {aspect} in this Text?\n"
                f"Explain the semantic relevance and emotional relevance of aspect {aspect}."
            )

        return a_input_sentence, ea_input_sentence, iea_input_sentence
    
    def get_output_sentence(self, aspect, output_labels, rc,relation_label):
        
        a_output_sentence =  f"The sentiment of this aspect {aspect}  is {output_labels}."

        ea_output_sentence = (
                f"The sentiment of this aspect {aspect}  is {output_labels}.\n"
                f"{rc}"
            )
        iea_output_sentence = (
                f"The sentiment of this aspect {aspect}  is {output_labels}.\n"
                f"{relation_label.strip()}"
            )

        return a_output_sentence, ea_output_sentence,iea_output_sentence

    def transform(self,line,reason,imges):
        max_input_len =self.max_seq_len
        value=line
        responses=reason
        imgs=imges
        input_texts = value["text"]
        input_aspects = value["aspect"]
        input_hidden_states = imgs["hidden_state"]
        # input_pooler_outputs = value["pooler_output"]
        output_labels = value["sentiment"]
        rc=responses['response']
        filename_with_extension = value["image"].split('/')[-1]
        # 使用 '.' 分割，并取第一部分来去除扩展名
        input_img = filename_with_extension.split('.')[0]
        relation_label= self.get_rel_labels(input_img,input_aspects)
        semantic_rel,emotional_rel=relation_label.split(',')
        # === 🧠 构造任务 ===
        a_input_sentence, ea_input_sentence, iea_input_sentence= self.get_input_sentence(input_texts, input_aspects)
        a_output_sentence, ea_output_sentence,iea_output_sentence= self.get_output_sentence(input_aspects,output_labels, rc,relation_label)
        

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
        a_inputs = self.tokenizer(a_input_sentence, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        ea_inputs = self.tokenizer(ea_input_sentence, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        iea_inputs = self.tokenizer(iea_input_sentence, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        # 处理标签
        a_labels = self.tokenizer(a_output_sentence, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        a_labels['input_ids'][a_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        a_labels = a_labels["input_ids"]

        ea_labels = self.tokenizer(ea_output_sentence, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        ea_labels['input_ids'][ea_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        ea_labels = ea_labels["input_ids"]
        iea_labels = self.tokenizer(iea_output_sentence, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        iea_labels['input_ids'][iea_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        iea_labels = iea_labels["input_ids"]

        inputs = {}
        
        inputs["a_input_ids"] = a_inputs['input_ids']
        inputs["a_attention_mask"] = a_inputs['attention_mask']
        # inputs["ea_input_ids"] = ea_inputs['input_ids']
        # inputs["ea_attention_mask"] = ea_inputs['attention_mask']
        inputs["iea_input_ids"] = iea_inputs['input_ids']
        inputs["iea_attention_mask"] = iea_inputs['attention_mask']
        inputs["input_hidden_states"] = input_hidden_states
        inputs["a_labels"] = a_labels
        # inputs["ea_labels"] = ea_labels
        inputs["iea_labels"] = iea_labels

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["senti_label"] =torch.tensor(senti_label)
        inputs["semantic_rel_label"] =torch.tensor(semantic_rel_label)
        inputs["emotion_rel_label"] =torch.tensor(emotion_rel_label)

        return inputs
        # return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,task_type,labels,senti_label,semantic_rel_label,emotion_rel_label


def collate_fn(batch):
    
    batch_inputs = {
        "a_input_ids": torch.stack([item["a_input_ids"] for item in batch]),
        "a_attention_mask": torch.stack([item["a_attention_mask"] for item in batch]),
        "a_decoder_output_labels": torch.stack([item["a_labels"] for item in batch]),
        'iea_input_ids':torch.stack([item["iea_input_ids"] for item in batch]),
        'iea_attention_mask':torch.stack([item["iea_attention_mask"] for item in batch]),
        'iea_decoder_output_labels':torch.stack([item["iea_labels"] for item in batch]),
        "input_hidden_states": torch.stack([item["input_hidden_states"] for item in batch]),
        "senti_label": torch.stack([item["senti_label"] for item in batch]),
        "semantic_rel_label": torch.stack([item["semantic_rel_label"] for item in batch]),
        "emotion_rel_label": torch.stack([item["emotion_rel_label"] for item in batch]),
    }
    return batch_inputs
