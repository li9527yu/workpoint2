import json
import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import logging
import torch
from torch import nn
from PIL import Image

class MyDataset(Data.Dataset):
    def __init__(self,data_dir,img_feat_dir,processor,vision_model,config,max_seq_len=64):
        self.processor=processor
        self.vision_model=vision_model
        self.blip_config=config
        self.max_seq_len=max_seq_len
        self.img_feats=self.get_img_feat(img_feat_dir)
        self.examples=self.creat_examples(data_dir)
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
    
    def transform(self,line):
        max_input_len =self.max_seq_len
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        value=line

        
        input_text = value["text"]
        input_aspect = value["aspect"]
        img_='/public/home/ghfu/lzy/data/twitter2017_images/'+value["ImageID"].split('/')[-1]
        image = Image.open(img_).convert("RGB")
        instruction=f'analyze the relevance of aspect {input_aspect} in the image and text: {input_text}.'
        inputs = self.processor(images=image, text=instruction, return_tensors="pt").to(device)
        vision_outputs = self.vision_model(pixel_values=inputs['pixel_values'])
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = nn.Parameter(torch.zeros(self.blip_config.num_query_tokens, self.blip_config.qformer_config.hidden_size)).to(device)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        qformer_attention_mask = torch.cat([query_attention_mask, inputs['qformer_attention_mask']], dim=1)
        output_labels = value["relation_s"]
        # instruction
        

        return inputs['qformer_input_ids'],qformer_attention_mask,query_tokens,image_embeds,image_attention_mask,output_labels
