import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import json

def get_rel(relation):
    semantic_rel,emotional_rel=relation.split(',')
    semantic_relation_label,emotional_relation_label=0,0
    if 'irrelevant' in semantic_rel:
        semantic_relation_label=0
    else:
        semantic_relation_label=1

    if 'irrelevant' in emotional_rel:
        emotional_relation_label=0
    else:
        emotional_relation_label=1
   
    return semantic_relation_label,emotional_relation_label
class MyDataset(Data.Dataset):
    def __init__(self,args,data_dir,img_data_dir,tokenizer):
        self.tokenizer=tokenizer
        self.max_seq_len=args.max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.img_examples=self.creat_img_examples(img_data_dir)
        self.number = len(self.examples)

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        img_line=self.img_examples[index]
        return self.transform_three(line,img_line)   
        # return self.transform_three(line,img_line)   

    def creat_examples(self,data_dir):
        with open(data_dir,"r") as f:
            dict=json.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples

    def creat_img_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples
 
    def transform_three(self,line,img_line):
        max_input_len =self.max_seq_len
        input_hidden_states = img_line["hidden_state"]
        input_pooler_outputs = img_line["pooler_output"]
        value=line
        text = value["text"]
        aspect = value["aspect"]
        output_labels = value["label"]
        relation = value["relation"]
        text_clue= value["text_clue"]
        # aspect_Context= value["aspect_Context"]
        image_caption = value["image_caption"]
        image_emotion= value["image_emotion"]
        imagetext_meaning=value['imagetext_meaning']
        # text_emotion= value["text_emotion"]
        # text_description= value["text_description"]
        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        sentiment = sentiment_map[str(output_labels)]

        instruction_related = (
            "You are an expert in multimodal sentiment analysis. "
            "Determine the sentiment polarity (positive, neutral, or negative) for the given aspect, "
            "considering both text and image information if relevant."
        )
        
        semantic_relation_label,emotional_relation_label=get_rel(relation)
        # 2. 定义 relation_s 对应的文本说明
        relation_map = {
            0: "irrelevant (the image is unrelated to the aspect in both meaning and emotion)",
            1: "emotional relevant(the image conveys or enhances emotional expression toward the aspect)",
            2: "semantic relevant(the image provides semantic information related to the aspect)"
        }
        if emotional_relation_label == 1:
            relation_s = 1  # 情感相关
            unified_template_text = (
                f"{instruction_related} "
                f"Sentence: {text} "
                f"Aspect: {aspect} "
                f"Text Emotion: {text_clue or 'None'} "
                f"Image Emotion: {image_emotion} "
                f"Relation: {relation_map[relation_s]} "
                f"OPTIONS: -positive -neutral -negative OUTPUT:"
            )
        elif semantic_relation_label == 1:
            relation_s = 2  # 语义相关
            unified_template_text = (
                f"{instruction_related} "
                f"Sentence: {text} "
                f"Aspect: {aspect} "
                f"Text Emotion: {text_clue or 'None'} "
                f"Image Emotion: None "
                f"Relation: {relation_map[relation_s]} "
                f"OPTIONS: -positive -neutral -negative OUTPUT:"
            )
        else:
            relation_s = 0  # 无关
            unified_template_text = (
                f"{instruction_related} "
                f"Sentence: {text} "
                f"Aspect: {aspect} "
                f"Text Emotion: {text_clue or 'None'} "
                f"Image Emotion: None "
                f"Relation: {relation_map[relation_s]} "
                f"OPTIONS: -positive -neutral -negative OUTPUT:"
            )
 
       
        model_inputs = self.tokenizer(unified_template_text, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        # 获取 tokenized 的 input_ids 和 attention_mask
        input_ids = model_inputs["input_ids"].squeeze(0)
        input_attention_mask = model_inputs["attention_mask"].squeeze(0)

        # 处理标签
        model_inputs_labels = self.tokenizer(sentiment, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,output_labels,relation_s


 