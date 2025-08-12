import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import json
import torch
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
    def __init__(self,args,data_dir,img_data_dir,emotion_data,relation_data,tokenizer):
        self.tokenizer=tokenizer
        self.max_seq_len=args.max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.img_examples=self.creat_img_examples(img_data_dir)
        self.emotion_examples=self.creat_examples(emotion_data)
        self.relation_examples=self.creat_examples(relation_data)
        self.number = len(self.examples)

    def __len__(self):
        return self.number
    
    # 找到对应的图像特征
    def search_img(self,aspect,img):
        for item in self.img_examples:
            item_img=item['image'].split('/')[-1]
            if item['aspect']==aspect and item_img==img:
                return item['hidden_state']
        return None
    def search_emotion(self,aspect,img):
        for item in self.emotion_examples:
            if aspect==item['aspect'] and img==item['image']:
                return item['text_emotion'],item['image_emotion'],item['imagetext_meaning'],
        return None,None,None

    def search_relation(self,aspect,img):
        for x in self.relation_examples:
            item=x['conversations'][0]
            item_img=item['value'].split('<|vision_end|>')[0].split('<|vision_start|>')[1]
            item_img=item_img.split('/')[-1]
            if aspect==item['aspect'] and img==item_img:
                return item['relation'] 
        return None 
    
    def __getitem__(self,index):
        line=self.examples[index]
        return self.transform(line)   
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

    def transform(self,line):
        max_input_len =self.max_seq_len
        #原始数据 
        value=line
        text = value["text"]
        aspect = value["aspect"]
        text=text.replace('$T$',aspect)
        output_labels = value["sentiment"]
        # relation = value["relation_s"]
        # relation_label=torch.tensor()
        imgID=value['ImageID']
        imgID=imgID.split("/")[-1]
        text_clue = value["text_clue"]
        image_emtoion = value["image_emtoion"]
        imagetext_meaning = value["imagetext_meaning"]
        input_hidden_states = self.search_img(aspect,imgID)
        input_pooler_outputs=0
        # text_emotion,image_emotion,imagetext_meaning=self.search_emotion(aspect,imgID)
        sentiment_map = {'0': 'negative', '1': 'neutral', '2': 'positive'}
        sentiment = sentiment_map[output_labels]

        relation=self.search_relation(aspect,imgID)
        

        instruction_related = "Definition: Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
        instruction_irrelevant = "Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."
        
        # # relation:0:无关,1：来源,2：加强,3：相关
        # if relation=='1' or relation=='2':
        #     # inputs_multi= f"{instruction_related} Sentence: {text} aspect: {aspect} Description: {imagetext_meaning} OPTIONS: -positive -neutral -negative output:" 
        #     inputs=f"{instruction_related} Sentence: {text}  Image Emotion: {image_emtoion} Sentence Emotion: {text_clue} aspect: {aspect}  OPTIONS: -positive -neutral -negative output:"
        # elif relation=='3':
        #     # inputs_multi= f"{instruction_related} Sentence: {text} aspect: {aspect} Description: {imagetext_meaning} OPTIONS: -positive -neutral -negative output:" 
        #     inputs=f"{instruction_related} Sentence: {text} Sentence Emotion: {text_clue}  aspect: {aspect}  OPTIONS: -positive -neutral -negative output:"
        # else:
        #     # inputs_multi= f"{instruction_related} Sentence: {text} aspect: {aspect} Description: {imagetext_meaning} OPTIONS: -positive -neutral -negative output:" 
        #     inputs=f"{instruction_irrelevant} Sentence: {text} Sentence Emotion: {text_clue}  aspect: {aspect} OPTIONS: -positive -neutral -negative output:"


        # 使用银标相关性
        if relation=="the semantic relevance is relevant, the emotional relevance is relevant" :
            input_relation='1'
            inputs=f"{instruction_related} Sentence: {text}  Image Emotion: {image_emtoion} Sentence Emotion: {text_clue} aspect: {aspect}  OPTIONS: -positive -neutral -negative output:"
        elif relation=="the semantic relevance is relevant, the emotional relevance is irrelevant":
            input_relation='2'
            inputs=f"{instruction_related} Sentence: {text} Sentence Emotion: {text_clue}  aspect: {aspect}  OPTIONS: -positive -neutral -negative output:"
        else:
            input_relation='0'
            inputs=f"{instruction_irrelevant} Sentence: {text} Sentence Emotion: {text_clue}  aspect: {aspect} OPTIONS: -positive -neutral -negative output:"


        inputs_multi= f"{instruction_related} Sentence: {text} aspect: {aspect} Description: {imagetext_meaning} OPTIONS: -positive -neutral -negative output:"    

        model_inputs = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
 
        input_ids = model_inputs["input_ids"].squeeze(0)
        input_attention_mask = model_inputs["attention_mask"].squeeze(0)

        tokenize_inputs_multi = self.tokenizer(inputs_multi, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        # 获取 tokenized 的 input_ids 和 attention_mask
        input_multi_ids = tokenize_inputs_multi["input_ids"].squeeze(0)
        input_multi_attention_mask = tokenize_inputs_multi["attention_mask"].squeeze(0)



        # 处理标签
        model_inputs_labels = self.tokenizer(sentiment, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)
        # 
        return input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,input_hidden_states,input_pooler_outputs,labels,int(output_labels),int(input_relation)
    
