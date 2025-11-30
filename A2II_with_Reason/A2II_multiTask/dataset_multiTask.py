import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import logging
import torch
import re
import json
from torch.nn.utils.rnn import pad_sequence
from utils import read_json

class MyDataset(Data.Dataset):
    def __init__(self,args,split):
        self.args = args
        self.max_seq_len=self.args.max_seq_length
        self.data_path = os.path.join(args.data_dir, args.dataset)
        self.caption_data_dir = os.path.join(args.caption_data_dir, args.dataset)
        self.captions_path = os.path.join(self.caption_data_dir, 'captions.json')
        self.caption_data = self.get_captions(self.captions_path)

        if split == 'train':
            data_path=f"{self.data_path}/train.pkl"
            self.examples=self.create_examples(data_path)
        elif split == 'val':
            data_path=f"{self.data_path}/val.pkl"
            self.examples=self.create_examples(data_path)
        elif split == 'test':
            data_path=f"{self.data_path}/test.pkl"
            self.examples=self.create_examples(data_path)
        else:
            raise RuntimeError("split type is not exist!!!")
        
        self.rel_data=self.get_rel_data(args.dataset)
        self.number = len(self.examples)

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        return self.transform(line)   

    def create_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="create_examples"):
            examples.append(item)
        return examples
    
    def get_captions(self, data_path):
        data = read_json(data_path)
        return data
    
    def get_rel_data(self,dataset):
        path1=f'/data/lzy1211/code/gpt=api/{dataset}/test_process.json'
        path2=f'/data/lzy1211/code/gpt=api/{dataset}/train_process.json'

        with open(path1,'r') as f:
            rel1=json.load(f)
        f.close()
        with open(path2,'r') as f:
            rel2=json.load(f)
        f.close()

        rel=rel1+rel2
        return rel
    
    def search_rel_item(self,input_aspects,input_img):
        relation=''
        relation_label=-1
        for item in self.rel_data:
            img_index=item['conversations'][0]['value']
            item_aspect=item['conversations'][0]['aspect']
            rel=item['conversations'][0]['relation']
            # 使用正则表达式提取数字
            # match = re.ssrarch(r'/(\d+)\.jpg', img_index)
            # match = re.ssrarch(r'(\d{2}_\d{2}_\d{4})\.jpg', img_index)
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
        if 'irrelvant' in relation:
            relation_label=0
        else:
            relation_label=1
        return relation,relation_label
    
    def get_input_sentence(self, sentence, aspect, caption):
        a_instruction="qa: Combining information from image and the following sentence to identify the sentiment of aspect."
        sra_instruction="qsra: Combining information from image and the following sentence to identify the relevance and sentiment of aspect."
        ra_instruction="qra: Combining information from image and the following sentence to identify the relevance of aspect."
        a_input_sentence = f"{a_instruction}  Image Caption: '{caption}'. Sentence: '{sentence}' Aspect: '{aspect}' "
        sra_input_sentence = f"{sra_instruction} Image Caption: '{caption}'. Sentence: '{sentence}' Aspect: '{aspect}' "
        ra_input_sentence = f"{ra_instruction} Image Caption: '{caption}'. Sentence: '{sentence}' Aspect: '{aspect}' "
        return a_input_sentence, sra_input_sentence, ra_input_sentence
    
    def get_output_sentence(self, label, relation):
        
        a_output_sentence = '<emotion>{}</emotion>'.format(label)
        sra_output_sentence = '<relation>{}</relation><emotion>{}</emotion>'.format(relation, label)
        ra_output_sentence = '<relation>{}</relation>'.format(relation)
        return a_output_sentence, sra_output_sentence, ra_output_sentence

    def transform(self,line):
        max_input_len =self.max_seq_len
        value=line

        input_texts = value["text"]
        input_aspects = value["aspect"]
        input_hidden_states = value["hidden_state"]
        # input_pooler_outputs = value["pooler_output"]
        output_labels = value["sentiment"]
        sentiment_map = {'neutral':1, 'positive':2, 'negative':0}
        sentiment = sentiment_map[output_labels]
        filename_with_extension = value["image"].split('/')[-1]
        # 使用 '.' 分割，并取第一部分来去除扩展名
        input_img = filename_with_extension.split('.')[0]

        captions = self.caption_data[filename_with_extension]  
        relation,relation_label=self.search_rel_item(input_aspects,input_img)
        
        a_input_sentence, sra_input_sentence, ra_input_sentence = self.get_input_sentence(input_texts, input_aspects, captions[3])
        a_output_sentence, sra_output_sentence, ra_output_sentence = self.get_output_sentence(output_labels, relation)
        

        a_inputs=self.args.tokenizer(
            a_input_sentence,
            truncation=True,
            padding='max_length',
            max_length=max_input_len,
            return_attention_mask=True,
            add_special_tokens=True,  # 关键参数（自动加[SEP]/[CLS]等）
            return_tensors="pt"
        )
        sra_inputs=self.args.tokenizer(
            sra_input_sentence,
            truncation=True,
            padding='max_length',
            max_length=max_input_len,
            return_attention_mask=True,
            add_special_tokens=True,  # 关键参数（自动加[SEP]/[CLS]等）
            return_tensors="pt"
        )
        ra_inputs=self.args.tokenizer(
            ra_input_sentence,
            truncation=True,
            padding='max_length',
            max_length=max_input_len,
            return_attention_mask=True,
            add_special_tokens=True,  # 关键参数（自动加[SEP]/[CLS]等）
            return_tensors="pt"
        )

        a_output_labels = self.args.tokenizer(
                a_output_sentence,
                truncation=True,
                padding='max_length',
                max_length=self.args.max_output_length,
                add_special_tokens=True,  # 自动添加 EOS
                return_tensors="pt"
            ).input_ids
        sra_output_labels = self.args.tokenizer(
                sra_output_sentence,
                truncation=True,
                padding='max_length',
                max_length=self.args.max_output_length,
                add_special_tokens=True,  # 自动添加 EOS
                return_tensors="pt"
            ).input_ids
        ra_output_labels = self.args.tokenizer(
                ra_output_sentence,
                truncation=True,
                padding='max_length',
                max_length=self.args.max_output_length,
                add_special_tokens=True,  # 自动添加 EOS
                return_tensors="pt"
            ).input_ids

        return (a_inputs['input_ids'].squeeze(0), a_inputs["attention_mask"].squeeze(0), a_output_labels.squeeze(0),
                sra_inputs['input_ids'].squeeze(0), sra_inputs["attention_mask"].squeeze(0), sra_output_labels.squeeze(0),
                ra_inputs['input_ids'].squeeze(0), ra_inputs["attention_mask"].squeeze(0), ra_output_labels.squeeze(0),
                input_hidden_states, torch.tensor(relation_label),torch.tensor(sentiment))
    

def collate_fn_flant5(batch):
    '''
    Pad sentence a batch.
    Turn all into tensors.
    '''
    a_input_ids, a_attention_mask, a_output_labels, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature,relation_label, sentiment_labels = zip(*batch)
 
    a_input_ids = pad_sequence(a_input_ids, batch_first=True, padding_value=0)
    a_output_labels = pad_sequence(a_output_labels, batch_first=True, padding_value=-100)
    ea_input_ids = pad_sequence(ea_input_ids, batch_first=True, padding_value=0)
    ea_output_labels = pad_sequence(ea_output_labels, batch_first=True, padding_value=-100)
    iea_input_ids = pad_sequence(iea_input_ids, batch_first=True, padding_value=0)
    iea_output_labels = pad_sequence(iea_output_labels, batch_first=True, padding_value=-100)

    image_feature = pad_sequence(image_feature, batch_first=True, padding_value=0)

    a_attention_mask = pad_sequence(a_attention_mask, batch_first=True, padding_value=0)
    ea_attention_mask = pad_sequence(ea_attention_mask, batch_first=True, padding_value=0)
    iea_attention_mask = pad_sequence(iea_attention_mask, batch_first=True, padding_value=0)

    sentiment_labels = torch.tensor(sentiment_labels)
    relation_label = torch.tensor(relation_label)

    return a_input_ids, a_attention_mask, a_output_labels, ea_input_ids, ea_attention_mask, ea_output_labels, iea_input_ids, iea_attention_mask, iea_output_labels, image_feature,relation_label, sentiment_labels
     
