import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import logging

from transformers import RobertaTokenizer, RobertaModel

label_to_text = {0: "negative", 1: "neutral", 2: "positive"}

class MyDataset(Data.Dataset):
    def __init__(self,data_dir,imagefeat_dir,processor,max_seq_len=64):
        self.examples=self.creat_examples(data_dir)
        self.processer=processor
        self.number = len(self.examples)

    def creat_examples(self,data_dir):
        with open(data_dir, 'rb') as f:
            examples = pickle.load(f)
        return examples

        
    def __len__(self):
        return self.number
    def __getitem__(self,index):
        item=self.examples[index]
        sentence=item['text']
        aspect=item['aspect']
        instruction_related="Definition:Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
        instruction_irrelvant="Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."
        input_related=f'{instruction_related} Sentence: {sentence} aspect: {aspect} OPTIONS: -positive -neutral -negative output:'
        input_irrelvant=f'{instruction_irrelvant} Sentence: {sentence} aspect: {aspect} OPTIONS: -positive -neutral -negative output:'


        # Definition:Combining information from image and thefollowing sentence to identify the sentiment ofaspect in the sentence.
        # Sentenee:# Bears coach John Fox meets the Dolphins after his first game , a preseason win :
        # aspeet: John Fox
        # OPTIONS: -positive -neutral -negativeoutput: