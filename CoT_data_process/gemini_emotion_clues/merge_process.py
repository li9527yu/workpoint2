import base64
import requests
import json
from typing import List, Dict
import pickle
import os
from tqdm import tqdm
import base64, os, mimetypes
from google import genai
from google.genai import types
# ✅ 使用 Google GenAI SDK
from google import genai
from google.genai import types






def load_json_data(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_data(file_name, data):
    with open(file_name, "w") as f:
        json.dump(data, f,ensure_ascii=False,indent=2)




def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Generate entity descriptions using Gemini_Describer.")
    parser.add_argument("--dataset", type=str, default="twitter2017", help="Dataset name")
    parser.add_argument("--text_data_dir", type=str, default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/gemini_data_text", help="Dataset name")
    parser.add_argument("--img_data_dir", type=str, default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/gemini_data", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/merge_data", help="Dataset name")
    parser.add_argument("--img_dir", type=str, default="/data/lzy1211/code/twitterImage/", help="Dataset name")
    return parser.parse_args()


 

if __name__ == "__main__":
 
    args = parse_arguments()
    
    # 若文件夹不存在，创建
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # print(args) "train", "val",
    types_split = ["train", "val", "test"]
    for type in types_split:
        text_datas = load_json_data(os.path.join(args.text_data_dir, args.dataset, f"{type}.json"))
        img_datas = load_json_data(os.path.join(args.img_data_dir, args.dataset, f"{type}.json"))
        count = 0
        merge_results=text_datas
        for text_item,img_item in zip(merge_results,img_datas):
            text_item['img_clue']=img_item['img_clue']
            del text_item['image_emotion']
            del text_item['image_caption']
            del text_item['sentiment_reasoning']
            del text_item['imagetext_meaning']
            del text_item['text_clue']
        save_json_data(f"{args.output_dir}/{type}.json", merge_results)