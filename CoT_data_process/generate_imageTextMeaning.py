import base64
import requests
import json
from typing import List, Dict
import pickle
import os
from openai import OpenAI
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoConfig
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-da8e2b49bc49464e9ccd597192dcb353",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class QwenVL_EntityDescriber:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_itm(self, base64_image: str, prompt: str,max_tokens: int = 500) -> Dict:
        try:
            if base64_image is None:
                messages= [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", 
                            "text": "You are a multimodal sentiment understanding expert."}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt},
                        ],
                    }]
            else:
                messages= [{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }]
            response = client.chat.completions.create(
                            model="qwen-vl-plus",
                            messages=messages,
                            max_tokens=max_tokens,
                            seed=42
                        )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用出错: {e}") 
            return None

def load_data(file_path: str):
    """从pkl文件中加载数据"""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_json_data(file_path: str):
 
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_data(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def save_json_data(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f)


def search_relation(relation_datas,aspect,img):
    for item in relation_datas:
        img_index=item['conversations'][0]['value']
        item_aspect=item['conversations'][0]['aspect']
        rel=item['conversations'][0]['relation']
        # path_part = img_index.split("<|vision_start|>/")[1]
        img_index=img_index.split('<|vision_end|>')[0].split('/')[-1]
        # 第二次分割获取文件名（含扩展名）
        # filename_with_ext = path_part.split("/")[-1]
        if img_index==img and aspect==item_aspect:
            return  rel
    
    return None
def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Generate entity descriptions using QwenVL.")
    parser.add_argument("--dataset", type=str,default='twitter2015', help="Dataset name")
    parser.add_argument("--data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/sentiment_meaning', help="Dataset name")
    parser.add_argument("--sentence_data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/new_data', help="Dataset name")
    parser.add_argument("--aspect_data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/output_v3', help="Dataset name")
    parser.add_argument("--relevance_data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/two_relation_inference_output', help="Dataset name")
    parser.add_argument("--output_dir", type=str, default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/sentiment_meaning', help="Dataset name")
    parser.add_argument("--img_dir", type=str,default='/data/lzy1211/code/twitterImage/' , help="Dataset name")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    # 配置参数
    API_KEY = "sk-da8e2b49bc49464e9ccd597192dcb353"
    describer = QwenVL_EntityDescriber(API_KEY)
    args = parse_arguments()
    # 若文件夹不存在，创建
    args.output_dir=os.path.join(args.output_dir,args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        # "train",'val', "test"
    types=["train",'val', "test"]
    for type in types:
        datas = load_json_data(os.path.join(args.data_dir,args.dataset,f'{type}.json'))
        count=0

        for item in tqdm(datas,desc="data process"):
            if 'sentiment_meaning'  not in item or item['sentiment_meaning']==None:
                sentence=item['text']
                aspect=item['aspect']
                label=item['label']
                image_emotion=item['image_emotion']
                relation=item['relation']
                text_clue=item['text_clue']
                image_caption=item['image_caption']

                if  relation=='the semantic relevance is relevant, the emotional relevance is relevant':#情感相关
                    Question_meaning=f"Original text: {sentence} Text sentiment clue: {text_clue} Image sentiment clue: {image_emotion} Image description:{image_caption} Aspect: {aspect} Based on the provideed sentences, generate a sentiment understanding description toward the aspect '{aspect}' in 2-3 sentences."
                    
                elif relation=="the semantic relevance is relevant, the emotional relevance is irrelevant":#情感无关语义相关
                    Question_meaning=f"Original text: {sentence} Text sentiment clue: {text_clue} Image description:{image_caption} Aspect: {aspect} Based on the provideed sentences, generate a sentiment understanding description toward the aspect '{aspect}' in 2-3 sentences."
                    
                else:#图文无关
                    Question_meaning=f"Original text: {sentence} Text sentiment clue: {text_clue} Image description:{image_caption} Aspect: {aspect} Based on the provideed sentences, generate a sentiment understanding description toward the aspect '{aspect}' in 2-3 sentences."
                    
                response3 = describer.generate_itm(base64_image=None,prompt=Question_meaning,max_tokens=128)
                item['sentiment_meaning'] = response3
    
                count += 1
                if count == 200:
                    # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
                    print('save_a_part')
                    count = 0
                    save_json_data(f'{args.output_dir}/{type}.json', datas)
                    # save_data(f'{args.output_dir}/new_{type}.pkl', results)
                        
                save_json_data(f'{args.output_dir}/{type}.json', datas)
   
 
    
    
     
    
    