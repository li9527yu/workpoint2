import base64
import requests
import json
from typing import List, Dict
import pickle
import os
from openai import OpenAI
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoConfig
os.environ["http_proxy"] = "10.70.95.77:7788"
os.environ["https_proxy"] = "10.70.95.77:7788"
client = OpenAI(
    api_key="sk-tkgscc9Zhq0IzYkf5bFb7710Ab75436dAd962cC8B8BdBc90",# 
    base_url="https://api.gptapi.us/v1/chat/completions"  # 我们提供的 url
)

def call_gpt_api(prompt, model="gpt-3.5-turbo", max_tokens=500):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用出错: {e}")
        return None

def encode_image(image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
class QwenVL_EntityDescriber:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_itm(self, base64_image: str, prompt: str,max_tokens: int = 500) -> Dict:
        try:
            if base64_image is None:
                messages= [{
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


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Generate entity descriptions using QwenVL.")
    parser.add_argument("--dataset", type=str,default='twitter2015', help="Dataset name")
    parser.add_argument("--data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/new_data', help="Dataset name")
    parser.add_argument("--output_dir", type=str, default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/new_data', help="Dataset name")
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
    types=["train",'val', "test"]
    for type in types:
        datas = load_data(os.path.join(args.data_dir,args.dataset,f'new_{type}.pkl'))
        count=0
        for key,value in tqdm(datas.items(),desc="data process"):
            
            if value['image_emotion'] is None:
                image_caption=value['image_caption'][3]
                text=value['text']
                Question2=f"Analyze the image caption: {image_caption}.Identify its primary emotion as Positive, Negative, or Neutral.Provide a one-sentence explanation for your choice. Output strictly in this format: The image expresses [Emotion Label] because [Your one-sentence explanation]. Example:The image expresses Neutral because it's an objective depiction with no clear emotional cues."
                response2 = call_gpt_api(prompt=Question2,max_tokens=64)
                value['image_emotion'] = response2
                count += 1
                if count == 30:
                    # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
                    print('save_a_part')
                    count = 0
                    save_data(f'{args.output_dir}/new_{type}.pkl', datas)

                print("还有问题？！")
            if value['text_description'] is None:
                text=value['text']
                Question3 = f'The text is as follows: "{text}". Please analyze the meaning of the text. Note that there may be homophonic memes and puns, distinguish and explain them but do not over interpret while ensuring the correctness of the answer and be concise.'
                response3 = call_gpt_api(prompt=Question3,max_tokens=64)
                value['text_description'] = response3

                count += 1
                if count == 30:
                    # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
                    print('save_a_part')
                    count = 0
                    save_data(f'{args.output_dir}/new_{type}.pkl', datas)
                    
        save_data(f'{args.output_dir}/new_{type}.pkl', datas)
   

    
    
     
    
    