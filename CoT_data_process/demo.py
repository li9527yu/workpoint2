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
    parser.add_argument("--dataset", type=str,default='twitter2017', help="Dataset name")
    parser.add_argument("--data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/sentiment_relation_twitter_output/', help="Dataset name")
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
        datas = load_json_data(os.path.join(args.data_dir,args.dataset,f'{type}_result.json'))
        count=0
        image_dir_path=f'{args.img_dir}/{args.dataset}_images/'
        results={}
        # Question1:image_description Question2:image_emotion Quesion3:text_description 
        for item in tqdm(datas,desc="data process"):
            sentence=item["conversations"][0]['text']
            img=item["conversations"][0]['value']
            img=img.split('<|vision_end|>')[0].split('/')[-1]
            if img not in results or sentence!=results[img]['text']:
                results[img]={}
                results[img]['text']=sentence
                img_path = f'{image_dir_path}/{img}'
                base64_image = encode_image(img_path)
                # Question1 = 'Please describe the content in the image. Try to be concise while ensuring the correctness of your answers.'
                # response1 = describer.generate_itm(base64_image=base64_image,prompt=Question1,max_tokens=64)
                # results[img]['image_description'] = response1

                # Question2 = 'Please describe the emotion expressed in the image. Try to be concise while ensuring the correctness of your answers.'
                Question2="Analyze the provided image.Identify its primary emotion as Positive, Negative, or Neutral.Provide a one-sentence explanation for your choice. Output strictly in this format: The image expresses [Emotion Label] because [Your one-sentence explanation]. Example:The image expresses Neutral because it's an objective depiction with no clear emotional cues."
                response2 = describer.generate_itm(base64_image=base64_image,prompt=Question2,max_tokens=64)
                results[img]['image_emotion'] = response2

                # Question3 = f'Image description: {response1}; Text: "{sentence}"; Text description: {response2}. Please combine the image, text, and their description information and try to understand the deep meaning of the combination of the image and text. No need to describe images and text, only answer implicit meanings. Ensure the accuracy of the answer and try to be concise as much as possible.'
                # Question3 = f'The text is as follows: "{sentence}". Please analyze the meaning of the text. Note that there may be homophonic memes and puns, distinguish and explain them but do not over interpret while ensuring the correctness of the answer and be concise.'
                # response3 = describer.generate_itm(prompt=Question3,base64_image=None,max_tokens=64)
                # results[img]['text_description'] = response3

                count += 1
                if count == 100:
                    # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
                    print('save_a_part')
                    count = 0
                    save_data(f'{args.output_dir}/new_{type}.pkl', results)
                    
        save_data(f'{args.output_dir}/new_{type}.pkl', results)
   
    # output_dir = f"/data/lzy1211/GPT-API/output_v2/{dataset}"
    # image_path = f"/data/lzy1211/datasets/{dataset}_images/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # 初始化描述生成器
    
    
     
    
    