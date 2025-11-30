import base64
import requests
import json
from typing import List, Dict
import pickle
import os
from openai import OpenAI
from tqdm import tqdm
import argparse
from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from llm_openai import openai_chat_by_api_as_astream,InferenceParams
import asyncio 
# os.environ["http_proxy"] = "10.10.80.105:7788"
# os.environ["https_proxy"] = "10.10.80.105:7788"
client = OpenAI(
            api_key="sk-f46afc7fad9a43d1a6d0fecc1030eb39",# 
            base_url="https://api.deepseek.com"  # 我们提供的 url
        )

# async def call_gpt_api(prompt, model="deepseek-chat", max_tokens=500):
    
#     try:
#         # 调用异步函数
#         response = ""
#         async for chunk in openai_chat_by_api_as_astream(
#             model_name=model,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             inference_params=InferenceParams(temperature=0.5, max_tokens=max_tokens)
#         ):
#             response += chunk  # 拼接流式输出
#         return response
#     except Exception as e:
#         print(f"API调用出错: {e}")
#         return None

def call_gpt_api(prompt, model="deepseek-chat", max_tokens=500):
    
    try:
        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            seed=42,
                            temperature=0.5
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

def save_json_data(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f)
def construct_input(res_item):
    relation_map={"the semantic relevance is relevant, the emotional relevance is irrelevant":"relevant",
                  "the semantic relevance is relevant, the emotional relevance is relevant":"relevant",
                  "the semantic relevance is irrelevant, the emotional relevance is irrelevant":"irrelevant"}
    text = res_item['text']
    aspect = res_item['aspect']
    relevance = relation_map[res_item['relation']]
    prompt = f"""
You are a multimodal aspect-based sentiment analysis assistant.  
Use the text only to confirm which entity (aspect) is being referred to, but rely solely on the image to analyze emotional cues and determine the sentiment polarity.  
Use the provided relevance label to guide whether sentiment should be judged.  

Strictly output the following XML, wrapped with <<<BEGIN>>> and <<<END>>>.  
Only output the XML block between <<<BEGIN>>> and <<<END>>>. Do not output anything else.  

<<<BEGIN>>>
<result>
  <aspect></aspect>
  <image_has_aspect>true|false</image_has_aspect>
  <polarity>positive|neutral|negative</polarity>
  <evidence></evidence>
  <confidence>0.00~1.00</confidence>
  <visual_clues>
    <clue></clue>
    <clue></clue>
  </visual_clues>
</result>
<<<END>>>

Rules:
1) <aspect> must exactly match the input aspect.  
2) If the entity is not found in the image or no clear attitude is visible → set <polarity>neutral</polarity> and <confidence> between 0.00–0.30.  
3) <evidence> must be a single sentence , and must not include textual sentiment from the input.  
4) <visual_clues> can contain at most 3 items, each <clue> must be a short phrase only.  
5) <confidence> must be a decimal between 0.00 and 1.00 with exactly two decimal places (e.g., 0.25, 0.80).   
6) If RelevanceLabel is "irrelevant", output <image_has_aspect>false</image_has_aspect>, <polarity>neutral</polarity>, <confidence>0.00–0.30</confidence>.  
7) If RelevanceLabel is "relevant", analyze the image for sentiment clues related to the aspect.  

Input:
Text: "{text}"  
Aspect: "{aspect}" 
RelevanceLabel: "{relevance}" 
"""
    return prompt.strip()

def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Generate entity descriptions using QwenVL.")
    parser.add_argument("--dataset", type=str,default='twitter2015', help="Dataset name")
    parser.add_argument("--data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data/', help="Dataset name")
    parser.add_argument("--output_dir", type=str, default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue', help="Dataset name")
    parser.add_argument("--img_dir", type=str,default='/data/lzy1211/code/twitterImage/' , help="Dataset name")
    return parser.parse_args()

def main(args):
# 若文件夹不存在，创建
    args.output_dir=os.path.join(args.output_dir,args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        # "train",'val', "test"
    types=["train",'val', "test"]
    for type in types:
            # 原始数据 /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/twitter2017/train.json
        datas = load_json_data(os.path.join(args.data_dir,args.dataset,f'new_{type}.json'))
        # datas = load_json_data('/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/text_emotion/twitter2017/tt_train.json')
        count=0
        for item in tqdm(datas,desc="data process"):
            Question_reason=construct_input(item)
            # /item['text_clue'] is None
            if 'text_clue' not in item or item['text_clue'] is None:
                response = call_gpt_api(prompt=Question_reason,max_tokens=64)
                item['text_clue'] = response
                count += 1
                if count == 50:
                    # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
                    print('save_a_part')
                    count = 0
                    save_json_data(f'{args.output_dir}/{type}.json', datas)

                    
        save_json_data(f'{args.output_dir}/{type}.json', datas)
if __name__ == "__main__":
    # 解析命令行参数
    # 配置参数
    # API_KEY = "sk-da8e2b49bc49464e9ccd597192dcb353"
    # describer = QwenVL_EntityDescriber(API_KEY)
    args = parse_arguments()
    # asyncio.run(main(args))
    main(args)

