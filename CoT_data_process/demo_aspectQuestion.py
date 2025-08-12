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
    parser.add_argument("--dataset", type=str,default='twitter2017', help="Dataset name")
    parser.add_argument("--data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/reason_data/data/', help="Dataset name")
    parser.add_argument("--sentence_data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/new_data', help="Dataset name")
    parser.add_argument("--aspect_data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/output_v3', help="Dataset name")
    parser.add_argument("--relevance_data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/two_relation_inference_output', help="Dataset name")
    parser.add_argument("--output_dir", type=str, default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data', help="Dataset name")
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
    types=["train"]
    for type in types:
       
        if type=='val':
             # 原始数据
            datas = load_json_data(os.path.join(args.data_dir,args.dataset,f'dev_cause.json'))
            # 方面背景相关数据
            aspect_datas = load_json_data(os.path.join(args.aspect_data_dir,args.dataset,f'dev.json'))
        else:
             # 原始数据
            datas = load_json_data(os.path.join(args.data_dir,args.dataset,f'{type}_cause.json'))
            # 方面背景相关数据
            aspect_datas = load_json_data(os.path.join(args.aspect_data_dir,args.dataset,f'{type}.json'))
        
        # 前面生成的图文额外信息
        sentence_datas = load_data(os.path.join(args.sentence_data_dir,args.dataset,f'new_{type}.pkl'))
        # 方面的相关性信息
        if type=='test':
            relevance_datas = load_json_data(os.path.join(args.relevance_data_dir,args.dataset,f'test_result.json'))
        else:
            relevance_datas = load_json_data(os.path.join(args.relevance_data_dir,args.dataset,f'train_result.json'))
        count=0
        image_dir_path=f'{args.img_dir}/{args.dataset}_images/'
        # 最后的结果
        results=[]
        for item,item2 in tqdm(zip(datas,aspect_datas),desc="data process"):
            res_item={}
            sentence=item['sentence']
            img=item['image']
            aspect=item['aspect']
            label=item['label']
            sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
            sentiment = sentiment_map[str(label)]
            # 
            aspect_Context=item2['descriptions']
            sentence_item=sentence_datas[img]
            # 
            if sentence_item['text']==sentence:
                # res_item=sentence_datas[img]
                image_description=sentence_item['image_caption'][3]
                image_emotion=sentence_item['image_emotion']
                text_description=sentence_item['text_description']
            
            res_item={
                'text':sentence,
                'image_emotion':image_emotion,
                'image_caption':image_description,
                'text_description':text_description,
                'image':img,
                'aspect':aspect,
                'label':label,
                'aspect_Context':aspect_Context
            }
            # res_item['image']=img
            # res_item['aspect']=aspect
            # res_item['label']=label
            # res_item['aspect_Context']=aspect_Context
            # 
            relation=search_relation(relevance_datas,aspect,img)
            res_item['relation']=relation
            if relation is None:
                print("relation search error")
            else:
                semantic_rel,emotional_rel=relation.split(',')
                semantic_relation_label,emotional_relation_label=0,0
                Question_reason=''
                Question_meaning=''
                # - Sentiment&reasoning：- 根据不同的相关性进行不同的推理分支，输出对应的情感以及推理过程。
                # - img-text meaning：- 根据不同的相关性进行不同的推理分支，结合前面生成的信息对图文对进行理解
                if 'irrelevant' in semantic_rel:
                    semantic_relation_label=0
                else:
                    semantic_relation_label=1

                if 'irrelevant' in emotional_rel:
                    emotional_relation_label=0
                else:
                    emotional_relation_label=1
                
                if emotional_relation_label==1 and semantic_relation_label==1:#情感相关

                    Question_reason=f"Image description:{image_description}; Image emotion:{image_emotion};Text:{sentence};Text description:{text_description}; Please combine the image, text, image description, text description, and image emotion to explain why is the sentiment towards the '{aspect},{aspect_Context}' is {sentiment}? Output strictly in this format: Because [Your explanation]."
                    Question_meaning=f"Image description:{image_description}; Image emotion:{image_emotion};Text:{sentence}; Please combine the image description, image emotion, image, and text to form a comprehensive understanding of the meaning related to {aspect}. Provide a concise response within 2-3 sentences"
                
                elif emotional_relation_label==0 and semantic_relation_label==1:#情感无关语义相关

                    Question_reason=f"Image description:{image_description};Text:{sentence}; Text description:{text_description}; Please combine the image, text and their description to explain why is the sentiment towards the '{aspect},{aspect_Context}' is {sentiment}? Output strictly in this format: Because [Your explanation]. "
                    Question_meaning=f"Image description:{image_description}; Image emotion:{image_emotion};Text:{sentence}; Please combine the image description, image emotion, image, and text to form a comprehensive understanding of the meaning related to {aspect}. Provide a concise response within 2-3 sentences"
                
                else:#图文无关

                    Question_reason=f"Text:{sentence};Text description:{text_description}; Please combine the image, text and text description to explain why is the sentiment towards the '{aspect},{aspect_Context}' is {sentiment}? Output strictly in this format: Because [Your explanation]."
                    Question_meaning=f"Text:{sentence}; Please combine the image and text to form a comprehensive understanding of the meaning related to {aspect}. Provide a concise response within 2-3 sentences"

                img_path = f'{image_dir_path}/{img}'
                base64_image = encode_image(img_path)
                
                response2 = describer.generate_itm(base64_image=base64_image,prompt=Question_reason,max_tokens=100)
                res_item['sentiment_reasoning'] = response2

                response3 = describer.generate_itm(base64_image=base64_image,prompt=Question_meaning,max_tokens=100)
                res_item['imagetext_meaning'] = response3

                results.append(res_item)
                count += 1
                if count == 200:
                    # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
                    print('save_a_part')
                    count = 0
                    save_json_data(f'{args.output_dir}/{type}.json', results)
                    # save_data(f'{args.output_dir}/new_{type}.pkl', results)
                    
        save_json_data(f'{args.output_dir}/{type}.json', results)
   
    # output_dir = f"/data/lzy1211/GPT-API/output_v2/{dataset}"
    # image_path = f"/data/lzy1211/datasets/{dataset}_images/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # 初始化描述生成器
    
    
     
    
    