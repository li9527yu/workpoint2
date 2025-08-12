import base64
import requests
import json
from typing import List, Dict
import pickle
import os
from openai import OpenAI
from tqdm import tqdm
from modelscope import AutoModel , AutoTokenizer, AutoConfig
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os
import torch
from PIL import Image
from io import BytesIO
from typing import List, Optional, Union
from qwen_vl_utils import process_vision_info
from collections import defaultdict
from tqdm import tqdm

class QwenVLModel:
    def __init__(self, model_path, gpu_ids: Optional[List[int]] = None, use_multi_gpu=True):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"    
        # 加载模型和 tokenizer
        print(f"Loading model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")


    def load_image(self, image_path: str) -> Image.Image:
        if image_path.startswith("http"):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        return image.convert("RGB")

    def encode_image(self, image: Image.Image) -> str:
        # 将图片转换为 base64 格式
        image_bytes = BytesIO()
        image.save(image_bytes, format='JPEG')
        return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    def generate(self, prompt: str, image_path: Optional[str] = None, max_length=64, temperature=0.7, top_p=0.9, repetition_penalty=1.2) -> str:
        # 构建消息
        if image_path:
            # image = self.load_image(image_path)
            # base64_image = self.encode_image(image)
            messages = [{
                "role": "user",
                "content": [
                    {
                    "type": "image", 
                    "image": image_path
                    },
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }]
        
        with torch.no_grad():
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # 生成输出
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
    
        # 解码输出
        return output_text[0]
    
    def generate_batch(self, prompts: list, image_paths: Optional[list] = None, max_length=64, temperature=0.7, top_p=0.9, repetition_penalty=1.2) -> list:
        """
        批量推理函数。
        
        参数:
            prompts (list): 包含多个文本输入（prompt）的列表。
            image_paths (list, optional): 对应每个 prompt 的图片路径列表（可以为 None 或部分为空）。
            max_length (int): 生成文本的最大长度。
            temperature (float): 控制生成多样性的温度参数。
            top_p (float): 样本生成的 nucleus sampling 概率。
            repetition_penalty (float): 防止重复生成的惩罚系数。

        返回:
            list: 每个输入的生成文本结果列表。
        """
        # 检查输入是否匹配
        if image_paths is not None and len(prompts) != len(image_paths):
            raise ValueError("`prompts` 和 `image_paths` 的长度必须一致，或者 `image_paths` 为 None。")

        # 构建消息
        messages_batch = []
        for i, prompt in enumerate(prompts):
            if image_paths and image_paths[i]:
                messages = {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_paths[i]},
                        {"type": "text", "text": prompt},
                    ],
                }
            else:
                messages = {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            messages_batch.append(messages)

        with torch.no_grad():
            # 批量处理消息模板
            text_batch = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
            # 批量处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages_batch)
            
            # 构建批量输入
            inputs = self.processor(
                text=text_batch,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # 批量生成输出
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_length, 
                temperature=temperature, 
                top_p=top_p, 
                repetition_penalty=repetition_penalty
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # 返回所有生成的文本
        return output_texts
    def batch_generate(self, prompts: List[str], image_paths: Optional[List[Union[str, None]]] = None, 
                       max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2) -> List[str]:
        assert len(prompts) == len(image_paths), "prompts 和 image_paths 长度必须一致"
        
        # 构建批量消息
        messages = []
        for prompt, image_path in zip(prompts, image_paths):
            if image_path:
                image = self.load_image(image_path)
                base64_image = self.encode_image(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt},
                    ],
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                })
        
        # 模型批量推理
        with torch.no_grad():
            inputs = self.tokenizer(messages, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.module.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            ) if isinstance(self.model, torch.nn.DataParallel) else self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return responses


def load_data(file_path: str):
    """从pkl文件中加载数据"""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_json_data(file_path: str):
    """从pkl文件中加载数据"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_data(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Generate entity descriptions using QwenVL.")
    parser.add_argument("--model_path", type=str,default='/data/pub_jetwu/qwen2/Qwen2-VL-7B-Instruct/', help="Dataset name")
    parser.add_argument("--dataset", type=str,default='twitter2015', help="Dataset name")
    parser.add_argument("--data_dir", type=str,default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/sentiment_relation_twitter_output/', help="Dataset name")
    parser.add_argument("--output_dir", type=str, default='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/new_data', help="Dataset name")
    parser.add_argument("--img_dir", type=str,default='/data/lzy1211/code/twitterImage/' , help="Dataset name")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    qwen_model = QwenVLModel(args.model_path, gpu_ids=[5,6,7], use_multi_gpu=True)
    # 若文件夹不存在，创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    types=["train",'val', "test"]
    for type in types:
        datas = load_json_data(os.path.join(args.data_dir, args.dataset, f'{type}_result.json'))
        image_dir_path = f'{args.img_dir}/{args.dataset}_images/'
        results = defaultdict(dict)
        
        # 存储需要批量处理的数据
        batch_image_paths = []
        batch_prompts = []
        batch_keys = []
        
        count = 0  # 用于保存部分结果的计数器
        
        for item in tqdm(datas, desc="data process"):
            sentence = item["conversations"][0]['text']
            img = item["conversations"][0]['value']
            img = img.split('<|vision_end|>')[0].split('/')[-1]

            # 如果该图片还未处理或者文本发生变化，则需要处理
            if img not in results:
                results[img]['text'] = sentence
                img_path = f'{image_dir_path}/{img}'

                # 准备问题2的批量输入
                batch_image_paths.append(img_path)
                batch_prompts.append(
                    'Please describe the emotion expressed in the image. Try to be concise while ensuring the correctness of your answers.'
                )
                batch_keys.append(img)

                # 如果积累了一定数量的批量数据，进行推理并保存中间结果
                if len(batch_prompts) >= 4:  # 每次处理 10 张图片
                    responses = qwen_model.generate_batch(batch_prompts, batch_image_paths)
                    for key, response in zip(batch_keys, responses):
                        results[key]['image_emotion'] = response
                    
                    # 清空批量缓存
                    batch_image_paths = []
                    batch_prompts = []
                    batch_keys = []
                    count += 1

                    # 保存部分结果，防止中途意外中断
                    if count == 5:
                        print('save_a_part')
                        count = 0
                        save_data(f'{args.output_dir}/new_{type}.pkl', results)

        # 处理剩余未完成的批量数据
        if batch_prompts:
            responses = qwen_model.generate_batch(batch_prompts, batch_image_paths)
            for key, response in zip(batch_keys, responses):
                results[key]['image_emotion'] = response

        # 保存最终结果
        save_data(f'{args.output_dir}/new_{type}.pkl', results)

    # 保存最终结果
    save_data(f'{args.output_dir}/new_{type}.pkl', results)
    # for type in types:
    #     datas = load_json_data(os.path.join(args.data_dir,args.dataset,f'{type}_result.json'))
    #     count=0
    #     image_dir_path=f'{args.img_dir}/{args.dataset}_images/'
    #     results={}
    #     # Question1:image_description Question2:image_emotion Quesion3:text_description 
    #     for item in tqdm(datas,desc="data process"):
    #         sentence=item["conversations"][0]['text']
    #         img=item["conversations"][0]['value']
    #         img=img.split('<|vision_end|>')[0].split('/')[-1]
    #         if img not in results or sentence!=results[img]['text']:
    #             results[img]={}
    #             results[img]['text']=sentence
    #             img_path = f'{image_dir_path}/{img}'
    #             Question1 = 'Please describe the content in the image. Try to be concise while ensuring the correctness of your answers.'
    #             response1 = qwen_model.generate(image_path=img_path,prompt=Question1)
    #             results[img]['image_description'] = response1

    #             Question2 = 'Please describe the emotion expressed in the image. Try to be concise while ensuring the correctness of your answers.'
    #             response2 = qwen_model.generate(image_path=img_path,prompt=Question2)
    #             results[img]['image_emotion'] = response2

    #             # Question3 = f'Image description: {response1}; Text: "{sentence}"; Text description: {response2}. Please combine the image, text, and their description information and try to understand the deep meaning of the combination of the image and text. No need to describe images and text, only answer implicit meanings. Ensure the accuracy of the answer and try to be concise as much as possible.'
    #             Question3 = f'The text is as follows: "{sentence}". Please analyze the meaning of the text. Note that there may be homophonic memes and puns, distinguish and explain them but do not over interpret while ensuring the correctness of the answer and be concise.'
    #             response3 = qwen_model.generate(prompt=Question3)
    #             results[img]['text_description'] = response3

    #             count += 1
    #             if count == 10:
    #                 # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
    #                 print('save_a_part')
    #                 count = 0
    #                 save_data(f'{args.output_dir}/new_{type}.pkl', results)
                    
    #     save_data(f'{args.output_dir}/new_{type}.pkl', results)
   
    # output_dir = f"/data/lzy1211/GPT-API/output_v2/{dataset}"
    # image_path = f"/data/lzy1211/datasets/{dataset}_images/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # 初始化描述生成器
    
    
     
    
    