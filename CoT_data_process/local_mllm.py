import os
import torch
import base64
from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from PIL import Image
import requests
from io import BytesIO
from typing import List, Optional, Union

class QwenVLModel:
    def __init__(self, model_path, gpu_ids: Optional[List[int]] = None, use_multi_gpu=True):
        # 设置 GPU 设备
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            self.device = "cuda"
            print(f"Using specified GPUs: {gpu_ids}")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using default device: {self.device}")

        # 加载模型和 tokenizer
        print(f"Loading model from {model_path}...")
        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, config=self.config)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, config=self.config)

        # 多卡设置
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for inference...")
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)

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

    def generate(self, prompt: str, image_path: Optional[str] = None, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2) -> str:
        # 构建消息
        if image_path:
            image = self.load_image(image_path)
            base64_image = self.encode_image(image)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }]
        
        # 模型推理
        with torch.no_grad():
            inputs = self.tokenizer(messages, return_tensors="pt").to(self.device)
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
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def batch_generate(self, prompts: List[str], image_paths: Optional[List[Union[str, None]]] = None, 
                       max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2) -> List[str]:
        # 检查图片路径和 prompt 数量是否匹配
        if image_paths is None:
            image_paths = [None] * len(prompts)
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

# 示例调用
model_path = "/path/to/your/qwen-vl-model"
qwen_model = QwenVLModel(model_path, gpu_ids=[0, 1], use_multi_gpu=True)

# 单条推理（多模态）
response_image = qwen_model.generate("这张图片的情感是什么？", image_path="sample.jpg")
print("\n单条多模态生成内容：\n", response_image)

# 单条推理（仅文本）
response_text = qwen_model.generate("描述一种积极的情感。")
print("\n单条仅文本生成内容：\n", response_text)

# 批量推理
prompts = ["描述一种积极的情感。", "这张图片的情感是什么？", "什么是幸福？"]
image_paths = [None, "sample.jpg", None]
batch_responses = qwen_model.batch_generate(prompts, image_paths)
print("\n批量生成内容：")
for i, res in enumerate(batch_responses):
    print(f"Sample {i+1}: {res}")
