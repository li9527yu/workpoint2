from transformers import (
    InstructBlipVisionConfig,
    InstructBlipQFormerConfig,
    OPTConfig,
    InstructBlipConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,InstructBlipQFormerConfig, InstructBlipQFormerModel,
    InstructBlipVisionModel
)
import torch
from PIL import Image
import requests
from torch import nn
from datasets import Dataset
import pickle
from tqdm import tqdm  # 导入 tqdm

model = InstructBlipForConditionalGeneration.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
processor = InstructBlipProcessor.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
config = InstructBlipConfig.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
# Q_Former
# configuration = InstructBlipQFormerConfig.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model.to(device)
image = Image.open('/public/home/ghfu/lzy/code/instructBLIP/1.jpg').convert("RGB")
prompt = "What is unusual about this image?"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(generated_text)



