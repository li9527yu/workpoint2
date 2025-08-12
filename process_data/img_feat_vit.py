import json
import os
import pickle
from PIL import Image
import torch
from tqdm import tqdm  # 导入 tqdm 库
from transformers import (
    InstructBlipVisionConfig,
    InstructBlipQFormerConfig,
    OPTConfig,
    InstructBlipConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,InstructBlipQFormerConfig, InstructBlipQFormerModel,
    InstructBlipVisionModel
)

# 初始化模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = InstructBlipForConditionalGeneration.from_pretrained.from_pretrained(
#     "/public/home/ghfu/lzy/model/instructblip-flan-t5-xl"
# ).to(device)
config = InstructBlipConfig.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
vision_model = InstructBlipVisionModel(config.vision_config).to(device)
processor = InstructBlipProcessor.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")

def extract_image_features(img_path):
    """图像特征抽取函数"""
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = vision_model(**inputs)[0]
            
        return features.squeeze().cpu().numpy()  # 转换为numpy数组并压缩维度
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def process_files(file_names=["train", "test", "val"]):
    datatsets=['twitter2015','twitter2017']
    for dataset in datatsets:
        """处理所有JSON文件的主函数"""
        output_dir=f'/public/home/ghfu/lzy/code/instructBLIP/img_feat/'
        os.makedirs(output_dir, exist_ok=True)
        # 使用字典存储特征，格式为 datas[img_id] = feature
        datas = {}
        for file_name in file_names:
            # 读取JSON文件
            data_path=f'/public/home/ghfu/lzy/code/instructBLIP/data/{dataset}/{file_name}.json'   
            with open(data_path, "r") as f:
                data = json.load(f)
                  
            # 处理每个数据项，并添加进度条
            for item in tqdm(data, desc=f"Processing {file_name}", unit="image"):
                img_path = item["image"]
                
                # 提取图像ID（最后/后的文件名）
                img_id = os.path.splitext(os.path.basename(img_path))[0]
                
                # 提取图像特征
                features = extract_image_features(img_path)
                
                if features is not None:
                    datas[img_id] = features  # 以 img_id 为键，特征为值
            
        # 保存为pkl文件
        output_path=f'{output_dir}/{dataset}.pkl'
        with open(output_path, "wb") as f:
            pickle.dump(datas, f)
            
        print(f"Processed {len(datas)} items saved to {output_path}")
        
    

if __name__ == "__main__":
    process_files()