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
import json
from tqdm import tqdm  # 导入 tqdm

model = InstructBlipForConditionalGeneration.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
processor = InstructBlipProcessor.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
config = InstructBlipConfig.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
# Q_Former
# configuration = InstructBlipQFormerConfig.from_pretrained("/public/home/ghfu/lzy/model/instructblip-flan-t5-xl")
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model.to(device)
vision_model = InstructBlipVisionModel(config.vision_config).to(device)
query_tokens = nn.Parameter(torch.zeros(config.num_query_tokens, config.qformer_config.hidden_size)).to(device)
qformer = InstructBlipQFormerModel(config.qformer_config).to(device)




def imgdealByVit(img_path):
    
    # 检查图像是否为空
    if img is None:
        print(f"Warning: Unable to read image at path {img_path}. Skipping this file.")
        return None  # 返回 None 表示该图片未处理
    

    
    with torch.no_grad():
        outputs = model(**inputs)
        
    last_hidden_states = outputs.last_hidden_state.cpu()
    return last_hidden_states.squeeze()


if __name__ == '__main__':
    datatsets=['twitter2015','twitter2017']
    data_type=['train','val','test']
    with torch.no_grad():
        for dataset in datatsets:
            for type in data_type:
                datas = {}
                process_data=[]
                data_path=f'/public/home/ghfu/lzy/code/instructBLIP/data/{dataset}/{type}.json'
                output_path=f'/public/home/ghfu/lzy/code/instructBLIP/img_data/{dataset}/{type}.pkl'

                with open(data_path,'r') as f:
                    all_data=json.load(f)
                f.close()
                for item in tqdm(all_data, desc="Processing Data", unit="item"):
                    image = Image.open(item['image']).convert("RGB") 
                    inputs = processor(images=image, text=None, return_tensors="pt").to(device)
                    vision_outputs = vision_model(pixel_values=inputs['pixel_values'])
                    image_embeds = vision_outputs[0]
                    datas[item['image']] = image_embeds

    # img_dir_arr = ['/data/lzy1211/code/twitterImage/twitter2015_images','/data/lzy1211/code/twitterImage/twitter2017_images']
    img_dir_arr = ['/data/lzy1211/code/twitterImage/twitter2017_images']
    for img_dir in img_dir_arr:
        for filepath, dirnames, filenames in os.walk(img_dir):
            datas = {}
            count = 0
            for filename in tqdm(filenames, desc="deal data :" + img_dir):
                img_path = os.path.join(img_dir, filename)
                
                feature = imgdealByVit(img_path)
                # 仅在 feature 不为 None 时添加到字典中
                if feature is not None:
                    datas[filename] = feature
            saveFileName = img_dir.split('/')[-1]
            with open('/data/lzy1211/code/DPFN/DPFN-main/data/imgDealFile/'+ saveFileName + '.pkl', 'wb') as f:
                pickle.dump(datas, f)
    # data_path = '/data/tiancn/myModel01/imgDealFile/twitter2017_images.pkl'
    # with open(data_path, "rb") as f:
    #     A = pickle.load(f)
    # print(A)