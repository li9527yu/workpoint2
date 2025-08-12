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

model_path='/data/lzy1211/code/model/instructblip-flan-t5-xl'
model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
processor = InstructBlipProcessor.from_pretrained(model_path)
config = InstructBlipConfig.from_pretrained(model_path)
# Q_Former
# configuration = InstructBlipQFormerConfig.from_pretrained(model_path)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
vision_model = InstructBlipVisionModel(config.vision_config).to(device)
query_tokens = nn.Parameter(torch.zeros(config.num_query_tokens, config.qformer_config.hidden_size)).to(device)
qformer = InstructBlipQFormerModel(config.qformer_config).to(device)

# 读取数据集 15
datatsets=['twitter2017']
data_type=['train', 'dev','test']
with torch.no_grad():
    for dataset in datatsets:
        for type in data_type:
            process_data=[]
            # data_path=f'/data/lzy1211/code/A2II/instructBLIP/data/{dataset}/{type}.json'
            output_path=f'/data/lzy1211/code/A2II/instructBLIP/aspect_context_imgFeat/{dataset}/{type}.pkl'
            data_path=f'/data/lzy1211/code/A2II/instructBLIP/output_v3/{dataset}/{type}.json'
            with open(data_path,'r') as f:
                all_data=json.load(f)
            f.close()
            for item in tqdm(all_data, desc="Processing Data", unit="item"):
                img_path=item['image'].split('/')[-1]
                img_path=f'/data/lzy1211/code/twitterImage/{dataset}_images/{img_path}'
                image = Image.open(img_path).convert("RGB")
                aspect_caption=item['descriptions']
                prompt = f"{item['aspect']}: {aspect_caption}"
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                vision_outputs = vision_model(pixel_values=inputs['pixel_values'])
                image_embeds = vision_outputs[0]
                image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
                # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
                query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
                qformer_attention_mask = torch.cat([query_attention_mask, inputs['qformer_attention_mask']], dim=1)
                query_outputs = qformer(
                    input_ids=inputs['qformer_input_ids'],
                    attention_mask=qformer_attention_mask,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask
                )
                hidden_state=query_outputs[0][:, : query_tokens.size(1), :]
                item['hidden_state']=hidden_state.squeeze(0).cpu()
                item['pooler_output']=query_outputs[1].cpu()
                item['sentiment']=item['sentiment']
                # del item["label"] 
                process_data.append(item)

            # 保存为 pickle 文件
            with open(output_path, 'wb') as f:
                pickle.dump(process_data, f) 
            print("向量特征已保存为:",output_path)



