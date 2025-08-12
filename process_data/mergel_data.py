import torch
import argparse
import os
from datasets import Dataset
import pickle
import json

def convert_detail(data_path,type,img_path):
    # 从 pickle 文件读取数据
    img_path=f'{img_path}/{type}.pkl'
    with open(img_path, 'rb') as f:
        img_data = pickle.load(f)

    f.close()
    with open(data_path,'r') as f:
        datas=json.load(f)

    deal_data=[]
    for (item1,item2) in zip(datas,img_data):
        deal_data.append({
            "index": item1['index'],
            "aspect": item1['aspect'],
            "text": item1['text'],
            "image": item1['image'],
            "img_feat":item2['feat']
        })
    
    return deal_data


# 处理数据
data_dir='/public/home/ghfu/lzy/code/instructBLIP/data/'
datasets=['twitter2015','twitter2017']
output_dir='/public/home/ghfu/lzy/code/instructBLIP/deal_data/'
for dataset in datasets:
    img_path=f'/public/home/ghfu/lzy/code/instructBLIP/img_data/{dataset}/'
    output_dir=f'{output_dir}/{dataset}'
    if os.path.exists(output_dir):
        print("exists")
    else:
        os.makedirs(output_dir)
    train_data=f"{data_dir}/{dataset}/train.json"
    val_data=f"{data_dir}/{dataset}/val.json"
    test_data=f"{data_dir}/{dataset}/test.json"
    # 将图像特征加入到原始数据集中
    train_data=convert_detail(train_data,'train',img_path)
    output_train = f"{output_dir}/train.json"
    with open(output_train, 'wb') as f:
        pickle.dump(train_data, f)

    val_data=convert_detail(val_data,'val',img_path)
    output_val = f"{output_dir}/val.json"
    with open(output_val, 'wb') as f:
        pickle.dump(val_data, f)

    test_data=convert_detail(test_data,'test',img_path)
    output_test = f"{output_dir}/test.json"
    with open(output_test, 'wb') as f:
        pickle.dump(test_data, f)

    print(dataset,"complete")
