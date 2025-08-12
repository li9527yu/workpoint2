import pandas as pd
import json
import os
import numpy as np
def load_and_merge_datasets(train_path, val_path, test_path):
    """
    加载训练集、验证集和测试集，并合并成一个完整数据集，同时将数值标签转换为文本标签
    """
    train_df = pd.read_csv(train_path, sep='\t')
    val_df = pd.read_csv(val_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    
    # 合并数据集
    # combined_train_df = pd.concat([train_df,val_df], ignore_index=True)
    
    return  train_df,val_df,test_df

label_to_text = {0: "negative", 1: "neutral", 2: "positive"}
def process_data(df,dataset):
    conversations = []
    # 添加对话数据
    for i in range(len(df)):
        img_dir=f'/public/home/ghfu/lzy/data/{dataset}_images/'
        aspect = df.iloc[i]['aspect']
        text_processed=df.iloc[i]['text'].replace('$T$',aspect)
        image_path = img_dir+df.iloc[i]['ImageID']
        label=df.iloc[i]['sentiment']
        conversations.append({
            "index": f"{i+1}",
            "aspect":aspect,
            "text":text_processed,
            "image":image_path,
            'label':label_to_text[label]
        })
    return conversations

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
import json

class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)
        

# 主程序入口
if __name__ == "__main__":
    datasets=['twitter2015','twitter2017']
    for  dataset in datasets:
        # 数据集路径
        data_dir=f'/public/home/ghfu/lzy/data/{dataset}'
        output_dir=f"/public/home/ghfu/lzy/code/instructBLIP/data/{dataset}/"
        train_csv = f"{data_dir}/train.tsv"
        val_csv = f"{data_dir}/dev.tsv"
        test_csv = f"{data_dir}/test.tsv"
        # output_file = f"{output_dir}/{dataset}_filtter_results.json"
        if os.path.exists(output_dir):
            print("exists")
        else:
            os.makedirs(output_dir)
        # 合并数据集
        train_df,val_df,test_df = load_and_merge_datasets(train_csv, val_csv, test_csv)
        # 处理数据集并保存
        train=process_data(train_df,dataset)
        # 保存为Json
        output_train = f"{output_dir}/train.json"
        with open(output_train, 'w', encoding='utf-8') as f:
            json.dump(train, f, ensure_ascii=False, indent=2)

        val=process_data(val_df,dataset)
        # 保存为Json
        output_val = f"{output_dir}/val.json"
        with open(output_val, 'w', encoding='utf-8') as f:
            json.dump(val, f, ensure_ascii=False, indent=2)

        test=process_data(test_df,dataset)
        # 保存为Json
        output_test = f"{output_dir}/test.json"
        with open(output_test, 'w', encoding='utf-8') as f:
            json.dump(test, f, ensure_ascii=False, indent=2)
        
        
        