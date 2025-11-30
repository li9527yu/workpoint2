import json
from tqdm import tqdm
import os
import random


def load_json_data(file_path: str):
    """
    加载JSON数据文件
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_data(file_name, data):
    """
    保存处理后的JSON数据
    """
    with open(file_name, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def inject_noise_to_relation(data, noise_rate=0.1, seed=42):
    """
    向相关性标签（relation）注入噪音，随机翻转 `semantic` 或 `emotional` 标签。
    :param data: 原始数据
    :param noise_rate: 噪音比例（0到1之间，表示多少比例的标签会被翻转）
    :param seed: 随机种子
    :return: 带噪音的修改后的数据
    """
    random.seed(seed)
    
    # 定义原始的relation标签及其反转方式
    relation_flip_map = {
        "the semantic relevance is relevant, the emotional relevance is irrelevant": [
            "the semantic relevance is relevant, the emotional relevance is relevant", 
            "the semantic relevance is irrelevant, the emotional relevance is irrelevant"
        ],
        "the semantic relevance is relevant, the emotional relevance is relevant": [
            "the semantic relevance is relevant, the emotional relevance is irrelevant", 
            "the semantic relevance is irrelevant, the emotional relevance is irrelevant"
        ],
        "the semantic relevance is irrelevant, the emotional relevance is irrelevant": [
            "the semantic relevance is relevant, the emotional relevance is irrelevant", 
            "the semantic relevance is relevant, the emotional relevance is relevant"
        ]
    }

    # 处理每个样本
    for item in data:
        rr=random.random()
        if  rr< noise_rate:  # 控制噪音注入比例
            original_relation = item["relation"]
            # 随机选择关系并注入噪声
            choose=random.choice(relation_flip_map[original_relation])
            item["relation"] = choose

    return data


# 主处理流程
input_dir = '/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/twitter2017'
output_dir = '/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/relation_noise_20/twitter2017'

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

types = ['train', 'val', 'test']
for type in types:
    # 读取原始数据
    input_path = os.path.join(input_dir, f"{type}.json")
    datas = load_json_data(input_path)

    # 注入噪音
    noisy_data = inject_noise_to_relation(datas, noise_rate=0.2)

    # 保存带噪音的数据
    output_path = os.path.join(output_dir, f"{type}.json")
    save_json_data(output_path, noisy_data)

    print(f"数据已保存到 {output_path}")
