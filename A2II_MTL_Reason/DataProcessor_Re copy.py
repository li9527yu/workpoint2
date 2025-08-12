import json
import random
from PIL import Image
from torch.utils.data import Dataset

class UnifiedMultiTaskDataset(Dataset):
    def __init__(self, json_path, processor, task_weights=None,
                 max_input_length=128, max_output_length=256):
        """
        json_path: 数据路径，包含所有样本，每个样本有多个任务字段
        processor: InstructBLIP 的 processor
        task_weights: 任务概率字典，如 {"sentiment": 0.5, "reason": 0.2, "relevance": 0.3}
        """
        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.processor = processor
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.task_weights = task_weights or {
            "sentiment": 0.5,
            "reason": 0.3,
            "relevance": 0.2
        }

        self.task_types = list(self.task_weights.keys())
        self.task_probs = [self.task_weights[t] for t in self.task_types]
        total = sum(self.task_probs)
        self.task_probs = [p / total for p in self.task_probs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本
        sample = self.data[idx]
        image = Image.open(sample["image"]).convert("RGB")
        sentence = sample["sentence"].strip()
        aspect = sample.get("aspect", "this content")  # 可选字段

        # 动态选择任务
        task_type = random.choices(self.task_types, weights=self.task_probs)[0]

        # === 🧠 构造任务 ===
        if task_type == "sentiment":
            # 数值标签 -> 文字
            label_map = {0: "neutral", 1: "positive", 2: "negative"}
            label = label_map.get(sample["label"], "neutral")

            prompt = f"What is the sentiment of the following sentence about '{aspect}'?\n{sentence}"
            target = f"The sentiment of this aspect is {label}."

        elif task_type == "reason":
            reason = sample["response"].strip()
            label_map = {0: "neutral", 1: "positive", 2: "negative"}
            label = label_map.get(sample["label"], "neutral")

            prompt = f"What is the sentiment of the following sentence about '{aspect}'?\n{sentence}\nExplain your reasoning."
            target = f"The sentiment of this aspect is {label}.\n{reason}"

        elif task_type == "relevance":
            relevance = sample["relevance"].lower()
            prompt = f"Does the following text match the image content?\n{sentence}"
            target = f"The image and text are {relevance}."

        else:
            raise ValueError(f"Unsupported task: {task_type}")

        # 编码输入输出
        inputs = self.processor(
            images=image,
            text=prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )

        labels = self.processor.tokenizer(
            target,
            padding="max_length",
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt"
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = labels.squeeze(0)
        inputs["task_type"] = task_type  # 可用于 loss logging

        return inputs
