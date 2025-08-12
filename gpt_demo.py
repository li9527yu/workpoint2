from datasets import DatasetDict, Dataset
from transformers import T5Tokenizer, FlanT5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# 数据准备
data = {
    "train": [
        {"sentence": "I love this product!", "aspect": "product", "label": "positive"},
        {"sentence": "This is okay.", "aspect": "service", "label": "neutral"},
        {"sentence": "I hate this experience.", "aspect": "experience", "label": "negative"},
        {"sentence": "Amazing service!", "aspect": "service", "label": "positive"},
        {"sentence": "Not bad.", "aspect": "food", "label": "neutral"}
    ],
    "test": [
        {"sentence": "I will never buy this again.", "aspect": "product", "label": "negative"},
        {"sentence": "I really enjoy this.", "aspect": "product", "label": "positive"}
    ]
}

train_dataset = Dataset.from_dict(data["train"])
test_dataset = Dataset.from_dict(data["test"])

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

label2id = {"positive": "positive", "neutral": "neutral", "negative": "negative"}
id2label = { "positive": "positive", "neutral": "neutral", "negative": "negative" }

def preprocess_function(examples):
    inputs = [
        f"Definition: Combining information from image and the following sentence to identify the sentiment of aspect in the sentence. "
        f"Sentence: {ex['sentence']} aspect: {ex['aspect']} OPTIONS: -positive -neutral -negative output:" for ex in examples
    ]
    model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=128)
    labels = [label2id[ex["label"]] for ex in examples]
    model_inputs["labels"] = tokenizer(labels, padding=True, truncation=True, max_length=2).input_ids
    return model_inputs

train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"]
eval_dataset = train_val_split["test"]

encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
encoded_test_dataset = dataset["test"].map(preprocess_function, batched=True)

def compute_metrics(p):
    predictions, labels = p
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./results", 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=64,
    num_train_epochs=3, 
    logging_dir="./logs", 
    logging_strategy="epoch", 
    load_best_model_at_end=True, 
    metric_for_best_model="accuracy", 
    greater_is_better=True
)

model = FlanT5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
model.config.label2id = label2id
model.config.id2label = id2label

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 在测试集上进行评估
results = trainer.evaluate(encoded_test_dataset)
print(results)
