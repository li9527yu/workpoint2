import os
import json

def load_json_or_jsonl(path):
    """Load json or jsonl file return list of dict"""
    items = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return items


def save_json_or_jsonl(path, items):
    """Save to same format as path"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def merge_textual_clues(split):
    """
    split ∈ ["train", "dev", "test"]
    """
    print(f"Processing: {split}")

    data1_path = f"/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_img_clue/twitter2015/{split}.json"
    data2_path = f"/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_text_clues/twitter2015/{split}.json"

    out_path = f"/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_gemini/twitter2015/{split}.json"

    data1 = load_json_or_jsonl(data1_path)
    data2 = load_json_or_jsonl(data2_path)

    if len(data1) != len(data2):
        print(f"⚠️ Warning: {split} size mismatch — data1={len(data1)} data2={len(data2)}")

    merged = []
    for i in range(min(len(data1), len(data2))):
        item1 = data1[i]
        item2 = data2[i]

        # === 核心操作：复制 textual_clues_parsed ===
        if "textual_clues_parsed" in item2:
            item1["textual_clues_parsed"] = item2["textual_clues_parsed"]
        else:
            item1["textual_clues_parsed"] = None  # 防止不存在导致 KeyError

        merged.append(item1)

    save_json_or_jsonl(out_path, merged)
    print(f"✔ Saved to: {out_path}")


def main():
    for split in ["train", "val", "test"]:
        merge_textual_clues(split)

    print("\n🎉 All splits processed successfully!\n")


if __name__ == "__main__":
    main()
