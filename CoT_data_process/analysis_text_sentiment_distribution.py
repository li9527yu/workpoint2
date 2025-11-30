import json
import os
from collections import Counter, defaultdict
import pandas as pd

LABEL_MAP = {0: "neutral", 1: "positive", 2: "negative"}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_sentiment_from_text(text):
    """从text_clue字符串中提取情感极性"""
    text_lower = text.lower()
    if "positive" in text_lower:
        return "positive"
    elif "negative" in text_lower:
        return "negative"
    else:
        return "neutral"

def count_sentiments_and_agreement(data, print_unknown_samples=False):
    """统计text_clue、textual_clues_parsed、label分布及一致性"""
    clue_counter = Counter()
    parsed_counter = Counter()
    label_counter = Counter()

    # 一致性统计
    clue_label_same, clue_label_total = 0, 0
    parsed_label_same, parsed_label_total = 0, 0

    unknown_samples = []

    for item in data:
        # 1️⃣ label
        label_val = item.get("label", None)
        if isinstance(label_val, str) and label_val.isdigit():
            label_val = int(label_val)
        label_sent = LABEL_MAP.get(label_val, "unknown")
        label_counter[label_sent] += 1

        # 2️⃣ text_clue
        clue_sent = "unknown"
        if "text_clue" in item and item["text_clue"]:
            clue_sent = get_sentiment_from_text(item["text_clue"])
            clue_counter[clue_sent] += 1
            if clue_sent == "unknown" and len(unknown_samples) < 5:
                unknown_samples.append(item["text_clue"])

        # 3️⃣ textual_clues_parsed
        parsed_sent = "unknown"
        if "textual_clues_parsed" in item and isinstance(item["textual_clues_parsed"], dict):
            parsed_sent = item["textual_clues_parsed"].get("polarity", "").lower()
            if parsed_sent in ["positive", "negative", "neutral"]:
                parsed_counter[parsed_sent] += 1
            else:
                parsed_counter["unknown"] += 1

        # 一致性统计
        if clue_sent != "unknown" and label_sent != "unknown":
            clue_label_total += 1
            if clue_sent == label_sent:
                clue_label_same += 1

        if parsed_sent != "unknown" and label_sent != "unknown":
            parsed_label_total += 1
            if parsed_sent == label_sent:
                parsed_label_same += 1

    if print_unknown_samples and unknown_samples:
        print("\n⚠️ text_clue中 'unknown' 样例（最多展示5条）：")
        for i, ex in enumerate(unknown_samples, 1):
            print(f"【样例{i}】{ex[:150]}")

    # 计算一致率
    clue_acc = clue_label_same / clue_label_total if clue_label_total > 0 else 0
    parsed_acc = parsed_label_same / parsed_label_total if parsed_label_total > 0 else 0

    agreement = {
        "clue_label_same": clue_label_same,
        "clue_label_total": clue_label_total,
        "clue_label_acc": clue_acc,
        "parsed_label_same": parsed_label_same,
        "parsed_label_total": parsed_label_total,
        "parsed_label_acc": parsed_acc
    }

    return clue_counter, parsed_counter, label_counter, agreement


def main():
    files = ["train.json", "val.json", "test.json"]
    total = {"text_clue": Counter(), "textual_clues_parsed": Counter(), "label": Counter()}
    total_agree = {"clue_label_same": 0, "clue_label_total": 0,
                   "parsed_label_same": 0, "parsed_label_total": 0}
    rows = []
    input_dir='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_text_clues/twitter2015'

    for fname in files:
        input_path=os.path.join(input_dir,fname)
        try:
            data = load_json(input_path)
        except Exception as e:
            print(f"⚠️ 无法读取文件 {fname}: {e}")
            continue

        clue_counter, parsed_counter, label_counter, agreement = count_sentiments_and_agreement(
            data, print_unknown_samples=(fname == "train.json")  # 仅打印一次样例
        )

        total["text_clue"].update(clue_counter)
        total["textual_clues_parsed"].update(parsed_counter)
        total["label"].update(label_counter)
        total_agree["clue_label_same"] += agreement["clue_label_same"]
        total_agree["clue_label_total"] += agreement["clue_label_total"]
        total_agree["parsed_label_same"] += agreement["parsed_label_same"]
        total_agree["parsed_label_total"] += agreement["parsed_label_total"]

        print(f"\n📊 {fname} 情感统计：")
        print("text_clue分布：", dict(clue_counter))
        print("textual_clues_parsed分布：", dict(parsed_counter))
        print("label分布：", dict(label_counter))
        print(f"✅ text_clue与label一致率：{agreement['clue_label_same']}/{agreement['clue_label_total']} = {agreement['clue_label_acc']:.3f}")
        print(f"✅ textual_clues_parsed与label一致率：{agreement['parsed_label_same']}/{agreement['parsed_label_total']} = {agreement['parsed_label_acc']:.3f}")

        # 汇总表格
        for src, counter in [("text_clue", clue_counter), ("textual_clues_parsed", parsed_counter), ("label", label_counter)]:
            rows.append({
                "split": fname.replace(".json", ""),
                "source": src,
                "positive": counter.get("positive", 0),
                "neutral": counter.get("neutral", 0),
                "negative": counter.get("negative", 0),
                "unknown": counter.get("unknown", 0)
            })

    print("\n============================")
    print("📈 总体统计（train+dev+test）：")
    print("text_clue总体分布：", dict(total["text_clue"]))
    print("textual_clues_parsed总体分布：", dict(total["textual_clues_parsed"]))
    print("label总体分布：", dict(total["label"]))

    clue_acc = total_agree["clue_label_same"] / total_agree["clue_label_total"]
    parsed_acc = total_agree["parsed_label_same"] / total_agree["parsed_label_total"]
    print(f"\n🔎 总体一致性统计：")
    print(f"text_clue vs label 一致率：{total_agree['clue_label_same']}/{total_agree['clue_label_total']} = {clue_acc:.3f}")
    print(f"textual_clues_parsed vs label 一致率：{total_agree['parsed_label_same']}/{total_agree['parsed_label_total']} = {parsed_acc:.3f}")

    # 保存为CSV
    df = pd.DataFrame(rows)
    df.to_csv("sentiment_distribution_detailed.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 已保存统计结果到 sentiment_distribution_detailed.csv")
    print(df)


if __name__ == "__main__":
    main()
