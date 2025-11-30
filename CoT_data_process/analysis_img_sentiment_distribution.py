import json
import os
from collections import Counter
import pandas as pd

# 标签映射表
LABEL_MAP = {0: "neutral", 1: "positive", 2: "negative"}

# ===============================
# 工具函数
# ===============================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_sentiment_from_text(text):
    """从自然语言描述中提取情感极性关键词"""
    if not text or not isinstance(text, str):
        return "unknown"
    text_lower = text.lower()
    if "positive" in text_lower:
        return "positive"
    elif "negative" in text_lower:
        return "negative"
    elif "neutral" in text_lower:
        return "neutral"
    else:
        return "unknown"

# ===============================
# 主统计函数
# ===============================
def count_image_sentiments(data, print_unknown_samples=False):
    """统计 img_clue、image_emotion、label 分布及一致性"""
    imgclue_counter = Counter()
    emotion_counter = Counter()
    label_counter = Counter()

    # 一致性统计
    imgclue_label_same, imgclue_label_total = 0, 0
    emo_label_same, emo_label_total = 0, 0

    unknown_samples = []

    for item in data:
        # 1️⃣ label
        label_val = item.get("label", None)
        if isinstance(label_val, str) and label_val.isdigit():
            label_val = int(label_val)
        label_sent = LABEL_MAP.get(label_val, "unknown")
        label_counter[label_sent] += 1

        # 2️⃣ img_clue
        clue_sent = get_sentiment_from_text(item.get("img_clue", ""))
        imgclue_counter[clue_sent] += 1
        if clue_sent == "unknown" and len(unknown_samples) < 5:
            unknown_samples.append(item.get("img_clue", ""))

        # 3️⃣ image_emotion
        emo_sent = get_sentiment_from_text(item.get("image_emotion", ""))
        emotion_counter[emo_sent] += 1

        # 一致性计算
        if clue_sent != "unknown" and label_sent != "unknown":
            imgclue_label_total += 1
            if clue_sent == label_sent:
                imgclue_label_same += 1

        if emo_sent != "unknown" and label_sent != "unknown":
            emo_label_total += 1
            if emo_sent == label_sent:
                emo_label_same += 1

    if print_unknown_samples and unknown_samples:
        print("\n⚠️ img_clue 中 'unknown' 样例（最多展示5条）：")
        for i, ex in enumerate(unknown_samples, 1):
            print(f"【样例{i}】{ex[:150]}")

    # 一致率
    imgclue_acc = imgclue_label_same / imgclue_label_total if imgclue_label_total > 0 else 0
    emo_acc = emo_label_same / emo_label_total if emo_label_total > 0 else 0

    agreement = {
        "imgclue_label_same": imgclue_label_same,
        "imgclue_label_total": imgclue_label_total,
        "imgclue_label_acc": imgclue_acc,
        "emo_label_same": emo_label_same,
        "emo_label_total": emo_label_total,
        "emo_label_acc": emo_acc
    }

    return imgclue_counter, emotion_counter, label_counter, agreement


# ===============================
# 主函数
# ===============================
def main():
    files = ["train.json", "val.json", "test.json"]
    total = {"img_clue": Counter(), "image_emotion": Counter(), "label": Counter()}
    total_agree = {
        "imgclue_label_same": 0, "imgclue_label_total": 0,
        "emo_label_same": 0, "emo_label_total": 0
    }
    rows = []

    input_dir = "/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_img_clue/twitter2017"

    for fname in files:
        input_path = os.path.join(input_dir, fname)
        try:
            data = load_json(input_path)
        except Exception as e:
            print(f"⚠️ 无法读取文件 {fname}: {e}")
            continue

        imgclue_counter, emotion_counter, label_counter, agreement = count_image_sentiments(
            data, print_unknown_samples=(fname == "train.json")
        )

        total["img_clue"].update(imgclue_counter)
        total["image_emotion"].update(emotion_counter)
        total["label"].update(label_counter)

        total_agree["imgclue_label_same"] += agreement["imgclue_label_same"]
        total_agree["imgclue_label_total"] += agreement["imgclue_label_total"]
        total_agree["emo_label_same"] += agreement["emo_label_same"]
        total_agree["emo_label_total"] += agreement["emo_label_total"]

        print(f"\n📊 {fname} 图像情感统计：")
        print("img_clue分布：", dict(imgclue_counter))
        print("image_emotion分布：", dict(emotion_counter))
        print("label分布：", dict(label_counter))
        print(f"✅ img_clue与label一致率：{agreement['imgclue_label_same']}/{agreement['imgclue_label_total']} = {agreement['imgclue_label_acc']:.3f}")
        print(f"✅ image_emotion与label一致率：{agreement['emo_label_same']}/{agreement['emo_label_total']} = {agreement['emo_label_acc']:.3f}")

        for src, counter in [("img_clue", imgclue_counter), ("image_emotion", emotion_counter), ("label", label_counter)]:
            rows.append({
                "split": fname.replace(".json", ""),
                "source": src,
                "positive": counter.get("positive", 0),
                "neutral": counter.get("neutral", 0),
                "negative": counter.get("negative", 0),
                "unknown": counter.get("unknown", 0)
            })

    print("\n============================")
    print("📈 总体统计（train+val+test）：")
    print("img_clue总体分布：", dict(total["img_clue"]))
    print("image_emotion总体分布：", dict(total["image_emotion"]))
    print("label总体分布：", dict(total["label"]))

    imgclue_acc = total_agree["imgclue_label_same"] / total_agree["imgclue_label_total"]
    emo_acc = total_agree["emo_label_same"] / total_agree["emo_label_total"]
    print(f"\n🔎 总体一致性统计：")
    print(f"img_clue vs label 一致率：{total_agree['imgclue_label_same']}/{total_agree['imgclue_label_total']} = {imgclue_acc:.3f}")
    print(f"image_emotion vs label 一致率：{total_agree['emo_label_same']}/{total_agree['emo_label_total']} = {emo_acc:.3f}")

    # 保存为CSV
    df = pd.DataFrame(rows)
    df.to_csv("/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_img_clue/twitter2015/image_sentiment_distribution.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 已保存统计结果到 image_sentiment_distribution.csv")
    print(df)


if __name__ == "__main__":
    main()
