
import json
import os

# ============ 文件路径 ===============
total_train_path = "/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/twitter2017/train.json"  # 总的 train.json
split_dir = "/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/AECR_GeminiImg/"         # 子文件所在目录（请按实际修改）
split_files = ["train.json", "dev.json", "test.json"]

# ============ 1. 读取总文件 ===============
with open(total_train_path, "r", encoding="utf-8") as f:
    total_data = json.load(f)
print(f"📘 总文件样本数: {len(total_data)}")

# ============ 2. 读取所有子文件数据 ===============
split_data = []
for file in split_files:
    path = os.path.join(split_dir, file)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            split_data.extend(json.load(f))
    else:
        print(f"⚠️ 未找到子文件: {path}")
print(f"📗 子文件合计样本数: {len(split_data)}")

# ============ 3. 提取文件名工具函数 ===============
def get_filename(path_str):
    if not path_str:
        return None
    return os.path.basename(path_str.strip())

# ============ 4. 双重循环进行匹配 ===============
count_found = 0
unmatched = []

for total_item in total_data:
    total_filename = get_filename(total_item.get("image") or total_item.get("ImageID"))
    found = False

    # 遍历子文件寻找匹配项
    for sub_item in split_data:
        sub_filename = get_filename(sub_item.get("image") or sub_item.get("ImageID"))
        if total_filename == sub_filename:
            total_item["img_clue"] = sub_item.get("img_clue", None)
            found = True
            count_found += 1
            break  # 匹配到后立即停止当前循环
    
    # 没有匹配到则记录
    if not found:
        total_item["img_clue"] = None
        unmatched.append({
            "image": total_filename,
            "aspect": total_item.get("aspect", ""),
            "text": total_item.get("text", "")[:80]
        })

print(f"\n✅ 成功匹配 {count_found}/{len(total_data)} 条样本")
print(f"⚠️ 未匹配 {len(unmatched)} 条样本")

# 输出未匹配样本的前几条
if unmatched:
    print("\n未匹配样本示例（前5条）:")
    for u in unmatched[:5]:
        print(f"- image: {u['image']} | aspect: {u['aspect']} | text: {u['text']}...")

# 保存未匹配样本
with open("unmatched_samples.json", "w", encoding="utf-8") as f:
    json.dump(unmatched, f, ensure_ascii=False, indent=2)
print("📄 未匹配样本已保存到 unmatched_samples.json")

# ============ 5. 保存新文件 ===============
output_path = "/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_img_clue/twitter2017/train.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(total_data, f, ensure_ascii=False, indent=2)

print(f"\n🎯 已保存更新后的文件到: {output_path}")
