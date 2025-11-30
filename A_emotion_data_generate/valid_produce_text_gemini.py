import base64
import requests
import json
from typing import List, Dict
import pickle
import os
from tqdm import tqdm
import base64, os, mimetypes
from google import genai
from google.genai import types
# ✅ 使用 Google GenAI SDK
from google import genai
from google.genai import types


class Gemini_Describer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate_itm(self, image_path_or_b64: str, prompt: str, max_tokens: int = 1024) -> str:
        try:
            parts = [prompt]

            if image_path_or_b64:
                # 如果是文件路径，读文件字节；否则当作 base64
                if os.path.exists(image_path_or_b64):
                    img_bytes = open(image_path_or_b64, "rb").read()
                    mime = mimetypes.guess_type(image_path_or_b64)[0] or "image/jpeg"
                else:
                    img_bytes = base64.b64decode(image_path_or_b64)
                    # 如果是 b64，不知道类型就先用 jpeg（可按你的数据调整）
                    mime = "image/jpeg"

                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))


                    #             temperature=0.7,
                    # top_p=0.95,
            resp = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=parts,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.8,
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:
            print(f"API调用出错: {e}")
            return ""


def encode_image(image_path: str) -> str:
    """将图片编码为base64（不含 data: 前缀）"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_json_data(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_data(file_name, data):
    with open(file_name, "w") as f:
        json.dump(data, f,ensure_ascii=False,indent=2)


def construct_input(res_item):
    text = res_item["text"]
    aspect = res_item["aspect"]
    prompt = f"""
You are an aspect-based sentiment analysis assistant.  
Given a sentence and a target aspect, analyze the sentiment expressed toward that aspect **based only on the sentence text**.  

Strictly output the following XML, wrapped with <<<BEGIN>>> and <<<END>>>.  
Only output the XML block between <<<BEGIN>>> and <<<END>>>. Do not output anything else.  

<<<BEGIN>>>
<result>
  <aspect></aspect>
  <polarity>positive|neutral|negative</polarity>
  <evidence></evidence>
  <confidence>0.00~1.00</confidence>
  <text_clues>
    <clue></clue>
    <clue></clue>
  </text_clues>
</result>
<<<END>>>

Rules:
1) <aspect> must exactly match the input aspect.  
2) <evidence> must always be filled with one concise English sentence (≤20 words) that directly explains why the polarity was chosen, based only on the sentence text.  
3) <text_clues> can contain at most 3 items, each <clue> must be a short phrase (e.g., “terribly slow”, “very delicious”), not a full sentence.  
4) <confidence> must be a decimal between 0.00 and 1.00 with exactly two decimal places.  

Input:  
- Sentence: "{text}"  
- Aspect: "{aspect}"  

"""
    return prompt.strip()


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Generate entity descriptions using Gemini_Describer.")
    parser.add_argument("--dataset", type=str, default="twitter2015", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/gemini_data_text", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/gemini_data_text", help="Dataset name")
    parser.add_argument("--img_dir", type=str, default="/data/lzy1211/code/twitterImage/", help="Dataset name")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数  AIzaSyDr0LFz2AfqcrdEv4bdkag8nlXnTTXMPco  AIzaSyBYB_rIa6K4TBRQl7V36cLX5z5QpyLFDpg
    API_KEY = "AIzaSyB5o1bZW_UjD6n9cLAjrxHTEcO6Ot_X-l8"  # ← 建议用环境变量：export GOOGLE_API_KEY=xxxx
    describer = Gemini_Describer(API_KEY)
    args = parse_arguments()
    
    # 若文件夹不存在，创建
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # print(args) "train", "val",
    types_split = ["train", "val", "test"]
    for type in types_split:
        datas = load_json_data(os.path.join(args.data_dir, args.dataset, f"{type}.json"))
        count = 0
        image_dir_path = f"{args.img_dir}/{args.dataset}_images/"
        results = {}
        for item in tqdm(datas, desc="data process"):
            Question_reason = construct_input(item)
            if "textual_clues" not in item or item["textual_clues"]=="":
                # img_path = f"{image_dir_path}/{item['image']}"
 
                # base64_image = encode_image(img_path)
                response = describer.generate_itm(image_path_or_b64=None, prompt=Question_reason, max_tokens=2048)
                item["textual_clues"] = response
                count += 1
                if count == 50:
                    # checkpoint：防意外中断丢进度
                    print("save_a_part")
                    count = 0
                    save_json_data(f"{args.output_dir}/{type}.json", datas)

        # 原始逻辑保留（注意：results 未填充；如需使用请在上文写入）
        save_json_data(f"{args.output_dir}/{type}.json", datas)
        print(f"{type} done")
    print("ALL done!")
