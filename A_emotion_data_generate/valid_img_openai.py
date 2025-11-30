import base64
import json
from typing import List, Dict
import os
import mimetypes
from tqdm import tqdm

# ✅ 使用 OpenAI SDK（>=1.0）
from openai import OpenAI


class Gemini_Describer:
    """
    注意：虽然名称仍叫 Gemini_Describer，但已改为通过 OpenAI 兼容接口调用。
    - 可通过 base_url 指向 OpenAI 官方或你的兼容网关（如: https://api.whatai.cc/v1）
    - 使用 Chat Completions 接口，支持纯文本与图像（多模态）输入
    """
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.whatai.cc/v1",   # 你的兼容地址，或改为 "https://api.openai.com/v1"
        model: str ="gemini-2.5-pro"                  # 选择你网关支持的模型
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _build_user_content(self, prompt: str, image_path_or_b64: str = None):
        """
        生成 chat.completions 的 user 消息 content：
        - 纯文本： [{"type":"text","text": prompt}]
        - 文本+图片： [{"type":"text","text": prompt}, {"type":"image_url","image_url":{"url": "data:...;base64,..."}}]
        """
        content = [{"type": "text", "text": prompt}]

        if image_path_or_b64:
            # 如果是文件路径，读文件字节；否则当作 base64
            if os.path.exists(image_path_or_b64):
                img_bytes = open(image_path_or_b64, "rb").read()
                mime = mimetypes.guess_type(image_path_or_b64)[0] or "image/jpeg"
                b64 = base64.b64encode(img_bytes).decode("utf-8")
            else:
                # 已是 base64 字符串
                b64 = image_path_or_b64
                mime = "image/jpeg"  # 不确定类型时默认 jpeg
            data_url = f"data:{mime};base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        return content

    def generate_itm(self, image_path_or_b64: str, prompt: str, max_tokens: int = 1024) -> str:
        """
        使用 Chat Completions 进行（多模态）生成。
        """
        try:
            user_content = self._build_user_content(prompt, image_path_or_b64)

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.8,
            )
            return (resp.choices[0].message.content or "").strip()
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
        json.dump(data, f, ensure_ascii=False, indent=2)


def construct_input(res_item):
    relation_map = {
        "the semantic relevance is relevant, the emotional relevance is irrelevant": "relevant",
        "the semantic relevance is relevant, the emotional relevance is relevant": "relevant",
        "the semantic relevance is irrelevant, the emotional relevance is irrelevant": "irrelevant",
    }
    text = res_item["text"]
    aspect = res_item["aspect"]
    relevance = relation_map[res_item["relation"]]
    prompt = f"""
You are a multimodal aspect-based sentiment analysis assistant.  
Use the text only to confirm which entity (aspect) is being referred to, but rely solely on the image to analyze emotional cues and determine the sentiment polarity.  
Use the provided relevance label to guide whether sentiment should be judged.  

Strictly output the following XML, wrapped with <<<BEGIN>>> and <<<END>>>.  
Only output the XML block between <<<BEGIN>>> and <<<END>>>. Do not output anything else.  

<<<BEGIN>>>
<result>
  <aspect></aspect>
  <image_has_aspect>true|false</image_has_aspect>
  <polarity>positive|neutral|negative</polarity>
  <evidence></evidence>
  <confidence>0.00~1.00</confidence>
  <visual_clues>
    <clue></clue>
    <clue></clue>
  </visual_clues>
</result>
<<<END>>>

Rules:
1) <aspect> must exactly match the input aspect.  
2) If the entity is not found in the image or no clear attitude is visible → set <polarity>neutral</polarity> and <confidence> between 0.00–0.30.  
3) <evidence> must be a single sentence , and must not include textual sentiment from the input.  
4) <visual_clues> can contain at most 3 items, each <clue> must be a short phrase only.  
5) <confidence> must be a decimal between 0.00 and 1.00 with exactly two decimal places (e.g., 0.25, 0.80).   
6) If RelevanceLabel is "irrelevant", output <image_has_aspect>false</image_has_aspect>, <polarity>neutral</polarity>, <confidence>0.00–0.30</confidence>.  
7) If RelevanceLabel is "relevant", analyze the image for sentiment clues related to the aspect.  

Input:
Text: "{text}"  
Aspect: "{aspect}" 
RelevanceLabel: "{relevance}" 
"""
    return prompt.strip()


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Generate entity descriptions using Gemini_Describer (OpenAI-compatible).")
    parser.add_argument("--dataset", type=str, default="twitter2015", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_img_clue/twitter2015", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_img_clue/twitter2015", help="Dataset name")
    parser.add_argument("--img_dir", type=str, default="/data/lzy1211/code/twitterImage/", help="Dataset name")
    # 可选：模型/网关配置
    parser.add_argument("--base_url", type=str, default="https://api.whatai.cc/v1", help="OpenAI-compatible base url")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Model name supported by your gateway")
    return parser.parse_args()


if __name__ == "__main__":
    # ✅ 不要把密钥写在代码里，推荐用环境变量：export OPENAI_API_KEY=xxxx
    API_KEY ="sk-jQhszK1k8vksEOHWDQuSKIEaL1acbpcbVtNV7HLXjxMlYCWq"
    if not API_KEY:
        raise RuntimeError("请通过环境变量 OPENAI_API_KEY 提供密钥，例如：export OPENAI_API_KEY=xxxx")

    args = parse_arguments()
    describer = Gemini_Describer(api_key=API_KEY, base_url=args.base_url, model=args.model)

    # 若文件夹不存在，创建
    args.output_dir = os.path.join(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # "val", "test" "train","val",
    types_split = [ "train"]
    for type in types_split:
        datas = load_json_data(os.path.join(args.data_dir, f"{type}.json"))
        count = 0
        image_dir_path = f"{args.img_dir}/{args.dataset}_images/"
        results = {}
        for item in tqdm(datas, desc=f"data process ({type})"):
            # Question_reason = construct_input(item)
            question_img="Analyze the provided image.Identify its primary emotion as Positive, Negative, or Neutral.Provide a one-sentence explanation for your choice. Output strictly in this format: The image expresses [Emotion Label] because [Your one-sentence explanation]. Example:The image expresses Neutral because it's an objective depiction with no clear emotional cues."

            # 当前流程仅用文本；如需图像，取消注释以下两行并传入 generate_itm:
            # imgid=item['ImageID'].split('/')[-1]
            imgid=item['image'] 
            img_path = os.path.join(image_dir_path, imgid)
            base64_image = encode_image(img_path)
            # 2048
            if not item.get("img_clue"):
                response = describer.generate_itm(image_path_or_b64=base64_image, prompt=question_img, max_tokens=256)
                item["img_clue"] = response
                count += 1
                if count == 50:
                    # checkpoint：防意外中断丢进度
                    print("save_a_part")
                    count = 0
                    save_json_data(f"{args.output_dir}/{type}.json", datas)

        save_json_data(f"{args.output_dir}/{type}.json", datas)
        print(f"{type} done")
    print("ALL done!")
