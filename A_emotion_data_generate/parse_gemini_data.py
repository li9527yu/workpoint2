import json
import os
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Iterator, List
import re
BEGIN_TOK, END_TOK = "<<<BEGIN>>>", "<<<END>>>"




def parse_xml_block(block: Optional[str]) -> Optional[Dict]:
    """宽松解析大模型输出中的textual_clues/img_clue，支持无BEGIN/END的情况。"""
    if not block or not isinstance(block, str):
        return None
    s = block.strip()

    # --- 尝试直接解析 JSON 结构 ---
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass

    # --- 尝试截取 XML 核心部分 ---
    xml_match = re.search(r"<(result|aspect|polarity|evidence|clue)[\s\S]+?>[\s\S]+</(result|aspect|polarity|evidence|clue)>", s)
    if xml_match:
        xml_text = xml_match.group(0)
    else:
        # 如果没有明显的 XML 标签，但有BEGIN/END
        xml_text = s
        xml_text = re.sub(r"^<<<BEGIN>>>", "", xml_text)
        xml_text = re.sub(r"<<<END>>>$", "", xml_text)
        xml_text = xml_text.strip()

    # --- 尝试XML解析 ---
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # XML失败时尝试自然语言解析
        lower_s = s.lower()
        aspect = re.search(r"aspect[:：]?\s*([a-zA-Z0-9_\- ]+)", lower_s)
        polarity = re.search(r"(sentiment|polarity)[:：]?\s*(positive|negative|neutral)", lower_s)
        return {
            "aspect": aspect.group(1).strip() if aspect else None,
            "polarity": polarity.group(2) if polarity else None,
            "_raw": block
        }

    # --- 从 XML 节点提取信息 ---
    def _text(tag: str) -> Optional[str]:
        node = root.find(tag)
        return node.text.strip() if node is not None and node.text else None

    def _float(tag: str) -> Optional[float]:
        t = _text(tag)
        try:
            return float(t) if t is not None else None
        except Exception:
            return None

    image_has_aspect = root.find("image_has_aspect")
    if image_has_aspect is not None:
        val = (image_has_aspect.text or "").strip().lower()
        image_has_aspect = True if val in {"true", "1", "yes"} else False if val in {"false", "0", "no"} else None

    clues = [c.text.strip() for c in root.findall(".//clue") if c.text and c.text.strip()]

    return {
        "aspect": _text("aspect"),
        "polarity": _text("polarity"),
        "evidence": _text("evidence"),
        "confidence": _float("confidence"),
        "image_has_aspect": image_has_aspect,
        "clues": clues,
    }
 

def process_item(item: Dict, keep_original: bool = False, flatten: bool = True) -> Dict:
    out = dict(item)  # 浅拷贝

    # 解析 textual_clues
    # txt_parsed = parse_xml_block(out.get("textual_clues"))
    # 解析 img_clue
    img_parsed = parse_xml_block(out.get("img_clue"))

    # if txt_parsed is not None:
    #     out["textual_clues_parsed"] = txt_parsed

    if img_parsed is not None:
        out["img_clues_parsed"] = img_parsed
    if not keep_original:
        out.pop("textual_clues", None)

    return out

def save_items(input_path: str, output_path: str, keep_original: bool = False, flatten: bool = True) -> None:
 
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        data = json.load(f)
    processed=[]
    for obj in data:
        itm=process_item(obj, keep_original=keep_original, flatten=flatten)
        processed.append(itm)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

def repair_items(input_path: str, output_path: str, keep_original: bool = False, flatten: bool = True) -> None:
 
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        data = json.load(f)
    processed=[]
    for obj in data:
        if "_raw" in obj.get("img_clues_parsed", {}):
            itm=process_item(obj, keep_original=keep_original, flatten=flatten)
            processed.append(itm)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)



def main():
    parser = argparse.ArgumentParser(description="Parse textual_clues/img_clue XML blocks and save to new JSON.")
    parser.add_argument("--input", "-i",   help="Path to input JSON or NDJSON file.",default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/repair_img_clues")
    parser.add_argument("--dataset", "-d",   help="Path to input JSON or NDJSON file.",default="twitter2017")
    parser.add_argument("--output", "-o",   help="Path to output JSON or NDJSON file.",default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_img_clues")
    parser.add_argument("--keep-original", action="store_true", help="Keep original textual_clues/img_clue raw strings.",default=True)
    parser.add_argument("--no-flatten", action="store_true", help="Do not flatten; write *_parsed dict fields instead.")
    args = parser.parse_args()
    types=['train','val','test']
    input_dir=os.path.join(args.input,args.dataset)
    output_dir=os.path.join(args.output,args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for type in types:
        input_data_path=os.path.join(input_dir,f"{type}.json")
        output_data_path=os.path.join(output_dir,f"{type}.json")
    
        save_items(
            input_path=input_data_path,
            output_path=output_data_path,
            keep_original=args.keep_original,
            flatten=not args.no_flatten,
        )
        

        # 修复有问题的_raw:
        # repair_items(
        #     input_path=input_data_path,
        #     output_path=output_data_path,
        #     keep_original=args.keep_original,
        #     flatten=not args.no_flatten,
        # )
        print(f"Done. Wrote: {output_data_path}")

if __name__ == "__main__":
    main()

# python parse_gemini_data.py -i /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/merge_data/twitter2017/train.json   -o /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parse_merge_data/twitter2017/train.json --no-flatten
