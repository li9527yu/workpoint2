import json
import os

def load_json_data(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_data(file_name, data):
    with open(file_name, "w") as f:
        json.dump(data, f,ensure_ascii=False,indent=2)


input_dir="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/gemini_data/twitter2017"
left_input_dir="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_img_clues/twitter2017"
output_dir="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/repair_img_clues/twitter2017"
types=["train","val","test"]
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
for type in types:
    input_path=os.path.join(input_dir,f"{type}.json")
    left_input_path=os.path.join(left_input_dir,f"{type}.json")
    output_path=os.path.join(output_dir,f"{type}.json")
    
    origin_data=load_json_data(input_path)
    left_data=load_json_data(left_input_path)

    for item in origin_data:
        for left_item in left_data:
            if left_item['image']==item['image'] and left_item['aspect']==item['aspect']:
                item['img_clue']=left_item['img_clue']
    save_json_data(output_path,origin_data)
    print(f"{type} Done")

