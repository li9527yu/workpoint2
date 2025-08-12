import json
import os
from tqdm import tqdm

input_data_dir = "/data/lzy1211/code/annotation/data/"
types=['train', 'dev', 'test']
output_data_dir = "/data/lzy1211/code/A2II/instructBLIP/A2II_Three_AECR/data/"

text_clue_path='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/twitter2017/train.json'
with open(text_clue_path, 'r', encoding='utf-8') as infile:
        clues_data = json.load(infile)
def search_clue(item):
    for x in clues_data:
        if x['aspect'] == item['aspect'] and x['image'] == item['ImageID'].split('/')[-1]:
            return x['text_clue'], x['image_emotion']
    return None,None

for type in types:
    input_file = os.path.join(input_data_dir, f"{type}.json")
    output_file = os.path.join(output_data_dir, f"{type}.json")
   
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    for item in tqdm(data, desc=f"Processing {type} data"):
        text_clue,image_clue = search_clue(item)
        item['text_clue'] = text_clue
        item['image_clue'] = image_clue
    # Save the modified data to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"Processed {input_file} and saved to {output_file}")



 
