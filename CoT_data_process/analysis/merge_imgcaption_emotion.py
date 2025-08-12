import pickle
import json

datasets=['twitter2017','twitter2015']
types=["train",'val', "test"]
for dataset in datasets:
    caption_dir_path=f'//data/lzy1211/code/A2II/instructBLIP/reason_data/data//{dataset}/captions.json'
    with open(caption_dir_path, 'r') as file:
        caption_data = json.load(file)
    for type in types:
        cnt=0
        emotion_dir_path=f'/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/new_data/{dataset}/new_{type}.pkl'
        with open(emotion_dir_path,'rb') as f:
            emotion_data = pickle.load(f)
        for key, value in emotion_data.items():
            caption=caption_data[key]
            value['image_caption']=caption
        
        with open(emotion_dir_path, 'wb') as f:
            pickle.dump(emotion_data, f)