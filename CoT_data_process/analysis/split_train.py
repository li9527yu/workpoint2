import json
import os


dir='/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/sentiment_relation_twitter_output'
dataset='twitter2017'
path='train_result.json'
with open(os.path.join(dir,dataset, path), 'r') as f:
    data = json.load(f)
# index=3179
index=3562
train=data[:index]
val=data[index:]
# print(len(val))
output_train_file=os.path.join(dir,dataset, path)
output_val_file=os.path.join(dir,dataset, 'val_result.json')
with open(output_train_file, 'w') as f:
    json.dump(train,f)
with open(output_val_file, 'w') as f:
    json.dump(val,f)
print("done")
