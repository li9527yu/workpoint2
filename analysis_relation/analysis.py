import json

# 定义文件路径
# new_test_path = "/data/lzy1211/code/A2II/instructBLIP/analysis_relation/new_relation/twitter2017/new_test.json"
# test_process_path = "/data/lzy1211/code/A2II/instructBLIP/analysis_relation/twitter2017/test_process.json"
new_train='/data/lzy1211/code/A2II/instructBLIP/analysis_relation/new_relation/twitter2017/new_train.json'
new_val='/data/lzy1211/code/A2II/instructBLIP/analysis_relation/new_relation/twitter2017/new_val.json'
train_path='/data/lzy1211/code/A2II/instructBLIP/analysis_relation/twitter2017/train_process.json'

def get_rel(relation):
    semantic_rel,emotional_rel=relation.split(',')
    semantic_relation_label,emotional_relation_label=0,0
    if 'irrelevant' in semantic_rel:
        semantic_relation_label=0
    else:
        semantic_relation_label=1

    if 'irrelevant' in emotional_rel:
        emotional_relation_label=0
    else:
        emotional_relation_label=1
   
    return semantic_relation_label,emotional_relation_label


# 读取 JSON 文件的函数
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 从文件中读取数据
new_train_data=read_json(new_train)
new_val_data=read_json(new_val)
train_data=read_json(train_path)

# 打印读取的数据（可选）
# print("New Test Data:", new_test_data)
# print("Test Process Data:", test_process_data)

result=[]
different=[]


for item1,item2 in zip(new_train_data,train_data):
    relation1=item1['relation']
    relation2=item2['conversations'][0]['relation']
    resItem={
        'relation1':relation1,
        'relation2':relation2,
        'aspect':item1['aspect'],
        'sentence':item1['text'],
        'img':item1['image']
    }
    sr,er=get_rel(relation1)
    if sr==0 and er==0:
        res=0
    else:
        res=1
    if 'relevant' in relation2 :
        res2=1
    elif 'irrelvant' in relation2:
        res2=0
    if res2!=res:
        different.append(resItem)
    result.append(resItem)

print(len(different))
print("!")

