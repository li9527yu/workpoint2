import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

file_path = "/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_data/twitter2017/new_test.pkl"
data = load_json(file_path)
print(type(data))
print(data)
