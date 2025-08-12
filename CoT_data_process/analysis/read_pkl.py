import pickle



types=["train",'val', "test"]
for dataset in ['twitter2015','twitter2017']:
    print(dataset)
    for type in types:
        cnt=0
        # /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data/twitter2015/new_train.pkl
        dir_path=f'/data/lzy1211/code/A2II/instructBLIP/img_data/twitter2015/test.pkl'
        with open(dir_path,'rb') as f:
            data = pickle.load(f)
        print(len(data))
        # for key, value in data.items():
        #     if 'text_description' not in value:
        #         print("no key text_description")
        #         cnt+=1
        #     if value['text_description'] is None:
        #         print(" text_description is None ")
        #         # print(key)
        #         cnt+=1
        # for key, value in data.items():
        #     if 'image_emotion' not in value:
        #         print("no key image_emotion")
        #         cnt+=1
        #     if value['image_emotion'] is None:
        #         print(" image_emotion is None ")
        #         # print(key)
        #         cnt+=1
        print(cnt)