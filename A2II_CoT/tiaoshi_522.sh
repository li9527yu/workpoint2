export CUDA_VISIBLE_DEVICES=0
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 

# 5-22：二分类只使用图像情感
for dataset in  "twitter2015" "twitter2017" 
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python  train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-emotionInfo/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
done
# # 5-22：三分类（情感相关使用Image Emotion），语义相关使用Image Description 无关都使用
# for dataset in  "twitter2015" "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo+emotion/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

