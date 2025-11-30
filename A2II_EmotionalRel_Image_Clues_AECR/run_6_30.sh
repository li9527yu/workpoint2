export CUDA_VISIBLE_DEVICES=2
# 提供文本情感和图像情感，根据相关性选择性使用
for dataset in  "twitter2015"  
do
    for seed in "42"
    do
    echo ${dataset}
    echo ${seed}
    python  train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/text_emotion/  \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/text_image_emotion/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
done
