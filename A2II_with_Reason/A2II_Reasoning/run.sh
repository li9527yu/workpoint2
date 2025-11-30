export CUDA_VISIBLE_DEVICES=5
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 
for dataset in "AECR"  
do
    for seed in "24"
    do

    python -m debugpy --connect 127.0.0.1:2233 train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/A2II_Three_AECR/emotion_expaneded_data \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-Reasoning-AECR/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 15 \
    --LEARNING_RATE 5e-5  
    done
done
