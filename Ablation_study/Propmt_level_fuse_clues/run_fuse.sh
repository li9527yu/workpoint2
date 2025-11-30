export CUDA_VISIBLE_DEVICES=5

# 调整Prompt 融合图文情感线索的性能
for dataset in  "twitter2015" "twitter2017"
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python  train.py \
    --dataset ${dataset} \
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-Ablation/Propmt_level_fuse_clues/${dataset}/${seed} \
    --BATCH_SIZE 8 \
    --seed ${seed} \
    --EPOCHS 15 \
    --LEARNING_RATE 2e-5
    done
done