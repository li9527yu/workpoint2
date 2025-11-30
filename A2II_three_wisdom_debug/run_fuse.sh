export CUDA_VISIBLE_DEVICES=7

# -m debugpy --connect 127.0.0.1:2233
# 8-17,8-18:在相关性支路下，使用wisdom的上下文融合策略
for dataset in  "twitter2015"  
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python -m debugpy --connect 127.0.0.1:2233 train.py \
    --dataset ${dataset} \
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/wisdom_fuse_debug/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
done



# debug: -m debugpy --connect 127.0.0.1:2233
# 8-15：考虑晚期融合的结果
# for dataset in  "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/Late_fuse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 15 \
#     --LEARNING_RATE 5e-5
#     done
# done


# # 新的双路融合实验：调整后的情感线索支路+图文理解组合输入
# for dataset in  "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/two_infuse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 15 \
#     --LEARNING_RATE 5e-5
#     done
# done
