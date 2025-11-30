export CUDA_VISIBLE_DEVICES=6
#  -m debugpy --connect 127.0.0.1:2233


for dataset in  "twitter2017"  
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python   train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/ \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-ThreeInfo/fuse_improved_v1/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
done


# for dataset in  "twitter2015"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python    train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/ \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-ThreeInfo/text_clue/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done
