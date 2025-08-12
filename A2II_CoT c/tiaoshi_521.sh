export CUDA_VISIBLE_DEVICES=6
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 

# 5-21：有关无关都加入额外信息imagetext_meaning +有关无关都使用Image Caption Image Emotion
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
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-AllInfo/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
done

# # 5-21：三分类（情感相关使用Image Emotion），其他情况都使用Image Description  twitter2017
# for dataset in  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreenInfo/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

# 5-21：使用最新的相关性(二分类）中twitter17数据集性能异常的排查 换种方式
# for dataset in  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-originalReproduce/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

# # 5-21：使用最新的相关性(二分类）中twitter17数据集性能异常的排查 仅考虑 emotional_relation_label
# for dataset in  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-originalReproduce/${dataset}-emotional_relation_label/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

# # 5-21：使用最新的相关性(二分类）中twitter17数据集性能异常的排查 仅考虑semantic_relation_label
# for dataset in  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-originalReproduce/${dataset}-semantic_relation_label/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

# # 5-21：使用最新的相关性(二分类）中twitter17数据集性能异常的排查
# for dataset in  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-originalReproduce/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

# # 5-20：有关无关都加入额外信息：Image Caption Image Emotion 并使用aspect Context
# for dataset in "twitter2015" "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ImageInfo+Aspect/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

# # 5-20：twitter2017-有关无关都加入额外信息：Image Caption Image Emotion
# for dataset in "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ImageInfo/${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done
