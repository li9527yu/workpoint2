export CUDA_VISIBLE_DEVICES=3
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 


# 5-15：相关、无关的数据构造：加入额外信息：Image Caption Image Emotion
# for dataset in "twitter2015"  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-MoreImageInfoTransform-${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 10 \
#     --LEARNING_RATE 1e-4  
#     done
# done


# # 5-15：相关、无关的数据构造：加入额外信息：Text Description Image Caption
# for dataset in "twitter2015"  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-MoreInfoTransform-${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 10 \
#     --LEARNING_RATE 1e-4  
#     done
# done
# 5-15：第一个工作中相关、无关的数据构造
# for dataset in "twitter2015"  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-originalTransform-${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 10 \
#     --LEARNING_RATE 1e-4  
#     done
# done

