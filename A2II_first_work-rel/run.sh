export CUDA_VISIBLE_DEVICES=0
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 


# 5-16:跑原始的加入相关性的实验(更换计算指标方式)
for dataset in "twitter2015"  "twitter2017" 
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python  train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-FirstWork-changeMetric/${dataset}-${seed} \
    --BATCH_SIZE 16\
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
done


# 5-16:跑原始的加入相关性的实验
# for dataset in "twitter2015"  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-FirstWork/${dataset}-${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done
