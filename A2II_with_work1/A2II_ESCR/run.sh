export CUDA_VISIBLE_DEVICES=0
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 


for seed in "24" "1000" "3000"
    do
        echo "seed:${seed}"
        python  train.py \
            --data_dir /data/lzy1211/code/annotation/data \
            --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-ESCR/8-1e-4/${seed} \
            --BATCH_SIZE 8 \
            --seed ${seed} \
            --EPOCHS 15 \
            --LEARNING_RATE 1e-4
done

# for bs in "8" "16" 
# do
#     for lr in "1e-4" "5e-5"
#     do
#         echo "Batch Size: ${bs}, Learning Rate: ${lr}"
#         python  train.py \
#             --data_dir /data/lzy1211/code/annotation/data \
#             --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-ESCR/${bs}-${lr}/ \
#             --BATCH_SIZE ${bs} \
#             --seed 42 \
#             --EPOCHS 15 \
#             --LEARNING_RATE ${lr} 
#     done
# done


# # 
# python  train.py \
#     --data_dir /data/lzy1211/code/annotation/data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-ESCR/gold-6-2/ \
#     --BATCH_SIZE 16\
#     --seed 24 \
#     --EPOCHS 20 \
#     --LEARNING_RATE 1e-4 \

# 5-16:跑原始的加入相关性的实验(更换计算指标方式)
# for dataset in "twitter2015"  "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-FirstWork-changeMetric/${dataset}-${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done


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
