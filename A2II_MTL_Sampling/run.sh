export CUDA_VISIBLE_DEVICES=7
#  -m debugpy --connect 127.0.0.1:2233
for dataset in "twitter2017"
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-MTL-${dataset}/${seed} \
    --BATCH_SIZE 32\
    --seed ${seed} \
    --EPOCHS 5 \
    --LEARNING_RATE 5e-5  
    done
done
