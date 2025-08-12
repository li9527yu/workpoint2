export CUDA_VISIBLE_DEVICES=0
#  -m debugpy --connect 127.0.0.1:2233
for dataset in "twitter2015"
do
    for seed in "42"
    do
    echo ${dataset}
    echo ${seed}
    python A2II_aspectCaption/train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/aspectCaption/${dataset}-run/${seed} \
    --BATCH_SIZE 8 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5  
    done
done

