export CUDA_VISIBLE_DEVICES=4
# debug: -m debugpy --connect 127.0.0.1:2233

for dataset in "twitter2015"  "twitter2017"  
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python train.py \
    --dataset ${dataset} \
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-Ablation/Base_model+Clues+Wo_rel/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5 \
    --weight 1
    done
done