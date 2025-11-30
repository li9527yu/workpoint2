export CUDA_VISIBLE_DEVICES=4
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 



# base model
for dataset in "twitter2015" "twitter2017"
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python   train.py \
    --dataset ${dataset} \
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_text_clues/ \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-Ablation/Base_model/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
done
 