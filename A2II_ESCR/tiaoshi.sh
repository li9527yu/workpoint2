export CUDA_VISIBLE_DEVICES=7
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 
# -m debugpy --connect 127.0.0.1:2233 A2II_CoT_ESCR/

# -m debugpy --connect 127.0.0.1:2233 A2II_Three_AECR/
for seed in "42"
    do
    echo ${seed}
    python  train.py \
    --data_dir /data/lzy1211/code/A2II/instructBLIP/A2II_Three_AECR/emotion_expaneded_data \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ESCR/only_image_text_meaning/${seed} \
    --BATCH_SIZE 16\
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
