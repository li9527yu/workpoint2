# export CUDA_VISIBLE_DEVICES=1
# 在模型部分进行改进：加入一组额外特征
nohup python A2II_c/train.py \
    --dataset twitter2017\
    --data_dir /public/home/ghfu/lzy/code/instructBLIP/img_data \
    --img_feat_dir /public/home/ghfu/lzy/code/instructBLIP/img_data2 \
    --output_dir /public/home/ghfu/lzy/code/instructBLIP/results/twitter2017-improved-v1/ \
    --run_name twitter2017-a2ii-run \
    --BATCH_SIZE 8\
    --seed 42 \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5 \
    > twitter2017-improved-v1.log 2>&1 &