export CUDA_VISIBLE_DEVICES=0
nohup python A2II/train.py \
    --dataset twitter2015\
    --data_dir /public/home/ghfu/lzy/code/instructBLIP/img_data \
    --img_feat_dir /public/home/ghfu/lzy/code/instructBLIP/img_data2 \
    --output_dir /public/home/ghfu/lzy/code/instructBLIP/results/twitter2015-improved-v1/ \
    --run_name twitter2015-improved-v1 \
    --BATCH_SIZE 8\
    --seed 42 \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5 \
    > twitter2015-improved-v1.log 2>&1 &
# nohup python A2II/train.py \
#     --dataset twitter2015\
#     --data_dir /public/home/ghfu/lzy/code/instructBLIP/img_data \
#     --output_dir /public/home/ghfu/lzy/code/instructBLIP/results/twitter2015-a2ii-run/ \
#     --run_name twitter2015-a2ii-run \
#     --BATCH_SIZE 8\
#     --seed 42 \
#     --EPOCHS 10 \
#     --LEARNING_RATE 5e-5 \
#     > twitter2015-a2ii-run.log 2>&1 &
