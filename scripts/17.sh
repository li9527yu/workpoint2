export CUDA_VISIBLE_DEVICES=1
nohup python A2II/train.py \
    --dataset twitter2017\
    --data_dir /public/home/ghfu/lzy/code/instructBLIP/img_data \
    --output_dir /public/home/ghfu/lzy/code/instructBLIP/results/twitter2017-a2ii-run \
    --run_name twitter2017-a2ii-run \
    --BATCH_SIZE 8\
    --seed 42 \
    --EPOCHS 10 \
    --LEARNING_RATE 5e-5 \
    > twitter2017-a2ii-run.log 2>&1 &