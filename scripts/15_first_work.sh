export CUDA_VISIBLE_DEVICES=0
python -m debugpy --connect 127.0.0.1:2233  A2II_first_work/train.py \
    --dataset twitter2015\
    --data_dir /public/home/ghfu/lzy/code/instructBLIP/img_data \
    --Re_data_dir /public/home/ghfu/lzy/code/instructBLIP/relation_data \
    --img_feat_dir /public/home/ghfu/lzy/code/instructBLIP/img_feat \
    --output_dir /public/home/ghfu/lzy/code/instructBLIP/results/twitter2015-a2ii-first_work/ \
    --run_name first_work \
    --BATCH_SIZE 8\
    --seed 42 \
    --EPOCHS 5 \
    --LEARNING_RATE 5e-5 \
    --Rel_LEARNING_RATE 1e-5  