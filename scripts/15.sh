export CUDA_VISIBLE_DEVICES=0
#  -m debugpy --connect 127.0.0.1:2233
for dataset in "twitter2017"
do
    for seed in "24" "83" "42"
    do
    echo ${dataset}
    echo ${seed}
    python A2II/train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/${dataset}-run/${seed} \
    --run_name debug \
    --BATCH_SIZE 16\
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5  
    done
done

# python  A2II/train.py \
#     --dataset twitter2015\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/twitter2015-sweep/ \
#     --run_name twitter2015-a2ii-run \
#     --BATCH_SIZE 8\
#     --seed 42 \
#     --EPOCHS 20 \
#     --LEARNING_RATE 2e-5  
# nohup python A2II/train.py \
#     --dataset twitter2015\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/twitter2015-a2ii-run/ \
#     --run_name twitter2015-a2ii-run \
#     --BATCH_SIZE 8\
#     --seed 42 \
#     --EPOCHS 10 \
#     --LEARNING_RATE 5e-5 \
#     > twitter2015-a2ii-run.log 2>&1 &
