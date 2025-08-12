export CUDA_VISIBLE_DEVICES=0
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 

# 在单路表现最好的情感线索组合输入：只在无关类型中加入caption
for dataset in "twitter2015" "twitter2017"
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python   train.py \
    --dataset ${dataset}\
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/ \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/emotion+onlyRelationS=1/${dataset}/${seed} \
    --BATCH_SIZE 16\
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5
    done
done

# 在单路表现最好的情感线索组合输入：只在无关类型中加入caption
# for dataset in "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python   train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/ \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/emotion+onlyIrrelevant/${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

# # 在单路表现最好的情感线索组合输入中 使用vit的图像特征 
# for dataset in "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python   train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/ \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/emotion+wVIT/${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done


# 更新了下CoT版本下的图文理解描述
# for dataset in  "twitter2015"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python    train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/sentiment_meaning \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/CoT_sentiment_meaning/${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done


# 使用图像特征时，调试考虑文本、图像情感线索的最好输入组合
# for dataset in "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python    train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/ \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/modal_emotion_clue/${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done

# 不使用图像特征的Twitter表现最好的输入组合
# for dataset in "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python    train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/ \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/emotion+caption+woImg/${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done


#  图文理解
# for dataset in "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python    train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/imagetext_meaning/${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done




# # 7-17:在所有的相关性都使用相同的知识（图像描述+图像情感 考虑双路融合模型
# for dataset in "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python    train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/emotion+caption/${dataset}/${seed} \
#     --BATCH_SIZE 16\
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done