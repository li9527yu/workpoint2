export CUDA_VISIBLE_DEVICES=0
# debug: -m debugpy --connect 127.0.0.1:2233

# 分析加入50%相关性标签噪声的影响
# for dataset in "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/relation_noise_50/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_noise_50%/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done

# # 分析加入40%相关性标签噪声的影响
# for dataset in "twitter2015" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/relation_noise_40/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_noise_40%/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done

# 分析加入20%相关性标签噪声的影响
for dataset in "twitter2017" 
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python train.py \
    --dataset ${dataset} \
    --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/relation_noise_20/  \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_noise_20%/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5 \
    --weight 0.5
    done
done

# 分析粗粒度的使用相关性（二分类标签）
# for dataset in "twitter2017" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_noise_coarseUse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done

# 分析加入30%相关性标签噪声的影响
# for dataset in "twitter2015" 
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/relation_noise/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_noise_30%/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done




# Prompt中文本图像情感线索混用
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/late_fuse_hybridClues/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done


# 在融合模态中额外显式加入相关性标签
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/late_fuse_explicitRelation/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done



#  在晚期融合中不使用相关性来控制情感线索的使用，直接暴力全部的使用情感线索 （Gemini的文本情感线索）
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_text_clues/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/late_fuse_emotion_clues_GeminiText/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done

# 0909:测试超参数：权重
# for weight in 0.2 0.3 
# do
#     for seed in "24"
#     do
#         echo ${seed}
#         python    train.py \
#         --dataset twitter2015 \
#         --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#         --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#         --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_weight_fuse/twitter2015_weight${weight}/${seed} \
#         --BATCH_SIZE 16 \
#         --seed ${seed} \
#         --EPOCHS 20 \
#         --LEARNING_RATE 5e-5 \
#         --weight ${weight}
#     done
# done

# twitter2017
# for weight in 0.7 0.8 
# do
#     for seed in "24"
#     do
#         echo ${seed}
#         python    train.py \
#         --dataset twitter2017 \
#         --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#         --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#         --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_weight_fuse/twitter2017_weight${weight}/${seed} \
#         --BATCH_SIZE 16 \
#         --seed ${seed} \
#         --EPOCHS 20 \
#         --LEARNING_RATE 5e-5 \
#         --weight ${weight}
#     done
# done


# debug: 
# # 8-19: 调整融合方式  "twitter2017"  
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python    train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_weight_fuse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 25 \
#     --LEARNING_RATE 5e-5
#     done
# done

# # 8-17:调整晚期融合结果
# for dataset in  "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/Late_fuse_8_17_v2/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done




# 8-15：考虑晚期融合的结果
# for dataset in  "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/Late_fuse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 15 \
#     --LEARNING_RATE 5e-5
#     done
# done


# # 新的双路融合实验：调整后的情感线索支路+图文理解组合输入
# for dataset in  "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/two_infuse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 15 \
#     --LEARNING_RATE 5e-5
#     done
# done
