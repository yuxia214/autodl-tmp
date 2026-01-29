#!/bin/bash
######################################################################################################################
############################ 针对模态缺失场景(test2)优化的训练脚本 #############################
######################################################################################################################

# ============================================
# 方案1: 降低正则化强度（推荐先试这个）
# - 降低dropout: 0.5 -> 0.4
# - 降低modality_dropout: 0.3 -> 0.15
# - 增加patience让模型充分训练
# ============================================
python -u main-robust.py \
    --model='attention_robust' \
    --feat_type='utt' \
    --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --hidden_dim=128 \
    --dropout=0.4 \
    --modality_dropout=0.15 \
    --use_modality_dropout \
    --lr=5e-4 \
    --l2=5e-5 \
    --grad_clip=1.0 \
    --epochs=100 \
    --early_stopping_patience=25 \
    --lr_patience=12 \
    --lr_factor=0.5 \
    --batch_size=32 \
    --gpu=0


# ============================================
# 方案2: 中等正则化（如果方案1过拟合）
# ============================================
# python -u main-robust.py \
#     --model='attention_robust' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --hidden_dim=128 \
#     --dropout=0.45 \
#     --modality_dropout=0.2 \
#     --use_modality_dropout \
#     --lr=5e-4 \
#     --l2=1e-4 \
#     --grad_clip=1.0 \
#     --epochs=100 \
#     --early_stopping_patience=25 \
#     --lr_patience=12 \
#     --batch_size=32 \
#     --gpu=0


# ============================================
# 方案3: 不使用模态dropout，只用增强正则化
# （测试是否模态dropout本身有问题）
# ============================================
# python -u main-robust.py \
#     --model='attention_robust' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --hidden_dim=128 \
#     --dropout=0.5 \
#     --no_modality_dropout \
#     --lr=5e-4 \
#     --l2=1e-4 \
#     --grad_clip=1.0 \
#     --epochs=100 \
#     --early_stopping_patience=25 \
#     --batch_size=32 \
#     --gpu=0


# ============================================
# 方案4: 更小的hidden_dim减少模型容量
# ============================================
# python -u main-robust.py \
#     --model='attention_robust' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --hidden_dim=64 \
#     --dropout=0.4 \
#     --modality_dropout=0.15 \
#     --use_modality_dropout \
#     --lr=5e-4 \
#     --l2=5e-5 \
#     --grad_clip=1.0 \
#     --epochs=100 \
#     --early_stopping_patience=25 \
#     --batch_size=32 \
#     --gpu=0
