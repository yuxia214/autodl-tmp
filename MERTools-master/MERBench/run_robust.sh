#!/bin/bash
######################################################################################################################
############################ 针对模态缺失场景(test2)优化的训练脚本 #############################
######################################################################################################################

# ============================================
# 方案1: 使用 attention_robust 模型 (推荐)
# - 包含模态dropout训练策略
# - 更强的正则化
# - 早停机制
# ============================================

# 基础配置 - 使用固定优化后的超参数
python -u main-robust.py \
    --model='attention_robust' \
    --feat_type='utt' \
    --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --hidden_dim=128 \
    --dropout=0.5 \
    --modality_dropout=0.3 \
    --use_modality_dropout \
    --lr=5e-4 \
    --l2=1e-4 \
    --grad_clip=1.0 \
    --epochs=100 \
    --early_stopping_patience=20 \
    --lr_patience=10 \
    --lr_factor=0.5 \
    --batch_size=32 \
    --gpu=0


# ============================================
# 方案2: 更高的模态dropout比例 (如果test2提升不够)
# ============================================
# python -u main-robust.py \
#     --model='attention_robust' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --hidden_dim=128 \
#     --dropout=0.6 \
#     --modality_dropout=0.4 \
#     --use_modality_dropout \
#     --lr=3e-4 \
#     --l2=5e-4 \
#     --grad_clip=1.0 \
#     --epochs=100 \
#     --early_stopping_patience=15 \
#     --lr_patience=8 \
#     --batch_size=32 \
#     --gpu=0


# ============================================
# 方案3: 更小的hidden_dim (减少过拟合)
# ============================================
# python -u main-robust.py \
#     --model='attention_robust' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --hidden_dim=64 \
#     --dropout=0.5 \
#     --modality_dropout=0.3 \
#     --lr=5e-4 \
#     --l2=1e-4 \
#     --grad_clip=1.0 \
#     --epochs=100 \
#     --early_stopping_patience=20 \
#     --batch_size=32 \
#     --gpu=0


# ============================================
# 方案4: 使用原始attention模型但增强正则化
# (通过修改main-release.py的参数)
# ============================================
# python -u main-release.py \
#     --model='attention' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --lr=1e-4 \
#     --l2=1e-3 \
#     --epochs=50 \
#     --batch_size=32 \
#     --gpu=0
