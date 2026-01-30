#!/bin/bash
######################################################################################################################
############################ 针对模态缺失场景(test2)优化的训练脚本 #############################
######################################################################################################################

# ============================================
# 方案A: 渐进式模态dropout（推荐）
# - 先让模型学习完整模态信息（warmup 30 epochs）
# - 然后逐渐增加模态dropout
# - 结合原始attention的高test2 + 改进的泛化能力
# ============================================
python -u main-robust.py \
    --model='attention_robust' \
    --feat_type='utt' \
    --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --hidden_dim=128 \
    --dropout=0.35 \
    --modality_dropout=0.2 \
    --modality_dropout_warmup=30 \
    --use_modality_dropout \
    --lr=5e-4 \
    --l2=5e-5 \
    --grad_clip=1.0 \
    --epochs=100 \
    --early_stopping_patience=30 \
    --lr_patience=15 \
    --lr_factor=0.5 \
    --batch_size=32 \
    --gpu=0


# ============================================
# 方案B: 不使用模态dropout，只用早停控制过拟合
# （验证是否模态dropout本身对test2有害）
# ============================================
# python -u main-robust.py \
#     --model='attention_robust' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --hidden_dim=128 \
#     --dropout=0.35 \
#     --no_modality_dropout \
#     --lr=5e-4 \
#     --l2=5e-5 \
#     --grad_clip=1.0 \
#     --epochs=100 \
#     --early_stopping_patience=30 \
#     --lr_patience=15 \
#     --batch_size=32 \
#     --gpu=0


# ============================================
# 方案C: 更长的warmup + 更小的modality dropout
# ============================================
# python -u main-robust.py \
#     --model='attention_robust' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --hidden_dim=128 \
#     --dropout=0.3 \
#     --modality_dropout=0.1 \
#     --modality_dropout_warmup=40 \
#     --use_modality_dropout \
#     --lr=5e-4 \
#     --l2=5e-5 \
#     --grad_clip=1.0 \
#     --epochs=100 \
#     --early_stopping_patience=35 \
#     --lr_patience=15 \
#     --batch_size=32 \
#     --gpu=0


# ============================================
# 方案D: 使用原始main-release.py但减少epochs
# （对比基准：早停版原始attention）
# ============================================
# python -u main-release.py \
#     --model='attention' \
#     --feat_type='utt' \
#     --dataset='MER2023' \
#     --audio_feature='chinese-hubert-large-UTT' \
#     --text_feature='Baichuan-13B-Base-UTT' \
#     --video_feature='clip-vit-large-patch14-UTT' \
#     --epochs=50 \
#     --batch_size=32 \
#     --gpu=0
