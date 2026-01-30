#!/bin/bash
# =====================================================
# AttentionRobustV2 实验脚本
# 基于P-RMF的概率化多模态融合模型
# 目标：提升test2（模态缺失测试）的准确率
# =====================================================

# 基础配置
DATASET="MER2023"
AUDIO_FEATURE="chinese-hubert-large-UTT"
TEXT_FEATURE="Baichuan-13B-Base-UTT"
VIDEO_FEATURE="clip-vit-large-patch14-UTT"
GPU=0

# =====================================================
# 实验1: 基础VAE版本 (不使用代理注意力)
# 预期: test2 ~0.76
# =====================================================
echo "==================== 实验1: VAE基础版 ===================="
python -u main-robust.py \
    --model='attention_robust_v2' \
    --feat_type='utt' \
    --dataset=${DATASET} \
    --audio_feature=${AUDIO_FEATURE} \
    --text_feature=${TEXT_FEATURE} \
    --video_feature=${VIDEO_FEATURE} \
    --hidden_dim=128 \
    --dropout=0.35 \
    --use_vae \
    --kl_weight=0.01 \
    --recon_weight=0.1 \
    --cross_kl_weight=0.01 \
    --no_proxy_attention \
    --modality_dropout=0.15 \
    --use_modality_dropout \
    --modality_dropout_warmup=20 \
    --lr=5e-4 \
    --l2=5e-5 \
    --grad_clip=1.0 \
    --epochs=100 \
    --early_stopping_patience=30 \
    --lr_patience=15 \
    --batch_size=32 \
    --gpu=${GPU}

# =====================================================
# 实验2: 完整版 (VAE + 代理模态注意力)
# 预期: test2 ~0.77-0.78
# =====================================================
echo "==================== 实验2: VAE + Proxy Attention ===================="
python -u main-robust.py \
    --model='attention_robust_v2' \
    --feat_type='utt' \
    --dataset=${DATASET} \
    --audio_feature=${AUDIO_FEATURE} \
    --text_feature=${TEXT_FEATURE} \
    --video_feature=${VIDEO_FEATURE} \
    --hidden_dim=128 \
    --dropout=0.35 \
    --use_vae \
    --kl_weight=0.01 \
    --recon_weight=0.1 \
    --cross_kl_weight=0.01 \
    --use_proxy_attention \
    --fusion_temperature=1.0 \
    --num_attention_heads=4 \
    --modality_dropout=0.15 \
    --use_modality_dropout \
    --modality_dropout_warmup=20 \
    --lr=5e-4 \
    --l2=5e-5 \
    --grad_clip=1.0 \
    --epochs=100 \
    --early_stopping_patience=30 \
    --lr_patience=15 \
    --batch_size=32 \
    --gpu=${GPU}

# =====================================================
# 实验3: 调优KL权重 (更小的KL权重)
# =====================================================
echo "==================== 实验3: 小KL权重 ===================="
python -u main-robust.py \
    --model='attention_robust_v2' \
    --feat_type='utt' \
    --dataset=${DATASET} \
    --audio_feature=${AUDIO_FEATURE} \
    --text_feature=${TEXT_FEATURE} \
    --video_feature=${VIDEO_FEATURE} \
    --hidden_dim=128 \
    --dropout=0.35 \
    --use_vae \
    --kl_weight=0.005 \
    --recon_weight=0.1 \
    --cross_kl_weight=0.005 \
    --use_proxy_attention \
    --fusion_temperature=1.0 \
    --modality_dropout=0.15 \
    --modality_dropout_warmup=25 \
    --lr=5e-4 \
    --l2=5e-5 \
    --epochs=100 \
    --early_stopping_patience=30 \
    --gpu=${GPU}

# =====================================================
# 实验4: 调优融合温度 (更小的温度 = 更极端的权重)
# =====================================================
echo "==================== 实验4: 低温度融合 ===================="
python -u main-robust.py \
    --model='attention_robust_v2' \
    --feat_type='utt' \
    --dataset=${DATASET} \
    --audio_feature=${AUDIO_FEATURE} \
    --text_feature=${TEXT_FEATURE} \
    --video_feature=${VIDEO_FEATURE} \
    --hidden_dim=128 \
    --dropout=0.35 \
    --use_vae \
    --kl_weight=0.01 \
    --recon_weight=0.1 \
    --cross_kl_weight=0.01 \
    --use_proxy_attention \
    --fusion_temperature=0.5 \
    --modality_dropout=0.15 \
    --modality_dropout_warmup=20 \
    --lr=5e-4 \
    --l2=5e-5 \
    --epochs=100 \
    --early_stopping_patience=30 \
    --gpu=${GPU}

# =====================================================
# 实验5: 对比基线 (不使用VAE，保持原有融合方式)
# =====================================================
echo "==================== 实验5: 基线对比 (无VAE) ===================="
python -u main-robust.py \
    --model='attention_robust_v2' \
    --feat_type='utt' \
    --dataset=${DATASET} \
    --audio_feature=${AUDIO_FEATURE} \
    --text_feature=${TEXT_FEATURE} \
    --video_feature=${VIDEO_FEATURE} \
    --hidden_dim=128 \
    --dropout=0.35 \
    --no_vae \
    --modality_dropout=0.2 \
    --modality_dropout_warmup=30 \
    --lr=5e-4 \
    --l2=5e-5 \
    --epochs=100 \
    --early_stopping_patience=30 \
    --gpu=${GPU}

echo "==================== 所有实验完成 ===================="
