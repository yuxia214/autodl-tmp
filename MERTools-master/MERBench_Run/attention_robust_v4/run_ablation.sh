#!/bin/bash
# ============================================================
# AttentionRobustV4 消融实验脚本
# ============================================================
#
# 包含5组实验:
# 1. 仅模态Dropout (V1基准)
# 2. VAE编码器 + 不确定性加权
# 3. VAE + 代理模态注意力
# 4. 完整V2模型 (推荐)
# 5. 参数敏感性分析
#
# ============================================================

# ==================== 路径配置 ====================
MERBENCH_ROOT="/root/autodl-tmp/MERTools/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v4/outputs"

mkdir -p ${OUTPUT_DIR}/ablation_logs

# ==================== 通用配置 ====================
GPU_ID=0
DATASET="MER2023"
AUDIO_FEAT="chinese-hubert-large-UTT"
TEXT_FEAT="Baichuan-13B-Base-UTT"
VIDEO_FEAT="clip-vit-large-patch14-UTT"
FEAT_TYPE="utt"

cd ${MERBENCH_ROOT}

# ==================== 实验1: V1基准 (仅模态Dropout) ====================
echo "=========================================="
echo "实验1: V1基准 (仅模态Dropout)"
echo "=========================================="

python -u main-robust.py \
    --model='attention_robust' \
    --dataset=${DATASET} \
    --feat_type=${FEAT_TYPE} \
    --audio_feature=${AUDIO_FEAT} \
    --text_feature=${TEXT_FEAT} \
    --video_feature=${VIDEO_FEAT} \
    --save_root=${OUTPUT_DIR}/exp1_baseline \
    --hidden_dim=128 --dropout=0.5 \
    --modality_dropout=0.3 \
    --lr=5e-4 --l2=1e-4 \
    --epochs=100 --early_stopping_patience=20 \
    --gpu=${GPU_ID} \
    2>&1 | tee ${OUTPUT_DIR}/ablation_logs/exp1_baseline.log

# ==================== 实验2: VAE编码器 + 不确定性加权 (无代理注意力) ====================
echo "=========================================="
echo "实验2: VAE编码器 + 不确定性加权"
echo "=========================================="

python -u main-robust.py \
    --model='attention_robust_v4' \
    --dataset=${DATASET} \
    --feat_type=${FEAT_TYPE} \
    --audio_feature=${AUDIO_FEAT} \
    --text_feature=${TEXT_FEAT} \
    --video_feature=${VIDEO_FEAT} \
    --save_root=${OUTPUT_DIR}/exp2_vae_only \
    --hidden_dim=128 --dropout=0.35 \
    --use_vae --kl_weight=0.01 --recon_weight=0.1 \
    --no_proxy_attention \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --lr=5e-4 --l2=5e-5 \
    --epochs=100 --early_stopping_patience=30 \
    --gpu=${GPU_ID} \
    2>&1 | tee ${OUTPUT_DIR}/ablation_logs/exp2_vae_only.log

# ==================== 实验3: VAE + 代理模态注意力 ====================
echo "=========================================="
echo "实验3: VAE + 代理模态注意力"
echo "=========================================="

python -u main-robust.py \
    --model='attention_robust_v4' \
    --dataset=${DATASET} \
    --feat_type=${FEAT_TYPE} \
    --audio_feature=${AUDIO_FEAT} \
    --text_feature=${TEXT_FEAT} \
    --video_feature=${VIDEO_FEAT} \
    --save_root=${OUTPUT_DIR}/exp3_with_proxy \
    --hidden_dim=128 --dropout=0.35 \
    --use_vae --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --num_attention_heads=4 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --lr=5e-4 --l2=5e-5 \
    --epochs=100 --early_stopping_patience=30 \
    --gpu=${GPU_ID} \
    2>&1 | tee ${OUTPUT_DIR}/ablation_logs/exp3_with_proxy.log

# ==================== 实验4: 完整V2模型 (调优参数) ====================
echo "=========================================="
echo "实验4: 完整V2模型 (调优参数)"
echo "=========================================="

python -u main-robust.py \
    --model='attention_robust_v4' \
    --dataset=${DATASET} \
    --feat_type=${FEAT_TYPE} \
    --audio_feature=${AUDIO_FEAT} \
    --text_feature=${TEXT_FEAT} \
    --video_feature=${VIDEO_FEAT} \
    --save_root=${OUTPUT_DIR}/exp4_full_v2 \
    --hidden_dim=128 --dropout=0.4 \
    --use_vae --kl_weight=0.005 --recon_weight=0.15 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=0.8 --num_attention_heads=4 \
    --modality_dropout=0.2 --modality_dropout_warmup=15 \
    --lr=3e-4 --l2=1e-4 \
    --epochs=150 --early_stopping_patience=40 \
    --gpu=${GPU_ID} \
    2>&1 | tee ${OUTPUT_DIR}/ablation_logs/exp4_full_v2.log

# ==================== 实验5: 无VAE (仅代理注意力 - 对比实验) ====================
echo "=========================================="
echo "实验5: 无VAE模式 (对比)"
echo "=========================================="

python -u main-robust.py \
    --model='attention_robust_v4' \
    --dataset=${DATASET} \
    --feat_type=${FEAT_TYPE} \
    --audio_feature=${AUDIO_FEAT} \
    --text_feature=${TEXT_FEAT} \
    --video_feature=${VIDEO_FEAT} \
    --save_root=${OUTPUT_DIR}/exp5_no_vae \
    --hidden_dim=128 --dropout=0.35 \
    --no_vae \
    --modality_dropout=0.2 \
    --lr=5e-4 --l2=5e-5 \
    --epochs=100 --early_stopping_patience=30 \
    --gpu=${GPU_ID} \
    2>&1 | tee ${OUTPUT_DIR}/ablation_logs/exp5_no_vae.log

echo ""
echo "=============================================="
echo "所有消融实验完成!"
echo "=============================================="
echo "结果保存在: ${OUTPUT_DIR}"
echo ""
echo "各实验结果:"
echo "  实验1 (V1基准): ${OUTPUT_DIR}/exp1_baseline"
echo "  实验2 (VAE only): ${OUTPUT_DIR}/exp2_vae_only"
echo "  实验3 (VAE+Proxy): ${OUTPUT_DIR}/exp3_with_proxy"
echo "  实验4 (完整V2): ${OUTPUT_DIR}/exp4_full_v2"
echo "  实验5 (无VAE): ${OUTPUT_DIR}/exp5_no_vae"
