#!/bin/bash
# AttentionRobust V10 AV-only formal training

set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v10/outputs"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/results"

GPU_ID=0
DATASET="MER2023"

AUDIO_FEAT="chinese-hubert-large-UTT"
VIDEO_FEAT="clip-vit-large-patch14-UTT"
FEAT_TYPE="utt"

HIDDEN_DIM=128
DROPOUT=0.35

KL_WEIGHT=0.01
RECON_WEIGHT=0.10
CROSS_KL_WEIGHT=0.005
KL_WARMUP=20

FUSION_TEMP=1.0
NUM_HEADS=4

# AV-only时降低dropout强度，减少双模态信息损失
MODALITY_DROPOUT=0.08
WARMUP_EPOCHS=10

EMO_LOSS_WEIGHT=1.0
VAL_LOSS_WEIGHT=1.4
REG_LOSS_TYPE="smoothl1"
HUBER_BETA=0.8

VAL_CONSISTENCY_WEIGHT=0.10
VAL_CENTER_REG_WEIGHT=0.005
FEATURE_NOISE_STD=0.02
FEATURE_NOISE_PROB=0.30
FEATURE_NOISE_WARMUP=3

LR=5e-4
L2=5e-5
EPOCHS=100
BATCH_SIZE=32
EARLY_STOP=25
LR_PATIENCE=8

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/logs/train_v10_formal_${TIMESTAMP}.log"

cd "${MERBENCH_ROOT}"

echo "=============================================="
echo "AttentionRobustV10 AV-only Formal Training Start"
echo "Time: ${TIMESTAMP}"
echo "Log: ${LOG_FILE}"
echo "=============================================="

python -u main-robust.py \
    --model='attention_robust_v10' \
    --dataset="${DATASET}" \
    --feat_type="${FEAT_TYPE}" \
    --audio_feature="${AUDIO_FEAT}" \
    --video_feature="${VIDEO_FEAT}" \
    --save_root="${OUTPUT_DIR}/results" \
    --hidden_dim="${HIDDEN_DIM}" \
    --dropout="${DROPOUT}" \
    --use_vae \
    --kl_weight="${KL_WEIGHT}" \
    --recon_weight="${RECON_WEIGHT}" \
    --cross_kl_weight="${CROSS_KL_WEIGHT}" \
    --use_dynamic_kl \
    --kl_warmup_epochs="${KL_WARMUP}" \
    --use_proxy_attention \
    --fusion_temperature="${FUSION_TEMP}" \
    --num_attention_heads="${NUM_HEADS}" \
    --modality_dropout="${MODALITY_DROPOUT}" \
    --modality_dropout_warmup="${WARMUP_EPOCHS}" \
    --emo_loss_weight="${EMO_LOSS_WEIGHT}" \
    --val_loss_weight="${VAL_LOSS_WEIGHT}" \
    --reg_loss_type="${REG_LOSS_TYPE}" \
    --huber_beta="${HUBER_BETA}" \
    --use_valence_prior \
    --valence_consistency_weight="${VAL_CONSISTENCY_WEIGHT}" \
    --valence_center_reg_weight="${VAL_CENTER_REG_WEIGHT}" \
    --feature_noise_std="${FEATURE_NOISE_STD}" \
    --feature_noise_prob="${FEATURE_NOISE_PROB}" \
    --feature_noise_warmup="${FEATURE_NOISE_WARMUP}" \
    --lr="${LR}" \
    --l2="${L2}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --early_stopping_patience="${EARLY_STOP}" \
    --lr_patience="${LR_PATIENCE}" \
    --gpu="${GPU_ID}" \
    2>&1 | tee "${LOG_FILE}"

echo "=============================================="
echo "AttentionRobustV10 AV-only Formal Training End"
echo "=============================================="
