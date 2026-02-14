#!/bin/bash
# AttentionRobust V9 training script

set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v9/outputs"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/models" "${OUTPUT_DIR}/results"

GPU_ID=0
DATASET="MER2023"

AUDIO_FEAT="chinese-hubert-large-UTT"
TEXT_FEAT="Baichuan-13B-Base-UTT"
VIDEO_FEAT="clip-vit-large-patch14-UTT"
FEAT_TYPE="utt"

HIDDEN_DIM=128
DROPOUT=0.35

KL_WEIGHT=0.01
RECON_WEIGHT=0.1
CROSS_KL_WEIGHT=0.01
KL_WARMUP=20

FUSION_TEMP=1.0
NUM_HEADS=4

MODALITY_DROPOUT=0.18
WARMUP_EPOCHS=15

# output losses
EMO_LOSS_WEIGHT=1.0
VAL_LOSS_WEIGHT=1.4
REG_LOSS_TYPE="smoothl1"
HUBER_BETA=0.8

# v7 stable settings
VAL_CONSISTENCY_WEIGHT=0.10
VAL_CENTER_REG_WEIGHT=0.005
FEATURE_NOISE_STD=0.02
FEATURE_NOISE_PROB=0.35
FEATURE_NOISE_WARMUP=5

# v8/v9 fusion settings
GATE_ALPHA=0.55
RELIABILITY_TEMPERATURE=0.9
MODALITY_AGREEMENT_WEIGHT=0.008
WEIGHT_CONSISTENCY_WEIGHT=0.02

# v9 new settings
QUALITY_WEIGHT=0.6
IMPUTE_LOSS_WEIGHT=0.10
CONSISTENCY_EMO_WEIGHT=0.08
CONSISTENCY_VAL_WEIGHT=0.05
CORRUPTION_MAX_RATE=0.45
CORRUPTION_WARMUP_EPOCHS=25
DOUBLE_MASK_RATIO=0.35
LATENT_NOISE_STD=0.02

LR=5e-4
L2=5e-5
EPOCHS=100
BATCH_SIZE=32
EARLY_STOP=30
LR_PATIENCE=10

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/logs/train_v9_${TIMESTAMP}.log"

cd "${MERBENCH_ROOT}"

echo "=============================================="
echo "AttentionRobustV9 Training Start"
echo "Time: ${TIMESTAMP}"
echo "Log: ${LOG_FILE}"
echo "=============================================="

python -u main-robust.py \
    --model='attention_robust_v9' \
    --dataset="${DATASET}" \
    --feat_type="${FEAT_TYPE}" \
    --audio_feature="${AUDIO_FEAT}" \
    --text_feature="${TEXT_FEAT}" \
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
    --use_gated_uncertainty \
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
    --gate_alpha="${GATE_ALPHA}" \
    --reliability_temperature="${RELIABILITY_TEMPERATURE}" \
    --modality_agreement_weight="${MODALITY_AGREEMENT_WEIGHT}" \
    --weight_consistency_weight="${WEIGHT_CONSISTENCY_WEIGHT}" \
    --quality_weight="${QUALITY_WEIGHT}" \
    --impute_loss_weight="${IMPUTE_LOSS_WEIGHT}" \
    --consistency_emo_weight="${CONSISTENCY_EMO_WEIGHT}" \
    --consistency_val_weight="${CONSISTENCY_VAL_WEIGHT}" \
    --corruption_max_rate="${CORRUPTION_MAX_RATE}" \
    --corruption_warmup_epochs="${CORRUPTION_WARMUP_EPOCHS}" \
    --double_mask_ratio="${DOUBLE_MASK_RATIO}" \
    --latent_noise_std="${LATENT_NOISE_STD}" \
    --lr="${LR}" \
    --l2="${L2}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --early_stopping_patience="${EARLY_STOP}" \
    --lr_patience="${LR_PATIENCE}" \
    --gpu="${GPU_ID}" \
    2>&1 | tee "${LOG_FILE}"

echo "=============================================="
echo "AttentionRobustV9 Training End"
echo "=============================================="
