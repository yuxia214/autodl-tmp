#!/bin/bash
# ============================================================
# AttentionRobustV2 云端训练脚本
# ============================================================
# 
# 使用方法:
#   1. 修改下方路径配置为您的云服务器路径
#   2. chmod +x train_v2.sh
#   3. ./train_v2.sh
#
# 路径配置说明:
#   MERBENCH_ROOT: MERBench项目根目录
#   OUTPUT_DIR: 结果输出目录 (默认在attention_robust_v2/outputs下)
#   FEATURE_ROOT: 特征文件目录 (如需指定)
#
# ============================================================

# ==================== 路径配置 (请根据云服务器修改) ====================
# 项目根目录
MERBENCH_ROOT="/root/autodl-tmp/MERTools/MERBench"
# MERBENCH_ROOT="/home/user/MERTools/MERBench"

# 输出目录 (结果保存位置)
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v2/outputs"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/models
mkdir -p ${OUTPUT_DIR}/results

# ==================== 训练配置 ====================
# GPU设置
GPU_ID=0

# 数据集
DATASET="MER2023"

# 特征配置 (根据已有特征文件调整)
AUDIO_FEAT="chinese-hubert-large-UTT"
TEXT_FEAT="Baichuan-13B-Base-UTT"
VIDEO_FEAT="clip-vit-large-patch14-UTT"
FEAT_TYPE="utt"

# ==================== 模型超参数 ====================
HIDDEN_DIM=128
DROPOUT=0.35

# VAE参数
KL_WEIGHT=0.01
RECON_WEIGHT=0.1
CROSS_KL_WEIGHT=0.01
FUSION_TEMP=1.0
NUM_HEADS=4

# 模态Dropout
MODALITY_DROPOUT=0.15
WARMUP_EPOCHS=20

# 训练参数
LR=5e-4
L2=5e-5
EPOCHS=100
BATCH_SIZE=32
EARLY_STOP=30
LR_PATIENCE=10

# ==================== 日志配置 ====================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/logs/train_${TIMESTAMP}.log"

# ==================== 开始训练 ====================
echo "=============================================="
echo "AttentionRobustV2 训练开始"
echo "=============================================="
echo "时间戳: ${TIMESTAMP}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "=============================================="

cd ${MERBENCH_ROOT}

python -u main-robust.py \
    --model='attention_robust_v2' \
    --dataset=${DATASET} \
    --feat_type=${FEAT_TYPE} \
    --audio_feature=${AUDIO_FEAT} \
    --text_feature=${TEXT_FEAT} \
    --video_feature=${VIDEO_FEAT} \
    --save_root=${OUTPUT_DIR}/results \
    --hidden_dim=${HIDDEN_DIM} \
    --dropout=${DROPOUT} \
    --use_vae \
    --kl_weight=${KL_WEIGHT} \
    --recon_weight=${RECON_WEIGHT} \
    --cross_kl_weight=${CROSS_KL_WEIGHT} \
    --use_proxy_attention \
    --fusion_temperature=${FUSION_TEMP} \
    --num_attention_heads=${NUM_HEADS} \
    --modality_dropout=${MODALITY_DROPOUT} \
    --modality_dropout_warmup=${WARMUP_EPOCHS} \
    --lr=${LR} \
    --l2=${L2} \
    --epochs=${EPOCHS} \
    --batch_size=${BATCH_SIZE} \
    --early_stopping_patience=${EARLY_STOP} \
    --lr_patience=${LR_PATIENCE} \
    --gpu=${GPU_ID} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "=============================================="
echo "训练完成"
echo "=============================================="
echo "结果保存在: ${OUTPUT_DIR}/results"
echo "日志保存在: ${LOG_FILE}"
