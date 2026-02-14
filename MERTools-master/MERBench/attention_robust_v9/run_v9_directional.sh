#!/bin/bash
# V9 directional experiments (4 runs)

set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v9/outputs"
SWEEP_LOG_DIR="${OUTPUT_DIR}/logs/sweeps"
SWEEP_RESULT_DIR="${OUTPUT_DIR}/sweep_results"
SUMMARY_CSV="${OUTPUT_DIR}/directional_summary.csv"

mkdir -p "${SWEEP_LOG_DIR}" "${SWEEP_RESULT_DIR}"

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
EMO_LOSS_WEIGHT=1.0
VAL_LOSS_WEIGHT=1.4
REG_LOSS_TYPE="smoothl1"
HUBER_BETA=0.8
VAL_CONSISTENCY_WEIGHT=0.10
VAL_CENTER_REG_WEIGHT=0.005
FEATURE_NOISE_STD=0.02
FEATURE_NOISE_PROB=0.35
FEATURE_NOISE_WARMUP=5
GATE_ALPHA=0.55
RELIABILITY_TEMPERATURE=0.9
MODALITY_AGREEMENT_WEIGHT=0.008
WEIGHT_CONSISTENCY_WEIGHT=0.02

LR=5e-4
L2=5e-5
EPOCHS=100
BATCH_SIZE=32
EARLY_STOP=30
LR_PATIENCE=10

# format:
# name quality impute emo_cons val_cons corruption_max double_mask latent_noise
CONFIGS=(
  "base 0.60 0.10 0.08 0.05 0.45 0.35 0.02"
  "imp_strong 0.60 0.14 0.08 0.05 0.45 0.35 0.02"
  "cons_strong 0.60 0.10 0.12 0.08 0.45 0.35 0.02"
  "high_corrupt 0.70 0.12 0.10 0.06 0.55 0.45 0.03"
)

cd "${MERBENCH_ROOT}"

if [ ! -f "${SUMMARY_CSV}" ]; then
  echo "run_tag,quality_weight,impute_loss_weight,consistency_emo_weight,consistency_val_weight,corruption_max_rate,double_mask_ratio,latent_noise_std,test1_f1,test1_acc,test1_val,test1_combined,test1_gap_to_0.7005,test2_f1,test2_acc,test2_val,test2_combined,test2_gap_to_0.6846,test3_f1,test3_acc,test3_val,test3_gap_to_0.8911,log_file" > "${SUMMARY_CSV}"
fi

echo "[V9] waiting running v9 jobs to finish..."
while pgrep -af "python -u main-robust.py.*attention_robust_v9" >/dev/null; do
  echo "[V9] detected running v9 training, sleep 120s..."
  sleep 120
done

echo "[V9] start directional runs"
for cfg in "${CONFIGS[@]}"; do
  read -r NAME QW IW CE CV CR DM LN <<< "${cfg}"

  TAG="v9_${NAME}_qw${QW}_iw${IW}_ce${CE}_cv${CV}_cr${CR}_dm${DM}_ln${LN}_$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${SWEEP_LOG_DIR}/${TAG}.log"
  SAVE_ROOT="${SWEEP_RESULT_DIR}/${TAG}"

  echo "===================================================="
  echo "[V9] ${TAG}"
  echo "  quality_weight=${QW}"
  echo "  impute_loss_weight=${IW}"
  echo "  consistency_emo_weight=${CE}"
  echo "  consistency_val_weight=${CV}"
  echo "  corruption_max_rate=${CR}"
  echo "  double_mask_ratio=${DM}"
  echo "  latent_noise_std=${LN}"
  echo "===================================================="

  python -u main-robust.py \
    --model='attention_robust_v9' \
    --dataset="${DATASET}" \
    --feat_type="${FEAT_TYPE}" \
    --audio_feature="${AUDIO_FEAT}" \
    --text_feature="${TEXT_FEAT}" \
    --video_feature="${VIDEO_FEAT}" \
    --save_root="${SAVE_ROOT}" \
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
    --quality_weight="${QW}" \
    --impute_loss_weight="${IW}" \
    --consistency_emo_weight="${CE}" \
    --consistency_val_weight="${CV}" \
    --corruption_max_rate="${CR}" \
    --corruption_warmup_epochs=25 \
    --double_mask_ratio="${DM}" \
    --latent_noise_std="${LN}" \
    --lr="${LR}" \
    --l2="${L2}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --early_stopping_patience="${EARLY_STOP}" \
    --lr_patience="${LR_PATIENCE}" \
    --gpu="${GPU_ID}" \
    2>&1 | tee "${LOG_FILE}"

  python - <<'PY' "${LOG_FILE}" "${SUMMARY_CSV}" "${TAG}" "${QW}" "${IW}" "${CE}" "${CV}" "${CR}" "${DM}" "${LN}"
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
tag, qw, iw, ce, cv, cr, dm, ln = sys.argv[3:11]
text = log_path.read_text(encoding='utf-8', errors='ignore')

pat = {
    'test1': re.compile(r'test1_features:.*?f1:([0-9]+\.[0-9]+)_acc:([0-9]+\.[0-9]+)_val:([0-9]+\.[0-9]+)'),
    'test2': re.compile(r'test2_features:.*?f1:([0-9]+\.[0-9]+)_acc:([0-9]+\.[0-9]+)_val:([0-9]+\.[0-9]+)'),
    'test3': re.compile(r'test3_features:.*?f1:([0-9]+\.[0-9]+)_acc:([0-9]+\.[0-9]+)_val:([0-9]+\.[0-9]+)'),
}

vals = {}
for k, p in pat.items():
    m = None
    for mm in p.finditer(text):
        m = mm
    if m:
        vals[k] = tuple(float(x) for x in m.groups())
    else:
        vals[k] = (float('nan'), float('nan'), float('nan'))

f1_1, acc_1, val_1 = vals['test1']
f1_2, acc_2, val_2 = vals['test2']
f1_3, acc_3, val_3 = vals['test3']

c1 = f1_1 - 0.25 * val_1 if f1_1 == f1_1 and val_1 == val_1 else float('nan')
c2 = f1_2 - 0.25 * val_2 if f1_2 == f1_2 and val_2 == val_2 else float('nan')

gap1 = c1 - 0.7005 if c1 == c1 else float('nan')
gap2 = c2 - 0.6846 if c2 == c2 else float('nan')
gap3 = f1_3 - 0.8911 if f1_3 == f1_3 else float('nan')

row = [
    tag, qw, iw, ce, cv, cr, dm, ln,
    f"{f1_1:.4f}", f"{acc_1:.4f}", f"{val_1:.4f}", f"{c1:.4f}", f"{gap1:.4f}",
    f"{f1_2:.4f}", f"{acc_2:.4f}", f"{val_2:.4f}", f"{c2:.4f}", f"{gap2:.4f}",
    f"{f1_3:.4f}", f"{acc_3:.4f}", f"{val_3:.4f}", f"{gap3:.4f}",
    str(log_path)
]

with summary_csv.open('a', encoding='utf-8') as f:
    f.write(','.join(row) + '\n')

print('[V9] summary appended:', ','.join(row))
PY

done

echo "[V9] all directional runs finished"
echo "[V9] summary: ${SUMMARY_CSV}"
