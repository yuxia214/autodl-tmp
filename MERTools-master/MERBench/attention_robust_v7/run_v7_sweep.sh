#!/bin/bash
# V7 sweep runner: val_loss_weight / valence_consistency_weight / feature_noise_std
# Auto sequential runs, with summary report.

set -euo pipefail

MERBENCH_ROOT="/root/autodl-tmp/MERTools-master/MERBench"
OUTPUT_DIR="${MERBENCH_ROOT}/attention_robust_v7/outputs"
SWEEP_LOG_DIR="${OUTPUT_DIR}/logs/sweeps"
SWEEP_RESULT_DIR="${OUTPUT_DIR}/sweep_results"
SUMMARY_CSV="${OUTPUT_DIR}/sweep_summary.csv"

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
REG_LOSS_TYPE="smoothl1"
HUBER_BETA=0.8
VAL_CENTER_REG_WEIGHT=0.005
FEATURE_NOISE_PROB=0.35
FEATURE_NOISE_WARMUP=5

LR=5e-4
L2=5e-5
EPOCHS=100
BATCH_SIZE=32
EARLY_STOP=30
LR_PATIENCE=10

# sweep configs (3 runs)
# format: "<name> <val_loss_weight> <valence_consistency_weight> <feature_noise_std>"
CONFIGS=(
  "baseline 1.3 0.12 0.03"
  "mse_focus 1.4 0.10 0.02"
  "noise_focus 1.2 0.14 0.04"
)

cd "${MERBENCH_ROOT}"

if [ ! -f "${SUMMARY_CSV}" ]; then
  echo "run_tag,val_loss_weight,valence_consistency_weight,feature_noise_std,test1_f1,test1_acc,test1_val,test1_combined,test1_gap_to_0.7005,test2_f1,test2_acc,test2_val,test2_combined,test2_gap_to_0.6846,test3_f1,test3_acc,test3_val,test3_gap_to_0.8911,log_file" > "${SUMMARY_CSV}"
fi

echo "[SWEEP] waiting current v7 run to finish (if any)..."
while pgrep -af "python -u main-robust.py.*attention_robust_v7" >/dev/null; do
  echo "[SWEEP] detected running v7 training, sleep 120s..."
  sleep 120
done
echo "[SWEEP] start sequential sweep runs (3 configs)"

for cfg in "${CONFIGS[@]}"; do
  read -r CFG_NAME VLW VCW NSD <<< "${cfg}"
  TAG="v7sw_${CFG_NAME}_vlw${VLW}_vcw${VCW}_nsd${NSD}_$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${SWEEP_LOG_DIR}/${TAG}.log"
  SAVE_ROOT="${SWEEP_RESULT_DIR}/${TAG}"

  echo "===================================================="
  echo "[SWEEP] ${TAG}"
  echo "  val_loss_weight=${VLW}"
  echo "  valence_consistency_weight=${VCW}"
  echo "  feature_noise_std=${NSD}"
  echo "  log=${LOG_FILE}"
  echo "===================================================="

  python -u main-robust.py \
    --model='attention_robust_v7' \
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
    --fusion_temperature="${FUSION_TEMP}" \
    --num_attention_heads="${NUM_HEADS}" \
    --modality_dropout="${MODALITY_DROPOUT}" \
    --modality_dropout_warmup="${WARMUP_EPOCHS}" \
    --emo_loss_weight="${EMO_LOSS_WEIGHT}" \
    --val_loss_weight="${VLW}" \
    --reg_loss_type="${REG_LOSS_TYPE}" \
    --huber_beta="${HUBER_BETA}" \
    --use_valence_prior \
    --valence_consistency_weight="${VCW}" \
    --valence_center_reg_weight="${VAL_CENTER_REG_WEIGHT}" \
    --feature_noise_std="${NSD}" \
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

  python - <<'PY' "${LOG_FILE}" "${SUMMARY_CSV}" "${TAG}" "${VLW}" "${VCW}" "${NSD}"
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
tag, vlw, vcw, nsd = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
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
    tag, vlw, vcw, nsd,
    f"{f1_1:.4f}", f"{acc_1:.4f}", f"{val_1:.4f}", f"{c1:.4f}", f"{gap1:.4f}",
    f"{f1_2:.4f}", f"{acc_2:.4f}", f"{val_2:.4f}", f"{c2:.4f}", f"{gap2:.4f}",
    f"{f1_3:.4f}", f"{acc_3:.4f}", f"{val_3:.4f}", f"{gap3:.4f}",
    str(log_path)
]
with summary_csv.open('a', encoding='utf-8') as f:
    f.write(','.join(row) + '\n')

print('[SWEEP] summary appended:', ','.join(row))
PY
done

echo "[SWEEP] all runs finished"
echo "[SWEEP] summary: ${SUMMARY_CSV}"
