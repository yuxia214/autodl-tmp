#!/bin/bash

# === 配置区域 ===
GPU_ID=0
# 用户指定：只跑一次，跑通即可
RUN_TIMES=1
# 用户指定：加大批次到 512 以提速
BATCH_SIZE=512
# 保持安全学习率，防止 NaN
LR=0.0001

echo "========================================================"
echo "🚀 极速模式: MulT (Sequence-level) 测试"
echo "⚙️  配置: Batch_Size=$BATCH_SIZE | Runs=$RUN_TIMES"
echo "========================================================"

for i in $(seq 1 $RUN_TIMES)
do
   echo ">>> MulT 正在启动 (Run $i/$RUN_TIMES)..."
   
   python -u main-release.py \
       --model='mult' \
       --feat_type='frm_align' \
       --dataset='MER2023' \
       --audio_feature='chinese-hubert-large-FRA' \
       --text_feature='Baichuan-13B-Base-FRA' \
       --video_feature='clip-vit-large-patch14-FRA' \
       --batch_size=$BATCH_SIZE \
       --lr=$LR \
       --gpu=$GPU_ID
done

echo "===== 测试结束！请检查显存是否溢出 ====="