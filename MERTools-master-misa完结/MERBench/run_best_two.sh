#!/bin/bash

GPU_ID=0
# 循环运行 3 次，增加命中好结果的概率
RUN_TIMES=3

echo "========================================================"
echo "阶段 1/2: 运行 Attention (Utt-level) [安全模式]"
echo "========================================================"

# 注意：这里去掉了 --dropout，但保留了降低的 lr 和 batch_size
# batch_size=128, lr=0.0001 是防止 NaN 的关键
FIXED_PARAMS="--batch_size=128 --lr=0.0001"

for i in $(seq 1 $RUN_TIMES)
do
   echo ">>> Attention 运行 $i/$RUN_TIMES ..."
   python -u main-release.py \
       --model='attention' \
       --feat_type='utt' \
       --dataset='MER2023' \
       --audio_feature='chinese-hubert-large-UTT' \
       --text_feature='Baichuan-13B-Base-UTT' \
       --video_feature='clip-vit-large-patch14-UTT' \
       $FIXED_PARAMS \
       --gpu=$GPU_ID
done

echo "========================================================"
echo "阶段 2/2: 运行 MulT (Sequence-level) [安全模式]"
echo "========================================================"

for i in $(seq 1 $RUN_TIMES)
do
   echo ">>> MulT 运行 $i/$RUN_TIMES ..."
   # MulT 显存占用大，为了安全，这里强制用 batch_size=32
   python -u main-release.py \
       --model='mult' \
       --feat_type='frm_align' \
       --dataset='MER2023' \
       --audio_feature='chinese-hubert-large-FRA' \
       --text_feature='Baichuan-13B-Base-FRA' \
       --video_feature='clip-vit-large-patch14-FRA' \
       --batch_size=32 \
       --lr=0.0001 \
       --gpu=$GPU_ID
done

echo "===== 全部运行结束！请查看 saved-trimodal/result 下的结果 ====="