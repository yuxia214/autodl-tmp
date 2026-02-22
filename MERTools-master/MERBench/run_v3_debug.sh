#!/bin/bash
cd /root/autodl-tmp/MERTools-master/MERBench
python -u mult_robust_v3/run_mult_v3.py \
    --model='mult' \
    --feat_type='frm_align' \
    --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-FRA' \
    --text_feature='Baichuan-13B-Base-FRA' \
    --video_feature='clip-vit-large-patch14-FRA' \
    --gpu=0 \
    --batch_size=128 \
    --num_workers=4 \
    --l2=0.0001 \
    --lr=0.0005 \
    --epochs=1 \
    | tee mult_v3_debug.log
