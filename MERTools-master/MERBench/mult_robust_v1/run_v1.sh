#!/bin/bash
cd /root/autodl-tmp/MERTools-master/MERBench
python -u mult_robust_v1/run_mult_v1.py --model='mult' --feat_type='frm_align' --dataset='MER2023' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --gpu=0 --batch_size=512 --num_workers=12 | tee mult_v1_train.log
