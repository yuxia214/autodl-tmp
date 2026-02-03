#!/bin/bash
# V5 额外实验 - 充分利用GPU
# 添加更多超参数组合

PYTHON=/root/miniconda3/bin/python
cd /root/autodl-tmp/MERTools-master/MERBench

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "V5 额外实验 - 开始时间: $(date)"
echo "=========================================="

# ==================== 实验7: V5 + 更低dropout ====================
echo "启动实验7: V5 + dropout=0.25..."
screen -dmS v5_exp7 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.25 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp7_v5_dropout025_${TIMESTAMP}.log
exec bash
"

# ==================== 实验8: V5 + 更大KL权重 ====================
echo "启动实验8: V5 + kl_weight=0.05..."
screen -dmS v5_exp8 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.05 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp8_v5_kl005_${TIMESTAMP}.log
exec bash
"

# ==================== 实验9: V5 + 更长warmup ====================
echo "启动实验9: V5 + warmup=40..."
screen -dmS v5_exp9 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=40 \
    --use_dynamic_kl --kl_warmup_epochs=30 \
    --no_mixup \
    --lr=5e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp9_v5_warmup40_${TIMESTAMP}.log
exec bash
"

# ==================== 实验10: V5 + Mixup + hidden256 ====================
echo "启动实验10: V5 + Mixup + hidden256..."
screen -dmS v5_exp10 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=256 --dropout=0.4 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --use_mixup --mixup_alpha=0.3 \
    --lr=3e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp10_v5_mixup_h256_${TIMESTAMP}.log
exec bash
"

# ==================== 实验11: V5 + 更小学习率 ====================
echo "启动实验11: V5 + lr=1e-4..."
screen -dmS v5_exp11 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=1e-4 --l2=5e-5 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp11_v5_lr1e4_${TIMESTAMP}.log
exec bash
"

# ==================== 实验12: V5 + 更强L2正则 ====================
echo "启动实验12: V5 + l2=1e-4..."
screen -dmS v5_exp12 bash -c "
cd /root/autodl-tmp/MERTools-master/MERBench
$PYTHON -u main-robust.py \
    --model='attention_robust_v5' \
    --dataset=MER2023 --feat_type=utt \
    --audio_feature=chinese-hubert-large-UTT \
    --text_feature=Baichuan-13B-Base-UTT \
    --video_feature=clip-vit-large-patch14-UTT \
    --hidden_dim=128 --dropout=0.35 \
    --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --use_dynamic_kl --kl_warmup_epochs=20 \
    --no_mixup \
    --lr=5e-4 --l2=1e-4 --epochs=100 \
    --early_stopping_patience=30 --batch_size=32 --gpu=0 \
    2>&1 | tee logs/v5_experiments/exp12_v5_l2_1e4_${TIMESTAMP}.log
exec bash
"

sleep 3

echo ""
echo "=========================================="
echo "额外6个实验已启动! 现在共12个实验并行运行"
echo "=========================================="
screen -list | grep v5_exp
echo ""
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
