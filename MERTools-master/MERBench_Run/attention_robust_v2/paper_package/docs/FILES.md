# AttentionRobustV2 文件清单

## attention_robust_v2/ 独立文件夹

此文件夹包含完整的V2模型，可独立运行或通过setup_v2.py安装到toolkit。

```
attention_robust_v2/
├── __init__.py                    # 模块入口
├── attention_robust_v2.py         # 主模型 (支持动态导入)
├── setup_v2.py                    # 自动安装脚本
├── train_v2.sh                    # 训练脚本 (云端路径)
├── run_ablation.sh                # 消融实验脚本
├── README.md                      # 使用说明
├── FILES.md                       # 本文件
├── modules/
│   ├── __init__.py                # 子模块入口
│   ├── encoder.py                 # 基础编码器
│   └── variational_encoder.py     # 变分编码器组件
└── outputs/
    └── README.md                  # 输出目录说明
```

## toolkit/ 已安装的文件

这些文件已被复制到toolkit相应位置。

```
toolkit/
├── models/
│   ├── __init__.py                # 已添加 attention_robust_v2 注册
│   ├── attention_robust_v2.py     # 主模型 (直接导入)
│   └── modules/
│       ├── __init__.py            # 已添加 variational_encoder 导出
│       ├── encoder.py             # 原有
│       └── variational_encoder.py # 新增变分编码器
├── data/
│   └── __init__.py                # 已添加 attention_robust_v2 数据映射
└── model-tune.yaml                # 已添加 attention_robust_v2 超参数
```

## main-robust.py 修改

已添加以下参数支持:
- `--use_vae` / `--no_vae`
- `--kl_weight`
- `--recon_weight`
- `--cross_kl_weight`
- `--use_proxy_attention` / `--no_proxy_attention`
- `--fusion_temperature`
- `--num_attention_heads`

---

## 云端使用快速开始

```bash
# 1. 修改训练脚本中的路径
vim attention_robust_v2/train_v2.sh
# 修改 MERBENCH_ROOT="/your/server/path/MERBench"

# 2. 运行训练
cd /path/to/MERBench
chmod +x attention_robust_v2/train_v2.sh
./attention_robust_v2/train_v2.sh
```

## 或者使用命令行

```bash
cd /path/to/MERBench
python -u main-robust.py \
    --model='attention_robust_v2' \
    --dataset='MER2023' \
    --feat_type='utt' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --save_root='./attention_robust_v2/outputs/results' \
    --use_vae \
    --use_proxy_attention \
    --gpu=0
```
