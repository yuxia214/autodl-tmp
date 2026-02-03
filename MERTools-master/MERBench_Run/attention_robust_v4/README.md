# AttentionRobustV2 - 基于VAE的鲁棒多模态情感识别

## 概述

本模块实现了一个改进的多模态情感识别模型，核心创新点来自于P-RMF论文中的缺失模态处理技术。

**核心改进**：从确定性特征学习转向概率分布学习，通过变分编码器(VAE)实现对模态不确定性的建模。

### 主要技术

1. **变分编码 (VAE)**
   - 将各模态特征编码为高斯分布参数 (μ, σ)
   - μ 表示语义信息，σ 表示不确定性/置信度
   - 使用重参数化技巧保证梯度可传播

2. **不确定性加权融合**
   - 融合权重 w_i ∝ 1/σ_i
   - 高不确定性模态获得较低权重
   - 自动适应模态缺失场景

3. **代理模态 (Proxy Modality)**
   - 生成稳定的聚合特征 proxy = Σ(w_i × μ_i)
   - 使用proxy作为Query进行跨模态注意力
   - 引导各模态信息聚合

4. **多任务损失**
   - 分类损失 (CE) + KL散度 + 重建损失 + 跨模态KL

---

## 文件结构

```
attention_robust_v2/
├── __init__.py                 # 模块入口
├── attention_robust_v2.py      # 主模型文件
├── setup_v2.py                 # 自动安装脚本
├── train_v2.sh                 # 训练脚本 (云端)
├── run_ablation.sh             # 消融实验脚本
├── README.md                   # 本文档
├── modules/
│   ├── __init__.py             # 子模块入口
│   ├── encoder.py              # 基础编码器 (MLPEncoder, LSTMEncoder)
│   └── variational_encoder.py  # 变分编码器组件
└── outputs/                    # 训练输出 (自动生成)
    ├── logs/                   # 训练日志
    ├── models/                 # 保存的模型
    └── results/                # 结果文件
```

---

## 安装方法

### 方法1: 自动安装 (推荐)

```bash
cd /path/to/MERBench
python attention_robust_v2/setup_v2.py
```

脚本将自动:
1. 复制模型文件到 `toolkit/models/`
2. 复制模块到 `toolkit/models/modules/`
3. 更新 `__init__.py` 注册新模型

### 方法2: 手动安装

1. 复制 `modules/variational_encoder.py` 到 `toolkit/models/modules/`
2. 复制 `attention_robust_v2.py` 到 `toolkit/models/`
3. 修改 `toolkit/models/__init__.py`:
   ```python
   from .attention_robust_v2 import AttentionRobustV2
   # 在 MODEL_MAP 中添加:
   'attention_robust_v2': AttentionRobustV2,
   ```
4. 修改 `toolkit/data/__init__.py`:
   ```python
   # 在数据映射中添加:
   'attention_robust_v2': Data_Feat,
   ```

---

## 使用方法

### 基本训练

```bash
python -u main-robust.py \
    --model='attention_robust_v2' \
    --dataset='MER2023' \
    --feat_type='utt' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --use_vae \
    --use_proxy_attention \
    --gpu=0
```

### 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_vae` | True | 启用VAE编码器 |
| `--kl_weight` | 0.01 | KL散度损失权重 |
| `--recon_weight` | 0.1 | 重建损失权重 |
| `--cross_kl_weight` | 0.01 | 跨模态KL权重 |
| `--use_proxy_attention` | True | 启用代理模态注意力 |
| `--fusion_temperature` | 1.0 | 融合温度参数 |
| `--num_attention_heads` | 4 | 注意力头数 |
| `--modality_dropout` | 0.15 | 模态dropout率 |
| `--modality_dropout_warmup` | 20 | dropout预热epochs |
| `--hidden_dim` | 128 | 隐藏层维度 |
| `--dropout` | 0.35 | Dropout率 |

### 使用脚本训练

```bash
# 单次训练
chmod +x train_v2.sh
./train_v2.sh

# 消融实验
chmod +x run_ablation.sh
./run_ablation.sh
```

---

## 云端路径配置

编辑 `train_v2.sh` 中的路径:

```bash
# 根据您的服务器修改
MERBENCH_ROOT="/root/autodl-tmp/MERTools/MERBench"
# 或
MERBENCH_ROOT="/home/user/MERTools/MERBench"
```

---

## 模型架构

```
输入: audio [B, audio_dim], text [B, text_dim], video [B, video_dim]
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 1. 变分编码层 (Variational Encoders)                 │
│    audio → (z_a, μ_a, σ_a)                          │
│    text  → (z_t, μ_t, σ_t)                          │
│    video → (z_v, μ_v, σ_v)                          │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 2. 模态Dropout (训练时)                              │
│    随机置零某些模态的z值                              │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 3. 不确定性加权融合                                   │
│    w_i = softmax(1/σ_i / temperature)               │
│    proxy = Σ(w_i × μ_i)                             │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 4. 代理模态跨模态注意力                              │
│    Query: proxy                                      │
│    Key/Value: [μ_a, μ_t, μ_v] (加权)                │
│    Output: fused = Attention(Q, K, V)               │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 5. 输出层                                            │
│    emos_out = FC(fused)  # 情感分类                  │
│    vals_out = FC(fused)  # 情感回归                  │
└─────────────────────────────────────────────────────┘
  │
  ▼
输出: features, emos_out, vals_out, interloss
      (interloss = KL + recon + cross_KL)
```

---

## 训练逻辑说明

训练过程中的损失计算:

```
total_loss = classification_loss + interloss
           = CE(emos_out, emos) + (λ1×L_KL + λ2×L_recon + λ3×L_cross_KL)
```

其中:
- `L_KL`: KL散度，约束潜在空间 → 正则化
- `L_recon`: 重建损失，保持语义完整性
- `L_cross_KL`: 跨模态KL，鼓励模态间一致性

---

## 预期效果

| 模型 | test1 (完整) | test2 (缺失) |
|------|-------------|--------------|
| V1 baseline | ~0.85 | ~0.76 |
| **V2 (本模型)** | ~0.85 | **~0.78-0.79** |

主要提升在test2 (缺失模态测试)上。

---

## 调参建议

### 快速开始 (默认配置)
直接使用默认参数即可获得较好结果。

### 进阶调优
1. **KL权重** (`--kl_weight`): 0.005-0.02
   - 太大: 模型退化为简单高斯
   - 太小: 失去正则化效果

2. **重建权重** (`--recon_weight`): 0.05-0.2
   - 较大时语义保持更好

3. **模态dropout** (`--modality_dropout`): 0.1-0.25
   - 太大会影响正常训练

4. **融合温度** (`--fusion_temperature`): 0.5-2.0
   - <1: 加强高置信度模态
   - >1: 平滑权重分布

---

## 常见问题

**Q: 出现 ImportError？**
A: 运行 `python setup_v2.py` 自动安装，或检查模型是否正确注册。

**Q: 训练loss不下降？**
A: 尝试减小 `--kl_weight` 或 `--recon_weight`。

**Q: test2效果没有提升？**
A: 确保使用了正确的参数组合，特别是 `--use_vae` 和 `--modality_dropout`。

---

## 参考

- P-RMF: Proxy-Driven Robust Multimodal Sentiment Analysis (ACL 2025)
- VAE: Auto-Encoding Variational Bayes (Kingma & Welling, 2014)
