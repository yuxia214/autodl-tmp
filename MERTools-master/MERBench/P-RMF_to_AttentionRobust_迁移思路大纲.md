# P-RMF 核心技术迁移至 AttentionRobust 模型 - 深度修改方案 V2

## 📌 目标
将 P-RMF (Proxy-Driven Robust Multimodal Framework) 中处理缺失模态的核心技术迁移到现有的 `attention_robust.py` 模型中，**提升 test2（模态缺失测试）的准确率**。

---

## 🧠 核心理论转变

### 从"确定性特征学习"到"概率分布学习"

| 维度 | 现有方法 | P-RMF方法 | 优势 |
|------|----------|-----------|------|
| **特征表示** | 固定向量 $h \in \mathbb{R}^d$ | 高斯分布 $\mathcal{N}(\mu, \sigma^2)$ | 模态缺失时方差自动增大 |
| **缺失处理** | 置零/Dropout → 特征跳变 | 方差感知 → 权重自动降低 | 平滑过渡，无剧烈跳变 |
| **融合策略** | 固定Attention/拼接 | 反向方差加权 $w = 1/\sigma$ | 自动识别并抑制不可靠模态 |
| **学习目标** | 仅分类损失 | 分类+重建+KL散度 | 特征更完整，分布更合理 |

---

## 📖 P-RMF 核心技术分析

### 1. P-RMF 整体架构
```
输入 → 模态编码器 → VAE生成代理模态 → 跨模态注意力融合 → 预测
              ↓
    [完整输入用于重建监督]
```

### 2. 关键创新点

#### 2.1 代理模态生成器 (Proxy Modality Generator)
**文件位置**: `P-RMF-main/models/generate_proxy_modality.py`

**核心思想**:
- 使用 **VAE (变分自编码器)** 为每个模态学习潜在表示
- 通过 **不确定性加权** 融合三个模态的潜在表示，生成一个"代理模态"
- 代理模态能够在某个模态缺失时，从其他模态补充信息

**关键公式**:
```python
# 不确定性加权 - 标准差越小（越确定），权重越高
weight_m = exp(1/std) / sum(exp(1/std))
proxy_m = sum(weight_m * mu)  # 加权融合各模态的均值
```

#### 2.2 跨模态编码器 (CrossModal Encoder)
**文件位置**: `P-RMF-main/models/basic_layers.py`

**核心思想**:
- 使用代理模态作为 Query，各原始模态作为 Key/Value
- 根据不确定性权重动态调整各模态的贡献
```python
output = (cma_t(proxy_m, text) * weight_t +
          cma_a(proxy_m, audio) * weight_a +
          cma_v(proxy_m, video) * weight_v +
          proxy_m)  # 残差连接
```

#### 2.3 重建损失 (Reconstruction Loss)
**文件位置**: `P-RMF-main/core/losses.py`

**核心思想**:
- 训练时使用**完整数据**和**缺失数据**同时输入
- 使用重建损失强制模型从缺失数据重建出完整数据的表示
```python
l_rec = MSE(rec_feats, complete_feats)  # 重建损失
l_kl = kl_divergence(...)  # KL散度约束VAE
```

#### 2.4 训练时动态模态缺失
**文件位置**: `P-RMF-main/core/dataset.py`

**核心思想**:
- 训练时为每个样本随机生成缺失率 (0~1的均匀分布)
- 50%的样本保持某模态完整，50%的样本按缺失率mask
- 每个epoch重新生成缺失模式

---

## 🔄 迁移策略 (修订版)

### 方案对比

| 方案 | 复杂度 | 预期效果 | 推荐度 | 说明 |
|------|--------|----------|--------|------|
| A: 轻量级不确定性加权 | ⭐ | ⭐⭐ | ⭐⭐ | 仅加权，无VAE |
| B: 简化版VAE代理模态 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 原方案 |
| **C: 完整VAE+重建+代理模态** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **最终推荐** |

### 最终推荐方案：方案C - 完整概率化改造

**选择理由** (结合AI分析):
1. **从点到分布的转变是核心** - 这是提升缺失模态鲁棒性的物理基础
2. **重建损失是关键约束** - 迫使编码器即使在输入残缺时也能"脑补"完整语义
3. **interloss接口已预留** - 可直接利用，无需修改训练框架
4. **投入产出比高** - 虽然复杂度增加，但预期效果显著

---

## 📋 详细迁移计划 (四阶段深度改造)

### 第一阶段：不确定性估计模块

#### 1.1 添加模态不确定性估计器
```python
class ModalityUncertaintyEstimator(nn.Module):
    """估计每个模态表示的不确定性"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mu_layer = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar
```

#### 1.2 实现不确定性加权融合
```python
def uncertainty_weighted_fusion(self, audio_mu, audio_std, text_mu, text_std, video_mu, video_std):
    """
    根据不确定性（标准差的倒数）加权融合模态
    不确定性越低（std越小），权重越高
    """
    # 计算权重 - P-RMF的核心公式
    weights = torch.stack([
        torch.exp(1.0 / (audio_std.mean(dim=-1, keepdim=True) + 1e-6)),
        torch.exp(1.0 / (text_std.mean(dim=-1, keepdim=True) + 1e-6)),
        torch.exp(1.0 / (video_std.mean(dim=-1, keepdim=True) + 1e-6))
    ], dim=1)
    weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化
    
    # 加权融合
    mu_stack = torch.stack([audio_mu, text_mu, video_mu], dim=1)
    proxy = (weights.unsqueeze(-1) * mu_stack).sum(dim=1)
    
    return proxy, weights
```

### 第二阶段：代理模态引导的跨模态注意力

#### 2.1 添加代理模态跨模态注意力层
```python
class ProxyCrossModalAttention(nn.Module):
    """使用代理模态作为Query，原始模态作为Key/Value"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, proxy, modality, weight):
        """
        proxy: 代理模态 [B, H]
        modality: 原始模态 [B, H]
        weight: 该模态的不确定性权重 [B, 1]
        """
        # 扩展维度用于attention
        proxy_exp = proxy.unsqueeze(1)  # [B, 1, H]
        modality_exp = modality.unsqueeze(1)  # [B, 1, H]
        
        # Cross attention: proxy attend to modality
        attn_out, _ = self.attention(proxy_exp, modality_exp, modality_exp)
        attn_out = attn_out.squeeze(1)  # [B, H]
        
        # 加权残差连接
        out = self.norm(proxy + weight * attn_out)
        return out
```

### 第三阶段：KL散度正则化

#### 3.1 添加KL散度损失
```python
def compute_kl_loss(self, audio_mu, audio_logvar, text_mu, text_logvar, video_mu, video_logvar):
    """
    计算模态间KL散度，鼓励各模态学习相似的潜在分布
    """
    def kl_div(mu1, logvar1, mu2, logvar2):
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        kl = 0.5 * (logvar2 - logvar1 + var1/var2 + (mu1-mu2)**2/var2 - 1)
        return kl.mean()
    
    kl_at = kl_div(audio_mu, audio_logvar, text_mu, text_logvar)
    kl_av = kl_div(audio_mu, audio_logvar, video_mu, video_logvar)
    kl_tv = kl_div(text_mu, text_logvar, video_mu, video_logvar)
    
    return (kl_at + kl_av + kl_tv) / 3
```

### 第四阶段：改进的模态Dropout策略

#### 4.1 基于不确定性的智能模态Dropout
```python
def adaptive_modality_dropout(self, audio_hidden, text_hidden, video_hidden, 
                               audio_std, text_std, video_std):
    """
    智能模态dropout：优先丢弃不确定性高的模态
    而不是完全随机丢弃
    """
    if not self.training:
        return audio_hidden, text_hidden, video_hidden
    
    batch_size = audio_hidden.size(0)
    
    # 计算各模态的不确定性分数
    uncertainties = torch.stack([
        audio_std.mean(dim=-1),
        text_std.mean(dim=-1),
        video_std.mean(dim=-1)
    ], dim=1)  # [B, 3]
    
    # 不确定性越高，被dropout的概率越大
    dropout_probs = F.softmax(uncertainties * self.uncertainty_dropout_temp, dim=1)
    
    # 采样决定是否dropout
    for i in range(batch_size):
        if torch.rand(1).item() < self.modality_dropout:
            # 根据不确定性概率选择要dropout的模态
            drop_idx = torch.multinomial(dropout_probs[i], 1).item()
            if drop_idx == 0:
                audio_hidden[i] = audio_hidden[i] * 0
            elif drop_idx == 1:
                text_hidden[i] = text_hidden[i] * 0
            else:
                video_hidden[i] = video_hidden[i] * 0
    
    return audio_hidden, text_hidden, video_hidden
```

---

## 📁 需要修改的文件

### 主要修改

| 文件 | 修改内容 |
|------|----------|
| `toolkit/models/attention_robust.py` | 添加不确定性估计、代理模态生成、跨模态注意力 |
| `main-robust.py` | 添加KL损失权重参数，修改损失函数 |
| `toolkit/model-tune.yaml` | 添加新超参数配置 |

### 可选修改

| 文件 | 修改内容 |
|------|----------|
| `toolkit/dataloader/` | 若需要完整+缺失数据同时输入 |

---

## 🏗️ 新模型架构设计

```
AttentionRobustV2 模型架构
===========================

输入: audio [B, D_a], text [B, D_t], video [B, D_v]
                    ↓
        ┌──────────┼──────────┐
        ↓          ↓          ↓
   AudioEncoder  TextEncoder  VideoEncoder
        ↓          ↓          ↓
     [B, H]      [B, H]      [B, H]
        ↓          ↓          ↓
        ├──────────┼──────────┤
        ↓          ↓          ↓
   UncertaintyEstimator (为每个模态估计 μ 和 σ)
        ↓          ↓          ↓
   (μ_a, σ_a)  (μ_t, σ_t)  (μ_v, σ_v)
        ↓          ↓          ↓
        └──────────┴──────────┘
                   ↓
    Uncertainty-Weighted Fusion (生成代理模态)
                   ↓
              proxy [B, H]
                   ↓
    ┌──────────────┼──────────────┐
    ↓              ↓              ↓
CrossAttn(proxy,a) CrossAttn(proxy,t) CrossAttn(proxy,v)
    ↓              ↓              ↓
    └──────────────┴──────────────┘
                   ↓
           Weighted Sum (使用不确定性权重)
                   ↓
           fused_feat [B, H]
                   ↓
              FC Layers
                   ↓
           emos_out, vals_out

额外输出: kl_loss (用于训练正则化)
```

---

## ⚙️ 超参数配置

```yaml
attention_robust_v2:
  # 基础参数
  hidden_dim: 128
  dropout: 0.35
  
  # 不确定性估计参数
  use_uncertainty: true
  uncertainty_hidden_dim: 128
  
  # 代理模态参数
  use_proxy_modality: true
  proxy_attention_heads: 4
  proxy_attention_dropout: 0.1
  
  # KL损失参数
  kl_loss_weight: 0.01  # KL损失权重，需要调优
  
  # 模态dropout参数
  modality_dropout: 0.2
  modality_dropout_warmup: 30
  use_adaptive_dropout: false  # 是否使用基于不确定性的自适应dropout
  uncertainty_dropout_temp: 1.0  # 温度参数
  
  # 其他正则化
  grad_clip: 1.0
  l2: 5e-5
```

---

## 🧪 实验计划

### 阶段1：验证不确定性加权
- 仅添加不确定性估计+加权融合（不加代理模态注意力）
- 对比baseline和v3版本

### 阶段2：添加代理模态注意力
- 在阶段1基础上添加CrossModalAttention
- 观察对test2的影响

### 阶段3：添加KL正则化
- 调优KL损失权重 (建议范围: 0.001 ~ 0.1)
- 注意：过大可能导致模态塌缩

### 阶段4：消融实验
- 验证各组件的贡献度

---

## 📊 预期效果

基于P-RMF的核心思想迁移，预期效果：

| 指标 | 当前最佳(v3) | 预期提升 | 预期结果 |
|------|-------------|----------|----------|
| test2 | 0.7621 | +2~3% | 0.78~0.79 |
| test1 | 0.8248 | +1~2% | 0.83~0.84 |
| test3 | 0.8873 | ±0.5% | ~0.89 |

**核心提升点**:
1. **不确定性加权**：缺失模态的不确定性高→权重低→减少对融合的负面影响
2. **代理模态**：从可用模态生成虚拟模态，补充缺失信息
3. **KL正则化**：鼓励各模态学习相似的潜在空间，便于跨模态补充

---

## ⚠️ 注意事项

1. **计算开销**: 新增的不确定性估计和跨模态注意力会增加约30%的计算量
2. **调参敏感**: KL损失权重需要仔细调优，过大会导致模态表示趋同
3. **数据适配**: P-RMF使用时序数据，当前模型使用utt级别特征，需要适配
4. **保持兼容**: 确保新模型兼容现有的数据加载和评估流程

---

## 📝 实现优先级

1. **高优先级** (核心技术):
   - [ ] 不确定性估计模块
   - [ ] 不确定性加权融合

2. **中优先级** (性能提升):
   - [ ] 代理模态跨模态注意力
   - [ ] KL散度正则化

3. **低优先级** (可选优化):
   - [ ] 自适应模态dropout
   - [ ] 重建损失（需要修改数据加载器）

---

*文档创建日期: 2026年1月30日*
*参考论文: Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data (ACL 2025)*
