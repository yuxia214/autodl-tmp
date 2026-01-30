# AttentionRobustV2 - 基于VAE的鲁棒多模态情感识别模型
# 
# 该模型结合了P-RMF的缺失模态处理思想，通过变分编码器实现：
# 1. 概率分布学习 (μ, σ) 代替确定性特征
# 2. 不确定性加权融合 w = softmax(1/σ)
# 3. 代理模态生成与交叉注意力
# 4. 多任务损失：分类损失 + KL散度 + 重建损失

from .attention_robust_v2 import AttentionRobustV2
from .modules import (
    VariationalMLPEncoder,
    VariationalLSTMEncoder,
    ModalityDecoder,
    UncertaintyWeightedFusion,
    ProxyCrossModalAttention,
    VAELossComputer
)

__all__ = [
    'AttentionRobustV2',
    'VariationalMLPEncoder',
    'VariationalLSTMEncoder', 
    'ModalityDecoder',
    'UncertaintyWeightedFusion',
    'ProxyCrossModalAttention',
    'VAELossComputer'
]
