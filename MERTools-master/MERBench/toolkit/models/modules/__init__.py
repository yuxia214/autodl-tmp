# toolkit/models/modules 初始化文件

from .encoder import MLPEncoder, LSTMEncoder

# 变分编码器组件 (AttentionRobustV2)
from .variational_encoder import (
    VariationalMLPEncoder,
    VariationalLSTMEncoder,
    ModalityDecoder,
    UncertaintyWeightedFusion,
    ProxyCrossModalAttention,
    VAELossComputer
)

__all__ = [
    'MLPEncoder',
    'LSTMEncoder',
    'VariationalMLPEncoder',
    'VariationalLSTMEncoder',
    'ModalityDecoder',
    'UncertaintyWeightedFusion',
    'ProxyCrossModalAttention',
    'VAELossComputer'
]
