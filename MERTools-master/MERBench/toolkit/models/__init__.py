"""
get_models: get models and load default configs;
link: https://github.com/thuiar/MMSA-FET/tree/master
"""
import torch

from .tfn import TFN
from .lmf import LMF
from .mfn import MFN
from .mfm import MFM
from .mult import MULT
from .misa import MISA
from .mctn import MCTN
from .mmim import MMIM
from .graph_mfn import Graph_MFN
from .attention import Attention
from .attention_robust import AttentionRobust
from .attention_robust_v2 import AttentionRobustV2
from .attention_robust_v4 import AttentionRobustV4
from .attention_robust_v5 import AttentionRobustV5
from .attention_robust_v6 import AttentionRobustV6
from .attention_robust_v7 import AttentionRobustV7
from .attention_robust_v8 import AttentionRobustV8
from .attention_robust_v9 import AttentionRobustV9
from .attention_robust_v10 import AttentionRobustV10

class get_models(torch.nn.Module):
    def __init__(self, args):
        super(get_models, self).__init__()
        # misa/mmim在有些参数配置下会存在梯度爆炸的风险
        # tfn 显存占比比较高

        MODEL_MAP = {

            # 特征压缩到句子级再处理，所以支持 utt/align/unalign
            'attention': Attention,
            'attention_robust': AttentionRobust,  # 增强版attention，支持模态dropout
            'attention_robust_v2': AttentionRobustV2,  # VAE版attention，概率化多模态融合
            'attention_robust_v4': AttentionRobustV4,  # V4版attention，对比学习+门控融合
            'attention_robust_v5': AttentionRobustV5,  # V5版attention，深度VAE+自适应融合+Mixup
            'attention_robust_v6': AttentionRobustV6,  # V6版attention，V2稳态增强 + 动态KL
            'attention_robust_v7': AttentionRobustV7,  # V7版attention，情感-价度一致性 + 噪声增强
            'attention_robust_v8': AttentionRobustV8,  # V8版attention，双路径融合 + 可靠度建模
            'attention_robust_v9': AttentionRobustV9,  # V9版attention，质量估计 + 缺失补全 + 一致性学习
            'attention_robust_v10': AttentionRobustV10,  # V10版attention，AV-only鲁棒融合
            'lmf': LMF,
            'misa': MISA,
            'mmim': MMIM,
            'tfn': TFN,

            # 只支持align
            'mfn': MFN, # slow
            'graph_mfn': Graph_MFN, # slow
            'mfm': MFM, # slow
            'mctn': MCTN, # slow

            # 支持align/unalign
            'mult': MULT, # slow

        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
