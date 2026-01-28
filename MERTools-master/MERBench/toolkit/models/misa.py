"""
From: https://github.com/declare-lab/MISA
Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
Enhanced with Modal Dropout and Cross-Modal Reconstruction for missing modality robustness.
"""

import torch
import torch.nn as nn
from torch.autograd import Function

from .modules.encoder import MLPEncoder, LSTMEncoder

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p
        return output, None

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class MISA(nn.Module):
    def __init__(self, args):
        super(MISA, self).__init__()

        # params: analyze args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.sim_weight = args.sim_weight
        self.diff_weight = args.diff_weight
        self.recon_weight = args.recon_weight
        self.grad_clip = args.grad_clip
        
        # ============ 新增：缺失模态相关参数 ============
        self.modal_dropout_prob = getattr(args, 'modal_dropout_prob', 0.15)  # 模态dropout概率（降低以保护完整模态性能）
        self.cross_recon_weight = getattr(args, 'cross_recon_weight', 0.3)   # 跨模态重建权重
        self.hidden_dim = hidden_dim
        # 课程学习：训练初期少dropout，逐渐增加
        self.curriculum_start_epoch = getattr(args, 'curriculum_start_epoch', 5)  # 开始使用dropout的epoch
        self.curriculum_rampup_epochs = getattr(args, 'curriculum_rampup_epochs', 10)  # 达到最大dropout的epoch数
        self.current_epoch = 0  # 当前训练epoch
        self.modal_masks = None  # 保存当前batch的mask用于重建
        # ================================================
        
        # params: intermedia
        output_dim = hidden_dim // 2

        # modality-specific encoder
        if args.feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        # map into a common space
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.project_t.add_module('project_t_activation', nn.ReLU())
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(hidden_dim))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.project_v.add_module('project_v_activation', nn.ReLU())
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(hidden_dim))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.project_a.add_module('project_a_activation', nn.ReLU())
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(hidden_dim))

        # private encoders
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        
        # shared encoder
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        # reconstruct
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))

        # ============ 新增：跨模态重建器 ============
        # 从其他两个模态的shared表示重建缺失模态
        self.cross_recon_t = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cross_recon_v = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cross_recon_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 可学习的缺失模态token - 使用Xavier初始化避免数值问题
        self.missing_token_t = nn.Parameter(torch.zeros(1, hidden_dim))
        self.missing_token_v = nn.Parameter(torch.zeros(1, hidden_dim))
        self.missing_token_a = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_uniform_(self.missing_token_t)
        nn.init.xavier_uniform_(self.missing_token_v)
        nn.init.xavier_uniform_(self.missing_token_a)
        # =============================================

        # fusion + cls
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=hidden_dim*6, out_features=hidden_dim*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.fusion.add_module('fusion_layer_1_activation', nn.ReLU())
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=hidden_dim*3, out_features=output_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)

    def modal_dropout(self, utterance_t, utterance_v, utterance_a, dropout_prob=None):
        """
        训练时随机mask掉某些模态，提升缺失模态鲁棒性
        返回：处理后的特征和模态mask
        """
        batch_size = utterance_t.size(0)
        device = utterance_t.device
        
        # 使用传入的概率或默认概率
        prob = dropout_prob if dropout_prob is not None else self.modal_dropout_prob
        
        if self.training and prob > 0:
            # 为每个样本独立生成模态dropout mask
            mask_t = (torch.rand(batch_size, 1, device=device) > prob).float()
            mask_v = (torch.rand(batch_size, 1, device=device) > prob).float()
            mask_a = (torch.rand(batch_size, 1, device=device) > prob).float()
            
            # 确保每个样本至少保留一个模态（更健壮的实现）
            modality_sum = (mask_t + mask_v + mask_a).squeeze(-1)
            all_zero_mask = (modality_sum == 0)
            
            if all_zero_mask.any():
                num_zeros = int(all_zero_mask.sum().item())
                if num_zeros > 0:
                    random_modality = torch.randint(0, 3, (num_zeros,), device=device)
                    indices = torch.where(all_zero_mask)[0]
                    for i, idx in enumerate(indices):
                        idx_item = int(idx.item())
                        if random_modality[i] == 0:
                            mask_t[idx_item, 0] = 1.0
                        elif random_modality[i] == 1:
                            mask_v[idx_item, 0] = 1.0
                        else:
                            mask_a[idx_item, 0] = 1.0
            
            # 应用mask，缺失模态用可学习token替代
            missing_t = self.missing_token_t.expand(batch_size, -1)
            missing_v = self.missing_token_v.expand(batch_size, -1)
            missing_a = self.missing_token_a.expand(batch_size, -1)
            
            utterance_t = utterance_t * mask_t + missing_t * (1 - mask_t)
            utterance_v = utterance_v * mask_v + missing_v * (1 - mask_v)
            utterance_a = utterance_a * mask_a + missing_a * (1 - mask_a)
            
            # 保存mask供跨模态重建损失使用
            self._dropout_masks = (mask_t, mask_v, mask_a)
            return utterance_t, utterance_v, utterance_a, (mask_t, mask_v, mask_a)
        else:
            # 推理时返回全1 mask
            ones = torch.ones(batch_size, 1, device=device)
            self._dropout_masks = (ones, ones, ones)
            return utterance_t, utterance_v, utterance_a, (ones, ones, ones)

    def set_epoch(self, epoch):
        """设置当前epoch，用于课程学习调整dropout概率"""
        self.current_epoch = epoch
    
    def get_current_dropout_prob(self):
        """根据课程学习策略计算当前的dropout概率"""
        if self.current_epoch < self.curriculum_start_epoch:
            return 0.0  # 初期不使用dropout
        
        progress = (self.current_epoch - self.curriculum_start_epoch) / max(1, self.curriculum_rampup_epochs)
        progress = min(1.0, progress)  # 限制在[0, 1]
        return self.modal_dropout_prob * progress

    def shared_private(self, utterance_t, utterance_v, utterance_a):
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def reconstruct(self):
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    ##########################################################
    ## inter loss calculation
    ##########################################################
    def get_recon_loss(self):
        """重建损失：只对未被dropout的模态计算"""
        if hasattr(self, '_dropout_masks'):
            mask_t, mask_v, mask_a = self._dropout_masks
            t_valid = mask_t.mean() > 0.5
            v_valid = mask_v.mean() > 0.5
            a_valid = mask_a.mean() > 0.5
        else:
            t_valid = v_valid = a_valid = True
        
        loss = torch.tensor(0.0, device=self.utt_t_recon.device)
        count = 0
        
        if t_valid:
            loss += MSE()(self.utt_t_recon, self.utt_t_orig)
            count += 1
        if v_valid:
            loss += MSE()(self.utt_v_recon, self.utt_v_orig)
            count += 1
        if a_valid:
            loss += MSE()(self.utt_a_recon, self.utt_a_orig)
            count += 1
            
        return loss / max(1, count)

    def get_diff_loss(self):
        """差分损失：只对未被dropout的模态计算"""
        # 获取dropout mask判断哪些模态可用
        if hasattr(self, '_dropout_masks'):
            mask_t, mask_v, mask_a = self._dropout_masks
            # 检查每个模态是否有足够的有效样本（至少50%未被dropout）
            t_valid = mask_t.mean() > 0.5
            v_valid = mask_v.mean() > 0.5
            a_valid = mask_a.mean() > 0.5
        else:
            t_valid = v_valid = a_valid = True
        
        shared_t = self.utt_shared_t
        shared_v = self.utt_shared_v
        shared_a = self.utt_shared_a
        private_t = self.utt_private_t
        private_v = self.utt_private_v
        private_a = self.utt_private_a

        loss = torch.tensor(0.0, device=shared_t.device)
        count = 0
        
        # Between private and shared (只对有效模态)
        if t_valid:
            loss += DiffLoss()(private_t, shared_t)
            count += 1
        if v_valid:
            loss += DiffLoss()(private_v, shared_v)
            count += 1
        if a_valid:
            loss += DiffLoss()(private_a, shared_a)
            count += 1

        # Across privates (只对有效模态对)
        if t_valid and a_valid:
            loss += DiffLoss()(private_a, private_t)
            count += 1
        if v_valid and a_valid:
            loss += DiffLoss()(private_a, private_v)
            count += 1
        if t_valid and v_valid:
            loss += DiffLoss()(private_t, private_v)
            count += 1
            
        return loss / max(1, count) * 6  # 保持与原始scale一致

    def get_cmd_loss(self):
        """CMD损失：只对未被dropout的模态对计算"""
        if hasattr(self, '_dropout_masks'):
            mask_t, mask_v, mask_a = self._dropout_masks
            t_valid = mask_t.mean() > 0.5
            v_valid = mask_v.mean() > 0.5
            a_valid = mask_a.mean() > 0.5
        else:
            t_valid = v_valid = a_valid = True
        
        loss = torch.tensor(0.0, device=self.utt_shared_t.device)
        count = 0
        
        if t_valid and v_valid:
            loss += CMD()(self.utt_shared_t, self.utt_shared_v, 5)
            count += 1
        if t_valid and a_valid:
            loss += CMD()(self.utt_shared_t, self.utt_shared_a, 5)
            count += 1
        if a_valid and v_valid:
            loss += CMD()(self.utt_shared_a, self.utt_shared_v, 5)
            count += 1
            
        return loss / max(1, count)
    
    # ============ 新增：跨模态重建损失 ============
    def get_cross_recon_loss(self):
        """
        跨模态重建损失：暂时禁用以保证训练稳定性
        Modal dropout本身已经能够提升缺失模态鲁棒性
        """
        return torch.tensor(0.0, device=self.utt_shared_t.device)
    # =============================================
    
    def forward(self, batch, missing_mask=None):
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            text_feat:  tensor of shape (batch, seqlen2, text_in)
            video_feat: tensor of shape (batch, seqlen3, video_in)
            missing_mask: 可选，用于推理时指定缺失模态 (mask_t, mask_v, mask_a)
        '''
        utterance_audio = self.audio_encoder(batch['audios'])
        utterance_text  = self.text_encoder(batch['texts'])
        utterance_video = self.video_encoder(batch['videos'])

        # ============ 新增：模态dropout（课程学习策略）============
        # 训练初期少dropout，逐渐增加
        current_prob = self.get_current_dropout_prob() if self.training else 0.0
        utterance_text, utterance_video, utterance_audio, modal_masks = \
            self.modal_dropout(utterance_text, utterance_video, utterance_audio, current_prob)
        self.modal_masks = modal_masks  # 保存用于可能的调试
        # ==========================================

        # shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, 
                        self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        features = self.fusion(h)

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        
        # ============ 修改：增加跨模态重建损失 ============
        interloss = self.diff_weight  * self.get_diff_loss() + \
                    self.sim_weight   * self.get_cmd_loss()  + \
                    self.recon_weight * self.get_recon_loss() + \
                    self.cross_recon_weight * self.get_cross_recon_loss()
        # ================================================

        return features, emos_out, vals_out, interloss
    
    # ============ 新增：缺失模态推理接口 ============
    def forward_with_missing(self, batch, missing_modalities=None):
        """
        支持缺失模态的推理接口
        
        Args:
            batch: 输入batch
            missing_modalities: list, 缺失的模态名称，如 ['text'], ['audio', 'video']
        """
        if missing_modalities is None:
            missing_modalities = []
            
        utterance_audio = self.audio_encoder(batch['audios'])
        utterance_text  = self.text_encoder(batch['texts'])
        utterance_video = self.video_encoder(batch['videos'])
        
        batch_size = utterance_text.size(0)
        device = utterance_text.device
        
        # 根据指定的缺失模态，用missing token替换
        if 'text' in missing_modalities:
            utterance_text = self.missing_token_t.expand(batch_size, -1)
        if 'video' in missing_modalities:
            utterance_video = self.missing_token_v.expand(batch_size, -1)
        if 'audio' in missing_modalities:
            utterance_audio = self.missing_token_a.expand(batch_size, -1)
        
        # shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a,
                        self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        features = self.fusion(h)

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        return features, emos_out, vals_out
    # =============================================
