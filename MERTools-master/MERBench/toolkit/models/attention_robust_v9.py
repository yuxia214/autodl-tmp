'''
AttentionRobustV9 - 面向MER2023 challenge指标的定向迭代

目标:
- 重点提升 test2 (MER-NOISE) 的 Combined 指标
- 在保持 test1/test3 的同时增强缺失/噪声鲁棒性

核心设计:
1) Quality-aware fusion
   - 在不确定性权重基础上引入可学习质量打分
2) Cross-modal imputation
   - 训练期随机模态缺失并用其余模态补全隐藏表示
3) Teacher-Student consistency
   - clean teacher vs corrupted student 的输出一致性约束
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoder import MLPEncoder, LSTMEncoder
from .modules.variational_encoder import (
    VariationalMLPEncoder,
    VariationalLSTMEncoder,
    ModalityDecoder,
    UncertaintyWeightedFusion,
    GatedUncertaintyFusion,
    ProxyCrossModalAttention,
    VAELossComputer,
)


class LinearKLScheduler:
    """线性KL warmup调度器"""

    def __init__(self, init_weight=0.0, final_weight=0.01, warmup_epochs=20):
        self.init_weight = init_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs

    def get_weight(self, epoch):
        if self.warmup_epochs <= 0:
            return self.final_weight
        if epoch < self.warmup_epochs:
            return self.init_weight + (self.final_weight - self.init_weight) * (epoch / self.warmup_epochs)
        return self.final_weight


class AttentionRobustV9(nn.Module):
    """
    AttentionRobustV9

    训练时使用student分支输出参与监督，teacher仅用于一致性约束。
    推理时使用clean路径。
    """

    def __init__(self, args):
        super(AttentionRobustV9, self).__init__()

        text_dim = args.text_dim
        audio_dim = args.audio_dim
        video_dim = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.hidden_dim = hidden_dim
        self.grad_clip = args.grad_clip

        # ==================== VAE参数 ====================
        self.use_vae = getattr(args, 'use_vae', True)
        self.kl_weight = getattr(args, 'kl_weight', 0.01)
        self.recon_weight = getattr(args, 'recon_weight', 0.1)
        self.cross_kl_weight = getattr(args, 'cross_kl_weight', 0.01)
        self.use_dynamic_kl = getattr(args, 'use_dynamic_kl', True)
        self.kl_warmup_epochs = getattr(args, 'kl_warmup_epochs', 20)

        # ==================== 融合参数 ====================
        self.use_proxy_attention = getattr(args, 'use_proxy_attention', True)
        self.num_attention_heads = getattr(args, 'num_attention_heads', 4)
        self.fusion_temperature = getattr(args, 'fusion_temperature', 1.0)
        self.use_gated_uncertainty = getattr(args, 'use_gated_uncertainty', True)
        self.gate_alpha = getattr(args, 'gate_alpha', 0.5)
        self.reliability_temperature = getattr(args, 'reliability_temperature', 1.0)

        # ==================== V9新增参数 ====================
        self.quality_weight = getattr(args, 'quality_weight', 0.6)
        self.impute_loss_weight = getattr(args, 'impute_loss_weight', 0.10)
        self.consistency_emo_weight = getattr(args, 'consistency_emo_weight', 0.08)
        self.consistency_val_weight = getattr(args, 'consistency_val_weight', 0.05)

        self.corruption_max_rate = getattr(args, 'corruption_max_rate', 0.45)
        self.corruption_warmup_epochs = getattr(args, 'corruption_warmup_epochs', 25)
        self.double_mask_ratio = getattr(args, 'double_mask_ratio', 0.35)
        self.latent_noise_std = getattr(args, 'latent_noise_std', 0.02)

        self.weight_consistency_weight = getattr(args, 'weight_consistency_weight', 0.02)
        self.modality_agreement_weight = getattr(args, 'modality_agreement_weight', 0.008)

        # ==================== 模态dropout参数 ====================
        self.modality_dropout = getattr(args, 'modality_dropout', 0.18)
        self.use_modality_dropout = getattr(args, 'use_modality_dropout', True)
        self.warmup_epochs = getattr(args, 'modality_dropout_warmup', 0)

        # ==================== 训练时特征噪声参数 ====================
        self.feature_noise_std = getattr(args, 'feature_noise_std', 0.02)
        self.feature_noise_prob = getattr(args, 'feature_noise_prob', 0.3)
        self.feature_noise_warmup = getattr(args, 'feature_noise_warmup', 10)

        self.current_epoch = 0

        # ==================== 编码器 ====================
        if args.feat_type in ['utt']:
            if self.use_vae:
                self.audio_encoder = VariationalMLPEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = VariationalMLPEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = VariationalMLPEncoder(video_dim, hidden_dim, dropout)

                self.audio_decoder = ModalityDecoder(hidden_dim, audio_dim, dropout)
                self.text_decoder = ModalityDecoder(hidden_dim, text_dim, dropout)
                self.video_decoder = ModalityDecoder(hidden_dim, video_dim, dropout)
            else:
                self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = MLPEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            if self.use_vae:
                self.audio_encoder = VariationalLSTMEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = VariationalLSTMEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = VariationalLSTMEncoder(video_dim, hidden_dim, dropout)

                self.audio_decoder = ModalityDecoder(hidden_dim, audio_dim, dropout)
                self.text_decoder = ModalityDecoder(hidden_dim, text_dim, dropout)
                self.video_decoder = ModalityDecoder(hidden_dim, video_dim, dropout)
            else:
                self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = LSTMEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        # ==================== 融合模块 ====================
        if self.use_vae:
            if self.use_gated_uncertainty:
                self.uncertainty_fusion = GatedUncertaintyFusion(
                    hidden_dim,
                    temperature=self.fusion_temperature,
                    gate_alpha=self.gate_alpha,
                )
            else:
                self.uncertainty_fusion = UncertaintyWeightedFusion(
                    hidden_dim,
                    temperature=self.fusion_temperature,
                )

            if self.use_proxy_attention:
                self.proxy_attention = ProxyCrossModalAttention(
                    hidden_dim,
                    num_heads=self.num_attention_heads,
                    dropout=dropout,
                )

            self.loss_computer = VAELossComputer(
                kl_weight=self.kl_weight,
                recon_weight=self.recon_weight,
                cross_kl_weight=self.cross_kl_weight,
            )
            if self.use_dynamic_kl:
                self.kl_scheduler = LinearKLScheduler(
                    init_weight=0.0,
                    final_weight=self.kl_weight,
                    warmup_epochs=self.kl_warmup_epochs,
                )
        else:
            self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)
            self.fc_att = nn.Linear(hidden_dim, 3)

        # ==================== V9新增模块 ====================
        # 输入: [mu, std, recon_err, observed_flag]
        self.quality_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 用其余两模态补全当前模态
        self.impute_audio = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.impute_text = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.impute_video = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ==================== 输出层 ====================
        self.feat_dropout = nn.Dropout(p=dropout)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

        # ==================== Emotion-Valence一致性 ====================
        self.use_valence_prior = getattr(args, 'use_valence_prior', True)
        self.valence_consistency_weight = getattr(args, 'valence_consistency_weight', 0.08)
        self.valence_center_reg_weight = getattr(args, 'valence_center_reg_weight', 0.005)

        if output_dim1 == 6 and output_dim2 == 1:
            init_centers = torch.tensor([0.0, -2.0, 2.0, -2.0, -1.0, 0.3], dtype=torch.float32)
        else:
            init_centers = torch.zeros(max(output_dim1, 1), dtype=torch.float32)

        self.register_buffer('emo_center_init', init_centers.clone())
        self.emo_valence_centers = nn.Parameter(init_centers.clone())
        self.valence_prior_gate = nn.Parameter(torch.tensor(0.5))

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def apply_modality_dropout(self, z_audio, z_text, z_video):
        if not self.training or not self.use_modality_dropout:
            return z_audio, z_text, z_video

        if self.current_epoch < self.warmup_epochs:
            return z_audio, z_text, z_video

        if self.warmup_epochs > 0:
            progress = min(1.0, (self.current_epoch - self.warmup_epochs) / self.warmup_epochs)
            effective_dropout = self.modality_dropout * progress
        else:
            effective_dropout = self.modality_dropout

        batch_size = z_audio.size(0)
        device = z_audio.device
        masks = torch.ones(batch_size, 3, device=device)

        for i in range(batch_size):
            if torch.rand(1).item() < effective_dropout:
                drop_mode = torch.randint(0, 6, (1,)).item()
                if drop_mode == 0:
                    masks[i, 0] = 0
                elif drop_mode == 1:
                    masks[i, 1] = 0
                elif drop_mode == 2:
                    masks[i, 2] = 0
                elif drop_mode == 3:
                    masks[i, 0] = 0
                    masks[i, 1] = 0
                elif drop_mode == 4:
                    masks[i, 0] = 0
                    masks[i, 2] = 0
                elif drop_mode == 5:
                    masks[i, 1] = 0
                    masks[i, 2] = 0

        z_audio = z_audio * masks[:, 0:1]
        z_text = z_text * masks[:, 1:2]
        z_video = z_video * masks[:, 2:3]
        return z_audio, z_text, z_video

    def maybe_add_latent_noise(self, x):
        if not self.training:
            return x

        if self.latent_noise_std > 0:
            x = x + torch.randn_like(x) * self.latent_noise_std

        if self.current_epoch >= self.feature_noise_warmup and self.feature_noise_std > 0:
            if torch.rand(1).item() < self.feature_noise_prob:
                x = x + torch.randn_like(x) * self.feature_noise_std

        return x

    def current_corruption_rate(self):
        if self.corruption_warmup_epochs <= 0:
            return self.corruption_max_rate
        ratio = min(1.0, float(self.current_epoch) / float(self.corruption_warmup_epochs))
        return self.corruption_max_rate * ratio

    def sample_corruption_masks(self, batch_size, device):
        masks = torch.ones(batch_size, 3, device=device)
        if not self.training:
            return masks

        p = self.current_corruption_rate()
        if p <= 0:
            return masks

        for i in range(batch_size):
            if torch.rand(1).item() < p:
                if torch.rand(1).item() < self.double_mask_ratio:
                    pair_type = torch.randint(0, 3, (1,)).item()
                    if pair_type == 0:
                        masks[i, 0] = 0
                        masks[i, 1] = 0
                    elif pair_type == 1:
                        masks[i, 0] = 0
                        masks[i, 2] = 0
                    else:
                        masks[i, 1] = 0
                        masks[i, 2] = 0
                else:
                    m = torch.randint(0, 3, (1,)).item()
                    masks[i, m] = 0
        return masks

    def fuse_valence_with_emotion_prior(self, emos_out, raw_vals_out):
        if not self.use_valence_prior or emos_out.size(1) != self.emo_valence_centers.numel():
            return raw_vals_out, None

        probs = F.softmax(emos_out, dim=1)
        prior_val = torch.sum(probs * self.emo_valence_centers.unsqueeze(0), dim=1, keepdim=True)
        gate = torch.sigmoid(self.valence_prior_gate)
        vals_out = gate * raw_vals_out + (1.0 - gate) * prior_val
        return vals_out, prior_val

    def calc_recon_error(self, original, recon):
        return (recon - original).pow(2).mean(dim=1, keepdim=True)

    def modality_agreement_loss(self, mu_a, mu_t, mu_v):
        mu_a = F.normalize(mu_a, dim=1)
        mu_t = F.normalize(mu_t, dim=1)
        mu_v = F.normalize(mu_v, dim=1)

        loss_at = 1.0 - (mu_a * mu_t).sum(dim=1).mean()
        loss_av = 1.0 - (mu_a * mu_v).sum(dim=1).mean()
        loss_tv = 1.0 - (mu_t * mu_v).sum(dim=1).mean()
        return (loss_at + loss_av + loss_tv) / 3.0

    def impute_missing(self, mu_a, mu_t, mu_v, masks):
        pred_a = self.impute_audio(torch.cat([mu_t, mu_v], dim=1))
        pred_t = self.impute_text(torch.cat([mu_a, mu_v], dim=1))
        pred_v = self.impute_video(torch.cat([mu_a, mu_t], dim=1))

        ma = masks[:, 0:1]
        mt = masks[:, 1:2]
        mv = masks[:, 2:3]

        out_a = ma * mu_a + (1.0 - ma) * pred_a
        out_t = mt * mu_t + (1.0 - mt) * pred_t
        out_v = mv * mu_v + (1.0 - mv) * pred_v
        return out_a, out_t, out_v, pred_a, pred_t, pred_v

    def calc_imputation_loss(self, preds, targets, masks):
        losses = []
        for pred, target, mask in zip(preds, targets, [masks[:, 0:1], masks[:, 1:2], masks[:, 2:3]]):
            miss = 1.0 - mask
            miss_num = miss.sum()
            if miss_num.item() < 1:
                continue
            per = F.smooth_l1_loss(pred, target.detach(), reduction='none').mean(dim=1, keepdim=True)
            losses.append((per * miss).sum() / miss_num.clamp_min(1.0))

        if len(losses) == 0:
            return torch.tensor(0.0, device=targets[0].device)
        return sum(losses) / len(losses)

    def calc_quality_logits(self, mu_list, std_list, recon_errors, masks):
        logits = []
        for mu, std, err, obs in zip(mu_list, std_list, recon_errors, [masks[:, 0:1], masks[:, 1:2], masks[:, 2:3]]):
            inp = torch.cat([mu, std, err, obs], dim=1)
            logits.append(self.quality_net(inp))
        return torch.cat(logits, dim=1)

    def fuse_with_quality(self, mu_list, std_list, quality_logits):
        # 基础权重来自不确定性融合
        _, base_weights = self.uncertainty_fusion(mu_list, std_list)

        # 质量修正权重
        logits = torch.log(base_weights.clamp_min(1e-6)) + self.quality_weight * quality_logits
        temp = max(float(self.reliability_temperature), 1e-3)
        weights = F.softmax(logits / temp, dim=1)

        mu_stack = torch.stack(mu_list, dim=1)
        proxy = (mu_stack * weights.unsqueeze(-1)).sum(dim=1)
        return proxy, weights, base_weights

    def forward(self, batch):
        if self.use_vae:
            return self.forward_vae(batch)
        return self.forward_original(batch)

    def forward_vae(self, batch):
        audios = batch['audios']
        texts = batch['texts']
        videos = batch['videos']

        # 1) clean编码
        z_a, mu_a, logvar_a, std_a = self.audio_encoder(audios)
        z_t, mu_t, logvar_t, std_t = self.text_encoder(texts)
        z_v, mu_v, logvar_v, std_v = self.video_encoder(videos)

        # 2) 重建与recon误差
        recon_a = self.audio_decoder(z_a)
        recon_t = self.text_decoder(z_t)
        recon_v = self.video_decoder(z_v)

        err_a = self.calc_recon_error(audios, recon_a)
        err_t = self.calc_recon_error(texts, recon_t)
        err_v = self.calc_recon_error(videos, recon_v)

        # teacher: clean路径
        with torch.no_grad():
            ones_mask = torch.ones(mu_a.size(0), 3, device=mu_a.device)
            q_teacher = self.calc_quality_logits(
                [mu_a, mu_t, mu_v],
                [std_a, std_t, std_v],
                [err_a, err_t, err_v],
                ones_mask,
            )
            proxy_teacher, w_teacher, _ = self.fuse_with_quality(
                [mu_a, mu_t, mu_v],
                [std_a, std_t, std_v],
                q_teacher,
            )
            if self.use_proxy_attention:
                fused_teacher = self.proxy_attention(proxy_teacher, mu_a, mu_t, mu_v, w_teacher)
            else:
                fused_teacher = proxy_teacher
            teacher_emos = self.fc_out_1(fused_teacher)
            teacher_vals_raw = self.fc_out_2(fused_teacher)
            teacher_vals, _ = self.fuse_valence_with_emotion_prior(teacher_emos, teacher_vals_raw)

        # student: corrupted路径
        mu_a_s = self.maybe_add_latent_noise(mu_a)
        mu_t_s = self.maybe_add_latent_noise(mu_t)
        mu_v_s = self.maybe_add_latent_noise(mu_v)

        # 先做原有模态dropout，再做显式随机缺失
        mu_a_s, mu_t_s, mu_v_s = self.apply_modality_dropout(mu_a_s, mu_t_s, mu_v_s)

        masks = self.sample_corruption_masks(mu_a.size(0), mu_a.device)
        mu_a_s = mu_a_s * masks[:, 0:1]
        mu_t_s = mu_t_s * masks[:, 1:2]
        mu_v_s = mu_v_s * masks[:, 2:3]

        mu_a_i, mu_t_i, mu_v_i, pred_a, pred_t, pred_v = self.impute_missing(mu_a_s, mu_t_s, mu_v_s, masks)

        # 缺失模态不确定性抬高，防止不可信来源主导
        std_a_s = torch.where(masks[:, 0:1] > 0, std_a, torch.ones_like(std_a) * 1e6)
        std_t_s = torch.where(masks[:, 1:2] > 0, std_t, torch.ones_like(std_t) * 1e6)
        std_v_s = torch.where(masks[:, 2:3] > 0, std_v, torch.ones_like(std_v) * 1e6)

        q_student = self.calc_quality_logits(
            [mu_a_i, mu_t_i, mu_v_i],
            [std_a_s, std_t_s, std_v_s],
            [err_a, err_t, err_v],
            masks,
        )
        proxy_student, w_student, w_base = self.fuse_with_quality(
            [mu_a_i, mu_t_i, mu_v_i],
            [std_a_s, std_t_s, std_v_s],
            q_student,
        )

        if self.use_proxy_attention:
            fused_student = self.proxy_attention(proxy_student, mu_a_i, mu_t_i, mu_v_i, w_student)
        else:
            fused_student = proxy_student

        features = self.feat_dropout(fused_student)
        emos_out = self.fc_out_1(features)
        raw_vals_out = self.fc_out_2(features)
        vals_out, prior_val = self.fuse_valence_with_emotion_prior(emos_out, raw_vals_out)

        # 3) interloss
        if self.use_dynamic_kl:
            self.loss_computer.kl_weight = self.kl_scheduler.get_weight(self.current_epoch)

        interloss = self.loss_computer.compute(
            mu_list=[mu_a, mu_t, mu_v],
            logvar_list=[logvar_a, logvar_t, logvar_v],
            originals=[audios, texts, videos],
            reconstructions=[recon_a, recon_t, recon_v],
        )

        # valence先验一致性
        if prior_val is not None:
            prior_consistency = F.smooth_l1_loss(vals_out.view(-1, 1), prior_val.view(-1, 1))
            center_reg = F.mse_loss(self.emo_valence_centers, self.emo_center_init)
            interloss = interloss + self.valence_consistency_weight * prior_consistency + self.valence_center_reg_weight * center_reg

        # 缺失补全损失
        imp_loss = self.calc_imputation_loss(
            [pred_a, pred_t, pred_v],
            [mu_a, mu_t, mu_v],
            masks,
        )
        interloss = interloss + self.impute_loss_weight * imp_loss

        # clean-student一致性
        emo_cons = F.kl_div(
            F.log_softmax(emos_out, dim=1),
            F.softmax(teacher_emos.detach(), dim=1),
            reduction='batchmean',
        )
        val_cons = F.smooth_l1_loss(vals_out, teacher_vals.detach())
        interloss = interloss + self.consistency_emo_weight * emo_cons + self.consistency_val_weight * val_cons

        # 权重一致性 + 模态一致性
        weight_consistency = F.mse_loss(w_student, w_base)
        agreement = self.modality_agreement_loss(mu_a_i, mu_t_i, mu_v_i)
        interloss = interloss + self.weight_consistency_weight * weight_consistency + self.modality_agreement_weight * agreement

        return features, emos_out, vals_out, interloss

    def forward_original(self, batch):
        audio_hidden = self.audio_encoder(batch['audios'])
        text_hidden = self.text_encoder(batch['texts'])
        video_hidden = self.video_encoder(batch['videos'])

        audio_hidden, text_hidden, video_hidden = self.apply_modality_dropout(audio_hidden, text_hidden, video_hidden)

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1)
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = F.softmax(attention, dim=1)
        attention = attention.unsqueeze(2)

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2)
        fused_feat = torch.matmul(multi_hidden2, attention)

        features = fused_feat.squeeze(2)
        features = self.feat_dropout(features)

        emos_out = self.fc_out_1(features)
        raw_vals_out = self.fc_out_2(features)
        vals_out, _ = self.fuse_valence_with_emotion_prior(emos_out, raw_vals_out)
        interloss = torch.tensor(0.0, device=batch['audios'].device)

        return features, emos_out, vals_out, interloss
