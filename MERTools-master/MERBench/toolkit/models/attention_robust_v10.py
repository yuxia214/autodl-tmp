"""
AttentionRobustV10 - Audio-Video only robust model.

Design goals:
1) Remove text modality and keep AV-only training/inference.
2) Retain V7 stable components: VAE latent modeling, dynamic KL warmup,
   modality dropout, valence prior regularization, feature noise.
3) Keep proxy-attention fusion but adapt it to two modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoder import MLPEncoder, LSTMEncoder
from .modules.variational_encoder import (
    VariationalMLPEncoder,
    VariationalLSTMEncoder,
    ModalityDecoder,
)


class LinearKLScheduler:
    """Linear KL warmup scheduler."""

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


class AttentionRobustV10(nn.Module):
    """Audio-Video only variant of AttentionRobust."""

    def __init__(self, args):
        super(AttentionRobustV10, self).__init__()

        audio_dim = args.audio_dim
        video_dim = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        hidden_dim = args.hidden_dim
        dropout = args.dropout

        self.hidden_dim = hidden_dim
        self.grad_clip = args.grad_clip

        # VAE args
        self.use_vae = getattr(args, 'use_vae', True)
        self.kl_weight = getattr(args, 'kl_weight', 0.01)
        self.recon_weight = getattr(args, 'recon_weight', 0.1)
        self.cross_kl_weight = getattr(args, 'cross_kl_weight', 0.01)
        self.use_dynamic_kl = getattr(args, 'use_dynamic_kl', True)
        self.kl_warmup_epochs = getattr(args, 'kl_warmup_epochs', 20)

        # Fusion args
        self.use_proxy_attention = getattr(args, 'use_proxy_attention', True)
        self.fusion_temperature = getattr(args, 'fusion_temperature', 1.0)
        self.num_attention_heads = getattr(args, 'num_attention_heads', 4)

        # Modality dropout args
        self.modality_dropout = getattr(args, 'modality_dropout', 0.1)
        self.use_modality_dropout = getattr(args, 'use_modality_dropout', True)
        self.warmup_epochs = getattr(args, 'modality_dropout_warmup', 0)
        self.current_epoch = 0

        # Noise augmentation args
        self.feature_noise_std = getattr(args, 'feature_noise_std', 0.02)
        self.feature_noise_prob = getattr(args, 'feature_noise_prob', 0.3)
        self.feature_noise_warmup = getattr(args, 'feature_noise_warmup', 10)

        # Encoders/decoders
        if args.feat_type in ['utt']:
            if self.use_vae:
                self.audio_encoder = VariationalMLPEncoder(audio_dim, hidden_dim, dropout)
                self.video_encoder = VariationalMLPEncoder(video_dim, hidden_dim, dropout)
                self.audio_decoder = ModalityDecoder(hidden_dim, audio_dim, dropout)
                self.video_decoder = ModalityDecoder(hidden_dim, video_dim, dropout)
            else:
                self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
                self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            if self.use_vae:
                self.audio_encoder = VariationalLSTMEncoder(audio_dim, hidden_dim, dropout)
                self.video_encoder = VariationalLSTMEncoder(video_dim, hidden_dim, dropout)
                self.audio_decoder = ModalityDecoder(hidden_dim, audio_dim, dropout)
                self.video_decoder = ModalityDecoder(hidden_dim, video_dim, dropout)
            else:
                self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
                self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)
        else:
            raise ValueError(f"Unsupported feat_type: {args.feat_type}")

        if self.use_vae:
            if self.use_proxy_attention:
                self.cross_attn_audio = nn.MultiheadAttention(
                    hidden_dim, self.num_attention_heads, dropout=dropout, batch_first=True
                )
                self.cross_attn_video = nn.MultiheadAttention(
                    hidden_dim, self.num_attention_heads, dropout=dropout, batch_first=True
                )
                self.proxy_norm = nn.LayerNorm(hidden_dim)
                self.proxy_ffn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
            if self.use_dynamic_kl:
                self.kl_scheduler = LinearKLScheduler(
                    init_weight=0.0,
                    final_weight=self.kl_weight,
                    warmup_epochs=self.kl_warmup_epochs,
                )
        else:
            self.attention_mlp = MLPEncoder(hidden_dim * 2, hidden_dim, dropout)
            self.fc_att = nn.Linear(hidden_dim, 2)

        # Output heads
        self.feat_dropout = nn.Dropout(p=dropout)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

        # Emotion-valence prior
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

    def apply_feature_noise(self, x):
        if not self.training:
            return x
        if self.current_epoch < self.feature_noise_warmup:
            return x
        if self.feature_noise_std <= 0:
            return x
        if torch.rand(1).item() >= self.feature_noise_prob:
            return x
        noise = torch.randn_like(x) * self.feature_noise_std
        return x + noise

    def apply_modality_dropout(self, z_audio, z_video):
        """AV setting: each selected sample drops only one modality."""
        if not self.training or not self.use_modality_dropout:
            return z_audio, z_video
        if self.current_epoch < self.warmup_epochs:
            return z_audio, z_video

        if self.warmup_epochs > 0:
            progress = min(1.0, (self.current_epoch - self.warmup_epochs) / self.warmup_epochs)
            effective_dropout = self.modality_dropout * progress
        else:
            effective_dropout = self.modality_dropout

        batch_size = z_audio.size(0)
        masks = torch.ones(batch_size, 2, device=z_audio.device)

        for i in range(batch_size):
            if torch.rand(1).item() < effective_dropout:
                drop_mode = torch.randint(0, 2, (1,)).item()
                masks[i, drop_mode] = 0

        z_audio = z_audio * masks[:, 0:1]
        z_video = z_video * masks[:, 1:2]
        return z_audio, z_video

    def uncertainty_fusion_av(self, mu_a, mu_v, std_a, std_v):
        uncertainties = torch.cat(
            [std_a.mean(dim=-1, keepdim=True), std_v.mean(dim=-1, keepdim=True)],
            dim=1,
        )
        inv_uncertainties = 1.0 / (uncertainties + 1e-6)
        weights = F.softmax(inv_uncertainties / self.fusion_temperature, dim=1)
        proxy = weights[:, 0:1] * mu_a + weights[:, 1:2] * mu_v
        return proxy, weights

    def proxy_attention_av(self, proxy, mu_audio, mu_video, weights):
        proxy_exp = proxy.unsqueeze(1)
        audio_exp = mu_audio.unsqueeze(1)
        video_exp = mu_video.unsqueeze(1)

        attn_a, _ = self.cross_attn_audio(proxy_exp, audio_exp, audio_exp)
        attn_v, _ = self.cross_attn_video(proxy_exp, video_exp, video_exp)
        attn_a = attn_a.squeeze(1)
        attn_v = attn_v.squeeze(1)

        weighted_attn = weights[:, 0:1] * attn_a + weights[:, 1:2] * attn_v
        fused = self.proxy_norm(proxy + weighted_attn)
        fused = fused + self.proxy_ffn(fused)
        return fused

    @staticmethod
    def kl_to_standard_normal(mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    @staticmethod
    def cross_modal_kl(mu_a, lv_a, mu_v, lv_v):
        var_a = torch.exp(lv_a)
        var_v = torch.exp(lv_v) + 1e-6

        kl_av = 0.5 * (lv_v - lv_a + var_a / var_v + (mu_a - mu_v).pow(2) / var_v - 1)
        kl_va = 0.5 * (lv_a - lv_v + var_v / var_a + (mu_v - mu_a).pow(2) / var_a - 1)
        return 0.5 * (kl_av.mean() + kl_va.mean())

    def compute_vae_loss(self, mu_a, lv_a, mu_v, lv_v, audios, recon_a, videos, recon_v, kl_weight):
        kl_loss = 0.5 * (
            self.kl_to_standard_normal(mu_a, lv_a) +
            self.kl_to_standard_normal(mu_v, lv_v)
        )
        recon_loss = 0.5 * (
            F.mse_loss(recon_a, audios) +
            F.mse_loss(recon_v, videos)
        )
        cross_kl_loss = self.cross_modal_kl(mu_a, lv_a, mu_v, lv_v)

        return (
            kl_weight * kl_loss +
            self.recon_weight * recon_loss +
            self.cross_kl_weight * cross_kl_loss
        )

    def fuse_valence_with_emotion_prior(self, emos_out, raw_vals_out):
        if not self.use_valence_prior or emos_out.size(1) != self.emo_valence_centers.numel():
            return raw_vals_out, None

        probs = F.softmax(emos_out, dim=1)
        prior_val = torch.sum(probs * self.emo_valence_centers.unsqueeze(0), dim=1, keepdim=True)
        gate = torch.sigmoid(self.valence_prior_gate)
        vals_out = gate * raw_vals_out + (1.0 - gate) * prior_val
        return vals_out, prior_val

    def forward(self, batch):
        if self.use_vae:
            return self.forward_vae(batch)
        return self.forward_original(batch)

    def forward_vae(self, batch):
        audios = self.apply_feature_noise(batch['audios'])
        videos = self.apply_feature_noise(batch['videos'])

        z_a, mu_a, logvar_a, std_a = self.audio_encoder(audios)
        z_v, mu_v, logvar_v, std_v = self.video_encoder(videos)

        mu_a_dropped, mu_v_dropped = self.apply_modality_dropout(mu_a, mu_v)

        std_a_adj = torch.where(
            mu_a_dropped.abs().sum(dim=1, keepdim=True) == 0,
            torch.ones_like(std_a) * 1e6,
            std_a,
        )
        std_v_adj = torch.where(
            mu_v_dropped.abs().sum(dim=1, keepdim=True) == 0,
            torch.ones_like(std_v) * 1e6,
            std_v,
        )

        proxy, weights = self.uncertainty_fusion_av(mu_a_dropped, mu_v_dropped, std_a_adj, std_v_adj)
        if self.use_proxy_attention:
            fused = self.proxy_attention_av(proxy, mu_a_dropped, mu_v_dropped, weights)
        else:
            fused = proxy

        features = self.feat_dropout(fused)
        emos_out = self.fc_out_1(features)
        raw_vals_out = self.fc_out_2(features)
        vals_out, prior_val = self.fuse_valence_with_emotion_prior(emos_out, raw_vals_out)

        if self.training:
            recon_a = self.audio_decoder(z_a)
            recon_v = self.video_decoder(z_v)

            current_kl_weight = self.kl_weight
            if self.use_dynamic_kl:
                current_kl_weight = self.kl_scheduler.get_weight(self.current_epoch)

            interloss = self.compute_vae_loss(
                mu_a, logvar_a, mu_v, logvar_v,
                audios, recon_a, videos, recon_v,
                kl_weight=current_kl_weight,
            )

            if prior_val is not None:
                consistency = F.smooth_l1_loss(vals_out.view(-1, 1), prior_val.view(-1, 1))
                center_reg = F.mse_loss(self.emo_valence_centers, self.emo_center_init)
                interloss = interloss + self.valence_consistency_weight * consistency + self.valence_center_reg_weight * center_reg
        else:
            interloss = torch.tensor(0.0, device=audios.device)

        return features, emos_out, vals_out, interloss

    def forward_original(self, batch):
        audio_hidden = self.audio_encoder(batch['audios'])
        video_hidden = self.video_encoder(batch['videos'])

        audio_hidden, video_hidden = self.apply_modality_dropout(audio_hidden, video_hidden)

        multi_hidden1 = torch.cat([audio_hidden, video_hidden], dim=1)
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = F.softmax(attention, dim=1).unsqueeze(2)

        multi_hidden2 = torch.stack([audio_hidden, video_hidden], dim=2)
        fused_feat = torch.matmul(multi_hidden2, attention)

        features = self.feat_dropout(fused_feat.squeeze(2))
        emos_out = self.fc_out_1(features)
        raw_vals_out = self.fc_out_2(features)
        vals_out, _ = self.fuse_valence_with_emotion_prior(emos_out, raw_vals_out)
        interloss = torch.tensor(0.0, device=batch['audios'].device)

        return features, emos_out, vals_out, interloss
