'''
AttentionRobustV2 - 基于P-RMF的概率化多模态融合模型

核心改进：从确定性特征学习转向概率分布学习
- 使用VAE编码器输出 (μ, σ) 而非固定向量
- 基于不确定性的动态加权融合
- 代理模态跨模态注意力
- KL散度 + 重建损失正则化

Reference: Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data (ACL 2025)

使用方法:
1. 将此文件夹复制到服务器 MERBench 目录下
2. 运行 python setup_v2.py 自动安装到toolkit
3. 或手动将模型文件复制到 toolkit/models/ 下
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 动态导入：支持独立运行和集成运行
try:
    # 集成到toolkit时的导入
    from toolkit.models.modules.encoder import MLPEncoder, LSTMEncoder
    from toolkit.models.modules.variational_encoder import (
        VariationalMLPEncoder, 
        VariationalLSTMEncoder,
        ModalityDecoder,
        UncertaintyWeightedFusion,
        ProxyCrossModalAttention,
        VAELossComputer
    )
except ImportError:
    # 独立文件夹运行时的导入
    from modules.encoder import MLPEncoder, LSTMEncoder
    from modules.variational_encoder import (
        VariationalMLPEncoder, 
        VariationalLSTMEncoder,
        ModalityDecoder,
        UncertaintyWeightedFusion,
        ProxyCrossModalAttention,
        VAELossComputer
    )


class AttentionRobustV2(nn.Module):
    """
    AttentionRobustV2 - 概率化多模态情感识别模型
    
    架构:
    1. 变分编码层: 将各模态特征编码为高斯分布 (μ, σ)
    2. 不确定性加权融合: 生成代理模态 proxy = Σ(w_i * μ_i), w_i ∝ 1/σ_i
    3. 代理模态跨模态注意力: 使用proxy引导各模态信息聚合
    4. 重建层: 从潜在变量重建原始特征 (训练时)
    5. 输出层: 情感分类/回归
    
    辅助损失 (通过interloss接口):
    - L_KL: KL散度正则化
    - L_recon: 重建损失
    - L_cross_KL: 跨模态KL散度
    """
    
    def __init__(self, args):
        super(AttentionRobustV2, self).__init__()
        
        # ==================== 基础参数 ====================
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
        
        # ==================== 代理模态参数 ====================
        self.use_proxy_attention = getattr(args, 'use_proxy_attention', True)
        self.fusion_temperature = getattr(args, 'fusion_temperature', 1.0)
        self.num_attention_heads = getattr(args, 'num_attention_heads', 4)
        
        # ==================== 模态Dropout参数 (保留原有功能) ====================
        self.modality_dropout = getattr(args, 'modality_dropout', 0.2)
        self.use_modality_dropout = getattr(args, 'use_modality_dropout', True)
        self.warmup_epochs = getattr(args, 'modality_dropout_warmup', 0)
        self.current_epoch = 0
        
        # ==================== 编码器 ====================
        if args.feat_type in ['utt']:
            if self.use_vae:
                # 变分编码器
                self.audio_encoder = VariationalMLPEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = VariationalMLPEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = VariationalMLPEncoder(video_dim, hidden_dim, dropout)
                
                # 解码器 (用于重建损失)
                self.audio_decoder = ModalityDecoder(hidden_dim, audio_dim, dropout)
                self.text_decoder = ModalityDecoder(hidden_dim, text_dim, dropout)
                self.video_decoder = ModalityDecoder(hidden_dim, video_dim, dropout)
            else:
                # 原始编码器 (向后兼容)
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
            # 不确定性加权融合
            self.uncertainty_fusion = UncertaintyWeightedFusion(
                hidden_dim, 
                temperature=self.fusion_temperature
            )
            
            # 代理模态跨模态注意力 (可选)
            if self.use_proxy_attention:
                self.proxy_attention = ProxyCrossModalAttention(
                    hidden_dim, 
                    num_heads=self.num_attention_heads, 
                    dropout=dropout
                )
            
            # 损失计算器
            self.loss_computer = VAELossComputer(
                kl_weight=self.kl_weight,
                recon_weight=self.recon_weight,
                cross_kl_weight=self.cross_kl_weight
            )
        else:
            # 原始attention融合 (向后兼容)
            self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)
            self.fc_att = nn.Linear(hidden_dim, 3)
        
        # ==================== 输出层 ====================
        self.feat_dropout = nn.Dropout(p=dropout)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
    
    def set_epoch(self, epoch):
        """设置当前epoch，用于渐进式模态dropout"""
        self.current_epoch = epoch
    
    def apply_modality_dropout(self, z_audio, z_text, z_video):
        """
        模态dropout - 训练时随机置零某些模态
        支持VAE模式和原始模式
        """
        if not self.training or not self.use_modality_dropout:
            return z_audio, z_text, z_video
        
        # 渐进式：前warmup_epochs个epoch不使用模态dropout
        if self.current_epoch < self.warmup_epochs:
            return z_audio, z_text, z_video
        
        # 计算有效dropout率
        if self.warmup_epochs > 0:
            progress = min(1.0, (self.current_epoch - self.warmup_epochs) / self.warmup_epochs)
            effective_dropout = self.modality_dropout * progress
        else:
            effective_dropout = self.modality_dropout
        
        batch_size = z_audio.size(0)
        device = z_audio.device
        
        # 创建mask
        masks = torch.ones(batch_size, 3, device=device)
        
        for i in range(batch_size):
            if torch.rand(1).item() < effective_dropout:
                # 随机选择丢弃模式 (6种)
                drop_mode = torch.randint(0, 6, (1,)).item()
                if drop_mode == 0:      # 丢弃音频
                    masks[i, 0] = 0
                elif drop_mode == 1:    # 丢弃文本
                    masks[i, 1] = 0
                elif drop_mode == 2:    # 丢弃视频
                    masks[i, 2] = 0
                elif drop_mode == 3:    # 丢弃音频+文本
                    masks[i, 0] = 0
                    masks[i, 1] = 0
                elif drop_mode == 4:    # 丢弃音频+视频
                    masks[i, 0] = 0
                    masks[i, 2] = 0
                elif drop_mode == 5:    # 丢弃文本+视频
                    masks[i, 1] = 0
                    masks[i, 2] = 0
        
        # 应用mask
        z_audio = z_audio * masks[:, 0:1]
        z_text = z_text * masks[:, 1:2]
        z_video = z_video * masks[:, 2:3]
        
        return z_audio, z_text, z_video
    
    def forward(self, batch):
        """
        前向传播
        根据use_vae参数选择VAE模式或原始模式
        """
        if self.use_vae:
            return self.forward_vae(batch)
        else:
            return self.forward_original(batch)
    
    def forward_vae(self, batch):
        """
        VAE模式的前向传播
        核心流程：变分编码 → 模态dropout → 不确定性加权融合 → 跨模态注意力 → 输出
        """
        audios = batch['audios']
        texts = batch['texts']
        videos = batch['videos']
        
        # ========== 1. 变分编码 ==========
        # 输出: z (采样), mu (均值), logvar (对数方差), std (标准差)
        z_a, mu_a, logvar_a, std_a = self.audio_encoder(audios)
        z_t, mu_t, logvar_t, std_t = self.text_encoder(texts)
        z_v, mu_v, logvar_v, std_v = self.video_encoder(videos)
        
        # ========== 2. 模态Dropout (可选) ==========
        z_a_dropped, z_t_dropped, z_v_dropped = self.apply_modality_dropout(z_a, z_t, z_v)
        
        # ========== 3. 不确定性加权融合 → 生成代理模态 ==========
        # 使用均值和标准差计算权重，权重 ∝ 1/σ
        proxy, weights = self.uncertainty_fusion(
            [mu_a, mu_t, mu_v], 
            [std_a, std_t, std_v]
        )
        
        # ========== 4. 代理模态跨模态注意力 (可选) ==========
        if self.use_proxy_attention:
            fused = self.proxy_attention(proxy, mu_a, mu_t, mu_v, weights)
        else:
            fused = proxy
        
        # ========== 5. 输出层 ==========
        features = self.feat_dropout(fused)
        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        
        # ========== 6. 计算辅助损失 (interloss) ==========
        if self.training:
            # 使用dropout后的z进行重建
            recon_a = self.audio_decoder(z_a_dropped)
            recon_t = self.text_decoder(z_t_dropped)
            recon_v = self.video_decoder(z_v_dropped)
            
            interloss = self.loss_computer.compute(
                mu_list=[mu_a, mu_t, mu_v],
                logvar_list=[logvar_a, logvar_t, logvar_v],
                originals=[audios, texts, videos],
                reconstructions=[recon_a, recon_t, recon_v]
            )
        else:
            interloss = torch.tensor(0.0, device=audios.device)
        
        return features, emos_out, vals_out, interloss
    
    def forward_original(self, batch):
        """
        原始模式的前向传播 (向后兼容)
        与AttentionRobust保持一致
        """
        audio_hidden = self.audio_encoder(batch['audios'])
        text_hidden = self.text_encoder(batch['texts'])
        video_hidden = self.video_encoder(batch['videos'])
        
        # 模态dropout
        audio_hidden, text_hidden, video_hidden = self.apply_modality_dropout(
            audio_hidden, text_hidden, video_hidden
        )
        
        # 拼接 + attention
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
        vals_out = self.fc_out_2(features)
        interloss = torch.tensor(0.0, device=batch['audios'].device)
        
        return features, emos_out, vals_out, interloss
