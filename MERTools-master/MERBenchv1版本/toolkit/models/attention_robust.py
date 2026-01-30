'''
Description: Attention model with modality dropout for robustness against missing modalities
Improved version for better performance on test2 (missing modality test)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.encoder import MLPEncoder, LSTMEncoder


class AttentionRobust(nn.Module):
    def __init__(self, args):
        super(AttentionRobust, self).__init__()
        
        text_dim    = args.text_dim
        audio_dim   = args.audio_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip
        
        # 模态dropout概率 - 训练时随机丢弃模态以增强鲁棒性
        self.modality_dropout = getattr(args, 'modality_dropout', 0.2)
        # 是否使用训练模式的模态dropout
        self.use_modality_dropout = getattr(args, 'use_modality_dropout', True)
        # 渐进式模态dropout：前N个epoch不使用模态dropout
        self.warmup_epochs = getattr(args, 'modality_dropout_warmup', 0)
        self.current_epoch = 0

        if args.feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)

        self.fc_att   = nn.Linear(hidden_dim, 3)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
        
        # 额外的正则化层
        self.feat_dropout = nn.Dropout(p=dropout)
    
    def set_epoch(self, epoch):
        """设置当前epoch，用于渐进式模态dropout"""
        self.current_epoch = epoch
    
    def apply_modality_dropout(self, audio_hidden, text_hidden, video_hidden):
        """
        训练时随机将某些模态置零，模拟模态缺失的情况
        这有助于模型学习更鲁棒的表示
        """
        if not self.training or not self.use_modality_dropout:
            return audio_hidden, text_hidden, video_hidden
        
        # 渐进式：前warmup_epochs个epoch不使用模态dropout
        if self.current_epoch < self.warmup_epochs:
            return audio_hidden, text_hidden, video_hidden
        
        # 渐进式增加模态dropout概率
        if self.warmup_epochs > 0:
            progress = min(1.0, (self.current_epoch - self.warmup_epochs) / self.warmup_epochs)
            effective_dropout = self.modality_dropout * progress
        else:
            effective_dropout = self.modality_dropout
        
        batch_size = audio_hidden.size(0)
        device = audio_hidden.device
        
        # 创建mask来避免in-place操作
        audio_mask = torch.ones(batch_size, 1, device=device)
        text_mask = torch.ones(batch_size, 1, device=device)
        video_mask = torch.ones(batch_size, 1, device=device)
        
        # 为每个样本随机决定丢弃哪些模态
        # 策略：有一定概率丢弃1个或2个模态（不会全部丢弃）
        for i in range(batch_size):
            # 随机决定是否应用模态dropout
            if torch.rand(1).item() < effective_dropout:
                # 随机选择丢弃模式
                drop_mode = torch.randint(0, 6, (1,)).item()  # 6种丢弃模式
                
                if drop_mode == 0:  # 丢弃音频
                    audio_mask[i] = 0
                elif drop_mode == 1:  # 丢弃文本
                    text_mask[i] = 0
                elif drop_mode == 2:  # 丢弃视频
                    video_mask[i] = 0
                elif drop_mode == 3:  # 丢弃音频+文本
                    audio_mask[i] = 0
                    text_mask[i] = 0
                elif drop_mode == 4:  # 丢弃音频+视频
                    audio_mask[i] = 0
                    video_mask[i] = 0
                elif drop_mode == 5:  # 丢弃文本+视频
                    text_mask[i] = 0
                    video_mask[i] = 0
        
        # 使用mask进行乘法操作，避免in-place修改
        audio_hidden = audio_hidden * audio_mask
        text_hidden = text_hidden * text_mask
        video_hidden = video_hidden * video_mask
        
        return audio_hidden, text_hidden, video_hidden
    
    def forward(self, batch):
        '''
            support feat_type: utt | frm-align | frm-unalign
        '''
        audio_hidden = self.audio_encoder(batch['audios']) # [32, 128]
        text_hidden  = self.text_encoder(batch['texts'])   # [32, 128]
        video_hidden = self.video_encoder(batch['videos']) # [32, 128]
        
        # 应用模态dropout（仅训练时）
        audio_hidden, text_hidden, video_hidden = self.apply_modality_dropout(
            audio_hidden, text_hidden, video_hidden
        )

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = F.softmax(attention, dim=1)  # 使用softmax使注意力权重和为1
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)  # [32, 128, 3] * [32, 3, 1] = [32, 128, 1]

        features  = fused_feat.squeeze(axis=2) # [32, 128]
        features  = self.feat_dropout(features)  # 额外的dropout
        
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss
