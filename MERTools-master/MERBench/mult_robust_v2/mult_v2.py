import torch
import torch.nn as nn
import torch.nn.functional as F
from toolkit.models.modules.transformers_encoder.transformer import TransformerEncoder
import random

class MULTRobustV2(nn.Module):
    def __init__(self, args):
        super(MULTRobustV2, self).__init__()

        # params: analyze args
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2

        # params: analyze args
        self.attn_mask = True
        self.layers = args.layers # 4 
        self.dropout = args.dropout
        self.num_heads = args.num_heads # 8
        self.hidden_dim = args.hidden_dim # 128
        self.conv1d_kernel_size = args.conv1d_kernel_size # 5
        self.grad_clip = args.grad_clip
        
        # params: intermedia
        combined_dim = 2 * (self.hidden_dim + self.hidden_dim + self.hidden_dim)
        output_dim = self.hidden_dim // 2
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(text_dim,  self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_a = nn.Conv1d(audio_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_v = nn.Conv1d(video_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        
        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
    
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # cls layers
        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)

        # ==================== Robust V2 Features ====================
        # 1. Dynamic Gated Fusion (replaces static modality_weights)
        self.gating_network = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 3) # Output 3 weights for L, A, V
        )
        
        # 2. Emotion-Valence Prior Fusion
        self.valence_gate = nn.Sequential(
            nn.Linear(output_dim1 + output_dim, output_dim),
            nn.Sigmoid()
        )
        self.valence_proj = nn.Linear(output_dim, output_dim)

        # 3. LayerNorms for stability
        self.ln_l = nn.LayerNorm(self.hidden_dim)
        self.ln_a = nn.LayerNorm(self.hidden_dim)
        self.ln_v = nn.LayerNorm(self.hidden_dim)
        self.ln_final = nn.LayerNorm(combined_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask)

    def apply_modality_dropout(self, x_l, x_a, x_v):
        """
        Randomly drop modalities during training to prevent over-reliance on text.
        Text is dropped more frequently than audio/video.
        """
        if not self.training:
            return x_l, x_a, x_v

        # Probabilities of dropping each modality
        p_drop_l = 0.3  # Drop text 30% of the time
        p_drop_a = 0.1  # Drop audio 10% of the time
        p_drop_v = 0.1  # Drop video 10% of the time

        if random.random() < p_drop_l:
            x_l = torch.zeros_like(x_l)
        if random.random() < p_drop_a:
            x_a = torch.zeros_like(x_a)
        if random.random() < p_drop_v:
            x_v = torch.zeros_like(x_v)

        return x_l, x_a, x_v

    def apply_feature_noise(self, x, std=0.05):
        """
        Inject Gaussian noise to features during training to prevent overfitting.
        """
        if self.training:
            noise = torch.randn_like(x) * std
            return x + noise
        return x

    def forward(self, batch):
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            video_feat: tensor of shape (batch, seqlen2, video_in)
            text_feat:  tensor of shape (batch, seqlen3, text_in)
        '''
        x_l = batch['texts'].transpose(1, 2)
        x_a = batch['audios'].transpose(1, 2)
        x_v = batch['videos'].transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        # Apply LayerNorm
        proj_x_l = self.ln_l(proj_x_l)
        proj_x_a = self.ln_a(proj_x_a)
        proj_x_v = self.ln_v(proj_x_v)

        # Apply Feature Noise
        proj_x_l = self.apply_feature_noise(proj_x_l, std=0.05)
        proj_x_a = self.apply_feature_noise(proj_x_a, std=0.05)
        proj_x_v = self.apply_feature_noise(proj_x_v, std=0.05)

        # Apply Modality Dropout
        proj_x_l, proj_x_a, proj_x_v = self.apply_modality_dropout(proj_x_l, proj_x_a, proj_x_v)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) 
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as[-1]
        
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]
        
        # Apply Dynamic Gated Fusion
        # 1. Concatenate the features to form a joint representation
        concat_features = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # 2. Pass through the gating network to get raw logits
        gate_logits = self.gating_network(concat_features)
        
        # 3. Apply Softmax to get normalized weights (sum to 1)
        gate_weights = F.softmax(gate_logits, dim=-1) # Shape: (batch_size, 3)
        
        # 4. Multiply each modality by its dynamic weight
        # We multiply by 3 to keep the overall scale similar to before (average weight = 1)
        last_h_l = last_h_l * gate_weights[:, 0].unsqueeze(1) * 3.0
        last_h_a = last_h_a * gate_weights[:, 1].unsqueeze(1) * 3.0
        last_h_v = last_h_v * gate_weights[:, 2].unsqueeze(1) * 3.0

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        last_hs = self.ln_final(last_hs)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        features = self.out_layer(last_hs_proj)

        # store results
        emos_out  = self.fc_out_1(features)
        
        # Emotion-Valence Prior Fusion
        # Use emotion logits to gate valence features
        gate_input = torch.cat([emos_out.detach(), features], dim=-1) # Detach to prevent valence loss from affecting emotion too much
        gate = self.valence_gate(gate_input)
        val_features = self.valence_proj(features) * gate
        vals_out  = self.fc_out_2(val_features)

        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss
