import torch
import torch.nn as nn
import torch.nn.functional as F
from toolkit.models.modules.transformers_encoder.transformer import TransformerEncoder
import random

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1 = input1 - torch.mean(input1, dim=0, keepdim=True)
        input2 = input2 - torch.mean(input2, dim=0, keepdim=True)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss

class CMD(nn.Module):
    """
    Central Moment Discrepancy (CMD)
    Objective: minimize the discrepancy between the central moments of two representations.
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = (summed + 1e-8) ** (0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

class MULTRobustV3(nn.Module):
    def __init__(self, args):
        super(MULTRobustV3, self).__init__()

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
        # Now we have 3 shared features (from MULT, each 2*hidden_dim) + 3 private features (each hidden_dim)
        # Total dim = 2*128*3 + 128*3 = 9 * hidden_dim
        combined_dim = 9 * self.hidden_dim
        output_dim = self.hidden_dim // 2
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(text_dim,  self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_a = nn.Conv1d(audio_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_v = nn.Conv1d(video_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        
        # ==================== Robust V3 Features (MISA) ====================
        # Shared Projectors
        self.shared_l = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.shared_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.shared_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Private Projectors
        self.private_l = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.private_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.private_v = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.loss_diff = DiffLoss()
        self.loss_cmd = CMD()
        # ===================================================================

        # 2. Crossmodal Attentions (Only for Shared Features)
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

        # Dynamic Gated Fusion (for the 6 features: 3 shared fused, 3 private)
        self.gating_network = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 6) # Output 6 weights
        )
        
        # Emotion-Valence Prior Fusion
        self.valence_gate = nn.Sequential(
            nn.Linear(output_dim1 + output_dim, output_dim),
            nn.Sigmoid()
        )
        self.valence_proj = nn.Linear(output_dim, output_dim)

        # LayerNorms for stability
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
        Reduced dropout rate for V3 since we have feature factorization.
        """
        if not self.training:
            return x_l, x_a, x_v

        # Probabilities of dropping each modality (reduced for V3)
        p_drop_l = 0.15  # Drop text 15% of the time
        p_drop_a = 0.05  # Drop audio 5% of the time
        p_drop_v = 0.05  # Drop video 5% of the time

        if random.random() < p_drop_l:
            x_l = torch.zeros_like(x_l)
        if random.random() < p_drop_a:
            x_a = torch.zeros_like(x_a)
        if random.random() < p_drop_v:
            x_v = torch.zeros_like(x_v)

        return x_l, x_a, x_v

    def apply_feature_noise(self, x, std=0.05):
        if self.training:
            noise = torch.randn_like(x) * std
            return x + noise
        return x

    def forward(self, batch):
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

        # ==================== Feature Factorization (MISA) ====================
        # Extract Shared Features
        shared_x_l = self.shared_l(proj_x_l)
        shared_x_a = self.shared_a(proj_x_a)
        shared_x_v = self.shared_v(proj_x_v)

        # Extract Private Features
        private_x_l = self.private_l(proj_x_l)
        private_x_a = self.private_a(proj_x_a)
        private_x_v = self.private_v(proj_x_v)

        # Pool features for loss calculation (take the last token)
        shared_l_pooled = shared_x_l[-1]
        shared_a_pooled = shared_x_a[-1]
        shared_v_pooled = shared_x_v[-1]
        
        private_l_pooled = private_x_l[-1]
        private_a_pooled = private_x_a[-1]
        private_v_pooled = private_x_v[-1]

        # Calculate DiffLoss (Orthogonality between shared and private)
        diff_loss = self.loss_diff(shared_l_pooled, private_l_pooled) + \
                    self.loss_diff(shared_a_pooled, private_a_pooled) + \
                    self.loss_diff(shared_v_pooled, private_v_pooled)

        # Calculate SimLoss (Similarity between shared features of different modalities)
        # Use n_moments=2 to prevent gradient explosion/NaNs from high-order moments
        sim_loss = self.loss_cmd(shared_l_pooled, shared_a_pooled, 2) + \
                   self.loss_cmd(shared_l_pooled, shared_v_pooled, 2) + \
                   self.loss_cmd(shared_a_pooled, shared_v_pooled, 2)

        # Total interloss (weighted)
        interloss = 0.1 * diff_loss + 0.1 * sim_loss
        # ======================================================================

        # MULT Cross-Attention (Only on Shared Features)
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(shared_x_l, shared_x_a, shared_x_a) 
        h_l_with_vs = self.trans_l_with_v(shared_x_l, shared_x_v, shared_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(shared_x_a, shared_x_l, shared_x_l)
        h_a_with_vs = self.trans_a_with_v(shared_x_a, shared_x_v, shared_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as[-1]
        
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(shared_x_v, shared_x_l, shared_x_l)
        h_v_with_as = self.trans_v_with_a(shared_x_v, shared_x_a, shared_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]
        
        # Apply Dynamic Gated Fusion on all 6 features (3 shared fused + 3 private)
        concat_features = torch.cat([last_h_l, last_h_a, last_h_v, private_l_pooled, private_a_pooled, private_v_pooled], dim=1)
        
        gate_logits = self.gating_network(concat_features)
        gate_weights = F.softmax(gate_logits, dim=-1) # Shape: (batch_size, 6)
        
        # Multiply each feature by its dynamic weight (multiply by 6.0 to keep scale)
        last_h_l = last_h_l * gate_weights[:, 0].unsqueeze(1) * 6.0
        last_h_a = last_h_a * gate_weights[:, 1].unsqueeze(1) * 6.0
        last_h_v = last_h_v * gate_weights[:, 2].unsqueeze(1) * 6.0
        private_l_pooled = private_l_pooled * gate_weights[:, 3].unsqueeze(1) * 6.0
        private_a_pooled = private_a_pooled * gate_weights[:, 4].unsqueeze(1) * 6.0
        private_v_pooled = private_v_pooled * gate_weights[:, 5].unsqueeze(1) * 6.0

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, private_l_pooled, private_a_pooled, private_v_pooled], dim=1)
        last_hs = self.ln_final(last_hs)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        features = self.out_layer(last_hs_proj)

        # store results
        emos_out  = self.fc_out_1(features)
        
        # Emotion-Valence Prior Fusion
        gate_input = torch.cat([emos_out.detach(), features], dim=-1)
        gate = self.valence_gate(gate_input)
        val_features = self.valence_proj(features) * gate
        vals_out  = self.fc_out_2(val_features)

        return features, emos_out, vals_out, interloss
