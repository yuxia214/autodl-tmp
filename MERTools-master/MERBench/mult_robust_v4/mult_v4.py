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
        # Use 1e-6 instead of 1e-8 to prevent large gradients (0.5 / sqrt(1e-6) = 500)
        sqrt = (summed + 1e-6) ** (0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

class MULTRobustV4(nn.Module):
    def __init__(self, args):
        super(MULTRobustV4, self).__init__()

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

        # ==================== Robust V4 Features ====================
        # 1. Missing-Aware Prompts
        self.missing_prompt_l = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.missing_prompt_a = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.missing_prompt_v = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.missing_prompt_l, std=0.02)
        nn.init.trunc_normal_(self.missing_prompt_a, std=0.02)
        nn.init.trunc_normal_(self.missing_prompt_v, std=0.02)

        # 2. Lightweight Mixture of Experts (MoE) for Fusion
        self.num_experts = 4
        self.router = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_experts)
        )
        # Initialize router weights to be small to ensure uniform routing at the start
        nn.init.normal_(self.router[-1].weight, std=0.01)
        nn.init.zeros_(self.router[-1].bias)
        
        # Bottleneck Experts to prevent overfitting on small datasets
        expert_hidden_dim = combined_dim // 4
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(expert_hidden_dim, combined_dim)
            ) for _ in range(self.num_experts)
        ])
        # ============================================================
        
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
        self.ln_concat = nn.LayerNorm(combined_dim) # Added for MoE input stability
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

    def apply_modality_dropout(self, proj_x_l, proj_x_a, proj_x_v, orig_x_l, orig_x_a, orig_x_v):
        """
        1. Detects naturally missing modalities (e.g., in Test1) using original inputs and replaces them with Prompts.
        2. Randomly drops modalities during training (per-sample) to prevent over-reliance.
        """
        seq_len_l, batch_size, _ = proj_x_l.size()
        seq_len_a, _, _ = proj_x_a.size()
        seq_len_v, _, _ = proj_x_v.size()

        # 1. Detect naturally missing modalities from ORIGINAL inputs (energy close to 0)
        # orig_x_l shape: (batch_size, feature_dim, seq_len)
        energy_l = torch.sum(torch.abs(orig_x_l), dim=(1, 2))
        energy_a = torch.sum(torch.abs(orig_x_a), dim=(1, 2))
        energy_v = torch.sum(torch.abs(orig_x_v), dim=(1, 2))
        
        mask_l = (energy_l < 1e-4).view(1, batch_size, 1)
        mask_a = (energy_a < 1e-4).view(1, batch_size, 1)
        mask_v = (energy_v < 1e-4).view(1, batch_size, 1)

        # 2. Random dropout during training (per-sample)
        if self.training:
            p_drop_l = 0.15
            p_drop_a = 0.05
            p_drop_v = 0.05
            
            drop_l = (torch.rand(batch_size, device=proj_x_l.device) < p_drop_l).view(1, batch_size, 1)
            drop_a = (torch.rand(batch_size, device=proj_x_a.device) < p_drop_a).view(1, batch_size, 1)
            drop_v = (torch.rand(batch_size, device=proj_x_v.device) < p_drop_v).view(1, batch_size, 1)
            
            mask_l = mask_l | drop_l
            mask_a = mask_a | drop_a
            mask_v = mask_v | drop_v

        # Replace missing/dropped modalities with Prompts
        proj_x_l = torch.where(mask_l, self.missing_prompt_l.expand(seq_len_l, batch_size, -1), proj_x_l)
        proj_x_a = torch.where(mask_a, self.missing_prompt_a.expand(seq_len_a, batch_size, -1), proj_x_a)
        proj_x_v = torch.where(mask_v, self.missing_prompt_v.expand(seq_len_v, batch_size, -1), proj_x_v)

        return proj_x_l, proj_x_a, proj_x_v

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

        # Apply Modality Dropout (Pass original inputs for accurate energy detection)
        proj_x_l, proj_x_a, proj_x_v = self.apply_modality_dropout(
            proj_x_l, proj_x_a, proj_x_v, 
            x_l, x_a, x_v
        )

        # ==================== Feature Factorization (MISA) ====================
        # Extract Shared Features
        shared_x_l = self.shared_l(proj_x_l)
        shared_x_a = self.shared_a(proj_x_a)
        shared_x_v = self.shared_v(proj_x_v)

        # Extract Private Features
        private_x_l = self.private_l(proj_x_l)
        private_x_a = self.private_a(proj_x_a)
        private_x_v = self.private_v(proj_x_v)

        # Pool features for loss calculation (Mean pooling across sequence length)
        shared_l_pooled = torch.mean(shared_x_l, dim=0)
        shared_a_pooled = torch.mean(shared_x_a, dim=0)
        shared_v_pooled = torch.mean(shared_x_v, dim=0)
        
        private_l_pooled = torch.mean(private_x_l, dim=0)
        private_a_pooled = torch.mean(private_x_a, dim=0)
        private_v_pooled = torch.mean(private_x_v, dim=0)

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
        
        # Apply Lightweight MoE Fusion on all 6 features (3 shared fused + 3 private)
        concat_features = torch.cat([last_h_l, last_h_a, last_h_v, private_l_pooled, private_a_pooled, private_v_pooled], dim=1)
        
        # Apply LayerNorm before MoE to ensure shared and private features are on the same scale
        concat_features = self.ln_concat(concat_features)
        
        # 1. Router calculates weights for each expert
        router_logits = self.router(concat_features)
        router_weights = F.softmax(router_logits, dim=-1) # Shape: (batch_size, num_experts)
        
        # 2. Experts process the concatenated features
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(concat_features).unsqueeze(1)) # Shape: (batch_size, 1, combined_dim)
            
        expert_outputs = torch.cat(expert_outputs, dim=1) # Shape: (batch_size, num_experts, combined_dim)
        
        # 3. Weighted sum of expert outputs
        last_hs = torch.sum(expert_outputs * router_weights.unsqueeze(-1), dim=1) # Shape: (batch_size, combined_dim)
        
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
