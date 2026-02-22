import sys
sys.path.append('/root/autodl-tmp/MERTools-master/MERBench')
from mult_robust_v4.mult_v4 import MULTRobustV4
import argparse
import torch

args = argparse.Namespace(
    audio_dim=1024, text_dim=5120, video_dim=768, 
    output_dim1=7, output_dim2=1, layers=4, dropout=0.2, 
    num_heads=8, hidden_dim=128, conv1d_kernel_size=5, grad_clip=0.6
)
model = MULTRobustV4(args)

batch = {
    'texts': torch.randn(2, 10, 5120),
    'audios': torch.randn(2, 20, 1024),
    'videos': torch.randn(2, 30, 768)
}
features, emos_out, vals_out, interloss = model(batch)
print("Forward pass successful!")
print("Features shape:", features.shape)
print("Emos out shape:", emos_out.shape)
