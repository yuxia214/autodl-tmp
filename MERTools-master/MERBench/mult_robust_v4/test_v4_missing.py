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
model.eval() # Set to eval to test natural missing detection

# Create a batch where the second sample has missing audio (all zeros)
batch = {
    'texts': torch.randn(2, 10, 5120),
    'audios': torch.randn(2, 20, 1024),
    'videos': torch.randn(2, 30, 768)
}
batch['audios'][1] = 0.0 # Simulate missing audio for sample 1

features, emos_out, vals_out, interloss = model(batch)
print("Forward pass with missing modality successful!")
