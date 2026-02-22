import sys
import os
sys.path.append(os.path.abspath('/root/autodl-tmp/MERTools-master/MERBench'))
import torch
from mult_robust_v3.mult_v3 import MULTRobustV3

class Args:
    pass

args = Args()
args.audio_dim = 10
args.text_dim = 10
args.video_dim = 10
args.output_dim1 = 4
args.output_dim2 = 1
args.layers = 4
args.dropout = 0.1
args.num_heads = 8
args.hidden_dim = 128
args.conv1d_kernel_size = 5
args.grad_clip = -1

model = MULTRobustV3(args)
batch = {
    'texts': torch.randn(2, 50, 10),
    'audios': torch.randn(2, 50, 10),
    'videos': torch.randn(2, 50, 10)
}
out = model(batch)
print('interloss:', out[3].item())
print('Success!')
