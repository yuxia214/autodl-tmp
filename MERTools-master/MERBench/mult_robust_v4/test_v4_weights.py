import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import argparse
sys.path.append('/root/autodl-tmp/MERTools-master/MERBench/mult_robust_v4')
from mult_v4 import MULTRobustV4 as MULT
from run_mult_v4 import train_or_eval_model, gain_metric_from_results
from toolkit.dataloader import get_dataloaders
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_best_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MER2023')
    parser.add_argument('--train_dataset', type=str, default=None)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--save_root', type=str, default='./saved')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--savemodel', action='store_true', default=False)
    parser.add_argument('--save_iters', type=int, default=1e8)
    parser.add_argument('--audio_feature', type=str, default='chinese-hubert-large-FRA')
    parser.add_argument('--text_feature', type=str, default='Baichuan-13B-Base-FRA')
    parser.add_argument('--video_feature', type=str, default='clip-vit-large-patch14-FRA')
    parser.add_argument('--feat_type', type=str, default='frm_align')
    parser.add_argument('--feat_scale', type=int, default=6)
    parser.add_argument('--e2e_name', type=str, default=None)
    parser.add_argument('--e2e_dim', type=int, default=None)
    parser.add_argument('--n_classes', type=int, default=None)
    parser.add_argument('--hyper_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='mult')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--lr_adjust', type=str, default='case1')
    parser.add_argument('--print_iters', type=float, default=1e8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--conv1d_kernel_size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--metric_name', type=str, default='emoval')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("Loading Test Datasets...")
    dataloader_class = get_dataloaders(args)
    train_loaders, eval_loaders, test_loaders = dataloader_class.get_loaders()
    
    # Set dimensions from dataloader class
    args.audio_dim = 1024
    args.text_dim = 5120
    args.video_dim = 768
        
    print("Building Model...")
    model = MULT(args).cuda()
    reg_loss = nn.MSELoss().cuda()
    cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    
    # We will test the best model from folder 2 (which had the highest eval score 0.5953)
    best_model_path = '/root/autodl-tmp/MERTools-master/MERBench/best_model_v4_folder_2.pt'
    print(f"Loading weights from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    print("\n====== Testing Results (Folder 2 Best Model) ======")
    for jj, test_loader in enumerate(test_loaders):
        test_results = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, epoch=0, optimizer=None, train=False, dataloader_class=dataloader_class)
        f1_score = test_results.get('emofscore', 0)
        emoval_score = test_results.get('emoval', 0)
        if emoval_score == 0 and 'valmse' in test_results:
            emoval_score = test_results['emofscore'] - test_results['valmse'] * 0.25
        print(f"Test{jj+1} -> F1: {f1_score:.4f} | Emoval: {emoval_score:.4f} | ValMSE: {test_results.get('valmse', 0):.4f}")

if __name__ == '__main__':
    test_best_model()
