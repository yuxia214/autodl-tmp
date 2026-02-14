"""
改进版训练脚本 - 针对模态缺失场景(test2)优化
主要改进:
1. 使用AttentionRobust模型 (带模态dropout)
2. 添加早停机制 (Early Stopping)
3. 添加学习率调度器 (ReduceLROnPlateau)
4. 增强正则化 (更高的dropout和L2)
5. 固定超参数 (避免随机选择导致不稳定)

支持的模型:
- attention_robust: 带模态dropout的attention模型
- attention_robust_v2: 基于P-RMF的概率化多模态融合模型 (VAE + 不确定性加权 + 代理模态)

使用方法 (V1):
python -u main-robust.py --model='attention_robust' --feat_type='utt' --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --gpu=0

使用方法 (V2 - 推荐):
python -u main-robust.py --model='attention_robust_v2' --feat_type='utt' --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --hidden_dim=128 --dropout=0.35 \
    --use_vae --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --lr=5e-4 --l2=5e-5 --epochs=100 --early_stopping_patience=30 \
    --gpu=0
"""

import os
import time
import argparse
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from toolkit.utils.loss import *
from toolkit.utils.metric import *
from toolkit.utils.functions import *
from toolkit.models import get_models
from toolkit.dataloader import get_dataloaders


class EarlyStopping:
    """早停机制，防止过拟合"""
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, dataloader_class, epoch, optimizer=None, train=False):
    
    vidnames = []
    val_preds, val_labels = [], []
    emo_probs, emo_labels = [], []
    losses = []

    assert not train or optimizer!=None
    config.train = train
    if train:
        model.train()
    else:
        model.eval()
    
    for iter, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()
        
        batch, emos, vals, bnames = data
        vidnames += bnames
        for key in batch: batch[key] = batch[key].cuda()
        emos = emos.cuda()
        vals = vals.cuda()

        features, emos_out, vals_out, interloss = model(batch)

        loss = interloss
        if args.output_dim1 != 0:
            loss = loss + args.emo_loss_weight * cls_loss(emos_out, emos)
            emo_probs.append(emos_out.data.cpu().numpy())
            emo_labels.append(emos.data.cpu().numpy())
        if args.output_dim2 != 0: 
            loss = loss + args.val_loss_weight * reg_loss(vals_out, vals)
            val_preds.append(vals_out.data.cpu().numpy())
            val_labels.append(vals.data.cpu().numpy())
        losses.append(loss.data.cpu().numpy())
        
        if train:
            loss.backward()
            if model.model.grad_clip != -1:
                torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], model.model.grad_clip)
            optimizer.step()
        
        if (iter+1) % args.print_iters == 0:
            print (f'process on {iter+1}|{len(dataloader)}, meanloss: {np.mean(losses)}')

    if emo_probs  != []: emo_probs  = np.concatenate(emo_probs)
    if emo_labels != []: emo_labels = np.concatenate(emo_labels)
    if val_preds  != []: val_preds  = np.concatenate(val_preds)
    if val_labels != []: val_labels = np.concatenate(val_labels)
    results, _ = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
    save_results = dict(
        names = vidnames,
        loss  = np.mean(losses),
        **results,
    )
    return save_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Params for datasets
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--train_dataset', type=str, default=None, help='train dataset')
    parser.add_argument('--test_dataset',  type=str, default=None, help='test dataset')
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--savemodel', action='store_true', default=False, help='whether to save model')
    parser.add_argument('--save_iters', type=int, default=1e8, help='save models per iters')

    # Params for feature inputs
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature',  type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--feat_type',  type=str, default=None, help='feature type [utt, frm_align, frm_unalign]')
    parser.add_argument('--feat_scale', type=int, default=None, help='pre-compress input')
    parser.add_argument('--e2e_name', type=str, default=None, help='e2e pretrained model names')
    parser.add_argument('--e2e_dim',  type=int, default=None, help='e2e pretrained model hidden size')

    # Params for model
    parser.add_argument('--n_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--hyper_path', type=str, default=None, help='path to fixed hyperparams')
    parser.add_argument('--model', type=str, default='attention_robust', help='model name')

    # Params for training - 优化后的默认参数
    parser.add_argument('--lr', type=float, default=5e-4, metavar='lr', help='learning rate')
    parser.add_argument('--lr_adjust', type=str, default='case1', help='lr adjustment strategy')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2', help='L2 regularization weight (increased)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--print_iters', type=int, default=1e8, help='print per-iteration')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--emo_loss_weight', type=float, default=1.0, help='classification loss weight')
    parser.add_argument('--val_loss_weight', type=float, default=1.0, help='regression loss weight')
    parser.add_argument('--reg_loss_type', type=str, default='mse', choices=['mse', 'smoothl1'], help='regression loss type')
    parser.add_argument('--huber_beta', type=float, default=1.0, help='beta for smoothl1 regression loss')
    
    # 新增参数 - 针对模态缺失优化
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (increased for regularization)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping value')
    parser.add_argument('--modality_dropout', type=float, default=0.3, help='modality dropout rate for robustness')
    parser.add_argument('--use_modality_dropout', action='store_true', default=True, help='whether to use modality dropout')
    parser.add_argument('--no_modality_dropout', action='store_false', dest='use_modality_dropout', help='disable modality dropout')
    parser.add_argument('--modality_dropout_warmup', type=int, default=0, help='warmup epochs before applying modality dropout')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=10, help='lr scheduler patience')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='lr reduction factor')
    
    # AttentionRobustV2专用参数 - VAE + 代理模态
    parser.add_argument('--use_vae', action='store_true', default=True, help='whether to use VAE encoder')
    parser.add_argument('--no_vae', action='store_false', dest='use_vae', help='disable VAE encoder')
    parser.add_argument('--kl_weight', type=float, default=0.01, help='KL divergence loss weight')
    parser.add_argument('--recon_weight', type=float, default=0.1, help='reconstruction loss weight')
    parser.add_argument('--cross_kl_weight', type=float, default=0.01, help='cross-modal KL loss weight')
    parser.add_argument('--use_proxy_attention', action='store_true', default=True, help='whether to use proxy cross-modal attention')
    parser.add_argument('--no_proxy_attention', action='store_false', dest='use_proxy_attention', help='disable proxy attention')
    parser.add_argument('--fusion_temperature', type=float, default=1.0, help='temperature for uncertainty weighted fusion')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='number of attention heads for proxy attention')

    # V4新增参数: 对比学习
    parser.add_argument('--use_contrastive', action='store_true', default=True, help='whether to use contrastive learning')
    parser.add_argument('--no_contrastive', action='store_false', dest='use_contrastive', help='disable contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='contrastive loss weight')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07, help='temperature for InfoNCE loss')

    # V4新增参数: 门控融合
    parser.add_argument('--use_gated_fusion', action='store_true', default=True, help='whether to use gated uncertainty fusion')
    parser.add_argument('--no_gated_fusion', action='store_false', dest='use_gated_fusion', help='disable gated fusion')
    parser.add_argument('--gate_alpha', type=float, default=0.5, help='balance between uncertainty and gate weights')

    # V4新增参数: Focal Loss
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor')

    # V5新增参数: 深度编码器 + Mixup + 动态KL
    parser.add_argument('--use_mixup', action='store_true', default=False, help='whether to use mixup augmentation')
    parser.add_argument('--no_mixup', action='store_false', dest='use_mixup', help='disable mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.4, help='mixup alpha parameter')
    parser.add_argument('--use_dynamic_kl', action='store_true', default=True, help='whether to use dynamic KL scheduling')
    parser.add_argument('--no_dynamic_kl', action='store_false', dest='use_dynamic_kl', help='disable dynamic KL')
    parser.add_argument('--kl_warmup_epochs', type=int, default=20, help='KL warmup epochs')

    # V7新增参数: Emotion-Valence一致性 + 噪声增强
    parser.add_argument('--use_valence_prior', action='store_true', default=True, help='whether to use emotion-guided valence prior')
    parser.add_argument('--no_valence_prior', action='store_false', dest='use_valence_prior', help='disable emotion-guided valence prior')
    parser.add_argument('--valence_consistency_weight', type=float, default=0.08, help='weight of valence consistency regularization')
    parser.add_argument('--valence_center_reg_weight', type=float, default=0.005, help='weight of emotion-valence center regularization')
    parser.add_argument('--feature_noise_std', type=float, default=0.02, help='std of feature noise augmentation')
    parser.add_argument('--feature_noise_prob', type=float, default=0.3, help='probability of applying feature noise')
    parser.add_argument('--feature_noise_warmup', type=int, default=10, help='warmup epochs before noise augmentation')

    # V8新增参数: 双路径融合 + 可靠度建模
    parser.add_argument('--use_gated_uncertainty', action='store_true', default=True, help='whether to use gated uncertainty fusion')
    parser.add_argument('--no_gated_uncertainty', action='store_false', dest='use_gated_uncertainty', help='disable gated uncertainty fusion')
    parser.add_argument('--fusion_residual_scale', type=float, default=0.4, help='residual branch contribution in dual-path fusion')
    parser.add_argument('--reliability_temperature', type=float, default=1.0, help='temperature for reliability weighting')
    parser.add_argument('--modality_agreement_weight', type=float, default=0.01, help='weight of modality agreement regularization')
    parser.add_argument('--weight_consistency_weight', type=float, default=0.02, help='weight of reliability/fusion weight consistency')
    parser.add_argument('--quality_weight', type=float, default=0.6, help='quality logit weight for quality-aware fusion')
    parser.add_argument('--impute_loss_weight', type=float, default=0.10, help='weight of cross-modal imputation loss')
    parser.add_argument('--consistency_emo_weight', type=float, default=0.08, help='weight of teacher-student emotion consistency')
    parser.add_argument('--consistency_val_weight', type=float, default=0.05, help='weight of teacher-student valence consistency')
    parser.add_argument('--corruption_max_rate', type=float, default=0.45, help='max modality corruption rate for training')
    parser.add_argument('--corruption_warmup_epochs', type=int, default=25, help='warmup epochs to reach corruption_max_rate')
    parser.add_argument('--double_mask_ratio', type=float, default=0.35, help='ratio of double-modality masking among corrupted samples')
    parser.add_argument('--latent_noise_std', type=float, default=0.02, help='std of latent noise for student branch')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)


    print ('====== Params Pre-analysis =======')
    if args.feat_type == 'utt':
        args.feat_scale = 1
    elif args.feat_type == 'frm_align':
        assert args.audio_feature.endswith('FRA')
        assert args.text_feature.endswith('FRA')
        assert args.video_feature.endswith('FRA')
        args.feat_scale = 6
    elif args.feat_type == 'frm_unalign':
        assert args.audio_feature.endswith('FRA')
        assert args.text_feature.endswith('FRA')
        assert args.video_feature.endswith('FRA')
        args.feat_scale = 12

    ## define store folder
    if args.train_dataset is not None:
        args.save_root = f'{args.save_root}-cross'
    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    whole_features = [item for item in whole_features if item is not None]
    if len(set(whole_features)) == 0:
        args.save_root = f'{args.save_root}-others'
    elif len(set(whole_features)) == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif len(set(whole_features)) == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif len(set(whole_features)) == 3:
        args.save_root = f'{args.save_root}-trimodal'

    config.dataset = args.dataset
    print('args: ', args)

    ## save root
    save_resroot  = os.path.join(args.save_root, 'result')
    save_modelroot  = os.path.join(args.save_root, 'model')
    if not os.path.exists(save_resroot):  os.makedirs(save_resroot)
    if not os.path.exists(save_modelroot): os.makedirs(save_modelroot)
    
    feature_name = "+".join(sorted(list(set(whole_features))))
    model_name = f'{args.model}+{args.feat_type}+{args.e2e_name}'
    prefix_name = f'features:{feature_name}_dataset:{args.dataset}_model:{model_name}'
    if args.train_dataset is not None:
        assert args.test_dataset is not None
        prefix_name += f'_train:{args.train_dataset}_test:{args.test_dataset}'


    print ('====== Reading Data =======')
    dataloader_class = get_dataloaders(args)
    train_loaders, eval_loaders, test_loaders = dataloader_class.get_loaders()
    assert len(train_loaders) == len(eval_loaders)
    print (f'train&val folder:{len(train_loaders)}; test sets:{len(test_loaders)}')
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()


    print ('====== Training and Evaluation =======')
    folder_save = []
    folder_duration = []
    for ii in range(len(train_loaders)):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        eval_loader  = eval_loaders[ii]
        start_time = name_time = time.time()

        print (f'Step1: build model (each folder has its own model)')
        model = get_models(args).cuda()
        if args.reg_loss_type == 'smoothl1':
            reg_loss = SmoothL1Loss(beta=args.huber_beta).cuda()
        else:
            reg_loss = MSELoss().cuda()
        cls_loss = CELoss().cuda()

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        
        # 学习率调度器 - 当验证指标不再提升时降低学习率
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',  # 最大化验证指标
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=True,
            min_lr=1e-6
        )
        
        # 早停机制
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience, 
            min_delta=0.001, 
            mode='max'
        )

        print (f'Step2: training (multiple epoches)')
        whole_store = []
        whole_metrics = []
        best_model_state = None
        
        for epoch in range(args.epochs):
            epoch_store = {}
            
            # 设置当前epoch用于渐进式模态dropout
            if hasattr(model.model, 'set_epoch'):
                model.model.set_epoch(epoch)

            train_results = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, dataloader_class, epoch=epoch, optimizer=optimizer, train=True)
            eval_results  = train_or_eval_model(args, model, reg_loss, cls_loss, eval_loader, dataloader_class, epoch=epoch, optimizer=None,      train=False)
            func_update_storage(inputs=eval_results, prefix='eval', outputs=epoch_store)

            train_metric = gain_metric_from_results(train_results, args.metric_name)
            eval_metric  = gain_metric_from_results(eval_results,  args.metric_name)
            whole_metrics.append(eval_metric)
            
            # 更新学习率调度器
            scheduler.step(eval_metric)
            current_lr = optimizer.param_groups[0]['lr']
            
            print (f'epoch:{epoch+1}; metric:{args.metric_name}; train:{train_metric:.4f}; eval:{eval_metric:.4f}; lr:{current_lr:.6f}')

            for jj, test_loader in enumerate(test_loaders):
                test_results = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, dataloader_class, epoch=epoch, optimizer=None, train=False)
                func_update_storage(inputs=test_results, prefix=f'test{jj+1}', outputs=epoch_store)
            
            whole_store.append(epoch_store)
            
            # 保存最佳模型状态
            if eval_metric >= max(whole_metrics):
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # 检查早停
            if early_stopping(eval_metric, epoch):
                print(f'Early stopping at epoch {epoch+1}, best epoch: {early_stopping.best_epoch+1}')
                break

        print (f'Step3: saving and testing on the {ii+1} folder')
        best_index = np.argmax(np.array(whole_metrics))
        folder_save.append(whole_store[best_index])
        end_time = time.time()
        duration = end_time - start_time
        folder_duration.append(duration)
        print (f'>>>>> Finish: training on the {ii+1}-th folder, best_index: {best_index}, duration: {duration} >>>>>')
        
        del model
        del optimizer
        torch.cuda.empty_cache()


    print ('====== Prediction and Saving =======')
    args.duration = np.sum(folder_duration)
    cv_result = gain_cv_results(folder_save)
    save_path = f'{save_resroot}/cv_{prefix_name}_{cv_result}_{name_time}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path, args=np.array(args, dtype=object))

    for jj in range(len(test_loaders)):
        emo_labels, emo_probs = average_folder_for_emos(folder_save, f'test{jj+1}')
        val_labels, val_preds = average_folder_for_vals(folder_save, f'test{jj+1}')
        _, test_result = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
        save_path = f'{save_resroot}/test{jj+1}_{prefix_name}_{test_result}_{name_time}.npz'
        print (f'save results in {save_path}')
        np.savez_compressed(save_path, args=np.array(args, dtype=object))
