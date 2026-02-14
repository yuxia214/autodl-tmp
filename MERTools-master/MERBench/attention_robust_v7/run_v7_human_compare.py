#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MERBENCH_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if MERBENCH_ROOT not in sys.path:
    sys.path.insert(0, MERBENCH_ROOT)

from toolkit.dataloader import get_dataloaders
from toolkit.globals import config, emos_mer, idx2emo_mer
from toolkit.models import get_models
from toolkit.utils.loss import CELoss, MSELoss, SmoothL1Loss
from toolkit.utils.metric import gain_metric_from_results


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.mode == "max":
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(value: str) -> str:
    return str(value).strip().lower()


def normalize_clip_name(value: str) -> str:
    name = str(value).strip()
    if name.endswith(".avi"):
        return name[:-4]
    return name


def run_epoch(args, model, reg_loss, cls_loss, dataloader, dataloader_class, optimizer=None, train=False):
    assert not train or optimizer is not None
    config.train = train
    model.train() if train else model.eval()

    names: List[str] = []
    losses = []
    emo_probs = []
    emo_labels = []
    val_preds = []
    val_labels = []

    for batch, emos, vals, bnames in dataloader:
        names.extend(bnames)
        for key in batch:
            batch[key] = batch[key].cuda(non_blocking=True)
        emos = emos.cuda(non_blocking=True)
        vals = vals.cuda(non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            _, emos_out, vals_out, interloss = model(batch)
            loss = interloss
            if args.output_dim1 != 0:
                loss = loss + args.emo_loss_weight * cls_loss(emos_out, emos)
                emo_probs.append(emos_out.detach().cpu().numpy())
                emo_labels.append(emos.detach().cpu().numpy())
            if args.output_dim2 != 0:
                loss = loss + args.val_loss_weight * reg_loss(vals_out, vals)
                val_preds.append(vals_out.detach().cpu().numpy())
                val_labels.append(vals.detach().cpu().numpy())

            if train:
                loss.backward()
                if model.model.grad_clip != -1:
                    torch.nn.utils.clip_grad_value_(
                        [p for p in model.parameters() if p.requires_grad],
                        model.model.grad_clip,
                    )
                optimizer.step()

        losses.append(float(loss.detach().cpu().numpy()))

    emo_probs = np.concatenate(emo_probs) if emo_probs else np.array([])
    emo_labels = np.concatenate(emo_labels) if emo_labels else np.array([])
    val_preds = np.concatenate(val_preds) if val_preds else np.array([])
    val_labels = np.concatenate(val_labels) if val_labels else np.array([])
    metrics, _ = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
    return {
        "names": names,
        "loss": float(np.mean(losses)) if losses else 0.0,
        **metrics,
    }


def evaluate_all_tests(args, model, reg_loss, cls_loss, dataloader_class, test_loaders):
    outputs = {}
    for idx, test_loader in enumerate(test_loaders, start=1):
        split = f"test{idx}"
        outputs[split] = run_epoch(
            args,
            model,
            reg_loss,
            cls_loss,
            test_loader,
            dataloader_class,
            train=False,
        )
    return outputs


def ensemble_split_results(split_name: str, fold_results: List[Dict], dataloader_class) -> Tuple[pd.DataFrame, Dict]:
    names = fold_results[0]["names"]
    for fold in fold_results[1:]:
        if fold["names"] != names:
            raise RuntimeError(f"{split_name} sample order mismatch across folds")

    emoprobs = np.mean(np.stack([fold["emoprobs"] for fold in fold_results], axis=0), axis=0)
    emolabels = fold_results[0]["emolabels"].astype(int)
    valpreds = np.mean(np.stack([fold["valpreds"] for fold in fold_results], axis=0), axis=0).reshape(-1)
    vallabels = fold_results[0]["vallabels"].reshape(-1)

    metrics, _ = dataloader_class.calculate_results(emoprobs, emolabels, valpreds, vallabels)
    pred_indices = np.argmax(emoprobs, axis=1).astype(int)

    df = pd.DataFrame(
        {
            "split": split_name,
            "clip_name": names,
            "true_label": [idx2emo_mer[int(x)] for x in emolabels],
            "pred_label": [idx2emo_mer[int(x)] for x in pred_indices],
            "true_valence": vallabels,
            "pred_valence": valpreds,
            "is_correct": (pred_indices == emolabels).astype(int),
        }
    )
    for cls_idx, cls_name in enumerate(emos_mer):
        df[f"prob_{cls_name}"] = emoprobs[:, cls_idx]

    summary = {
        "split": split_name,
        "n_samples": int(len(df)),
        "acc": float(metrics["emoacc"]),
        "f1": float(metrics["emofscore"]),
        "val_mse": float(metrics["valmse"]),
    }
    return df, summary


def save_human_compare(human_csv: str, test1_df: pd.DataFrame, output_dir: str, run_tag: str):
    human_df = pd.read_csv(human_csv).copy()
    human_df["clip_name"] = human_df["clip_name"].map(normalize_clip_name)
    human_df["true_label"] = human_df["true_label"].map(normalize_label)
    human_df["pred_label"] = human_df["pred_label"].map(normalize_label)
    if "is_correct" in human_df.columns:
        human_df["human_correct"] = human_df["is_correct"].astype(int)
    else:
        human_df["human_correct"] = (human_df["pred_label"] == human_df["true_label"]).astype(int)

    machine_df = test1_df.copy()
    machine_df["clip_name"] = machine_df["clip_name"].map(normalize_clip_name)
    machine_df["model_true_label"] = machine_df["true_label"].map(normalize_label)
    machine_df["machine_pred_label"] = machine_df["pred_label"].map(normalize_label)
    machine_df["machine_correct"] = machine_df["is_correct"].astype(int)
    machine_df = machine_df[["clip_name", "model_true_label", "machine_pred_label", "machine_correct"]]

    merged = human_df.merge(machine_df, on="clip_name", how="left")
    merged["label_mismatch_with_dataset"] = (
        (~merged["model_true_label"].isna()) & (merged["model_true_label"] != merged["true_label"])
    ).astype(int)

    valid = merged[~merged["machine_pred_label"].isna()].copy()
    summary = {
        "samples_in_human_csv": int(len(human_df)),
        "matched_samples": int(len(valid)),
        "missing_machine_predictions": int(merged["machine_pred_label"].isna().sum()),
        "human_acc_on_matched": float(valid["human_correct"].mean()) if len(valid) > 0 else float("nan"),
        "machine_acc_on_matched": float(valid["machine_correct"].mean()) if len(valid) > 0 else float("nan"),
        "machine_minus_human": (
            float(valid["machine_correct"].mean() - valid["human_correct"].mean()) if len(valid) > 0 else float("nan")
        ),
        "label_mismatch_with_dataset_count": int(merged["label_mismatch_with_dataset"].sum()),
    }

    detail_path = os.path.join(output_dir, f"human_vs_machine_test1_{run_tag}.csv")
    summary_path = os.path.join(output_dir, f"human_vs_machine_summary_{run_tag}.csv")
    merged.to_csv(detail_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary, detail_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_csv", type=str, default=None, help="optional human labeling csv for test1 subset compare")
    parser.add_argument("--output_dir", type=str, default="./attention_robust_v7/outputs/human_compare")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--inference_only", action="store_true", default=False)

    parser.add_argument("--dataset", type=str, default="MER2023")
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--test_dataset", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--audio_feature", type=str, default="chinese-hubert-large-UTT")
    parser.add_argument("--text_feature", type=str, default="Baichuan-13B-Base-UTT")
    parser.add_argument("--video_feature", type=str, default="clip-vit-large-patch14-UTT")
    parser.add_argument("--feat_type", type=str, default="utt")
    parser.add_argument("--feat_scale", type=int, default=1)
    parser.add_argument("--e2e_name", type=str, default=None)
    parser.add_argument("--e2e_dim", type=int, default=None)
    parser.add_argument("--model", type=str, default="attention_robust_v7")
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--hyper_path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--l2", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_iters", type=float, default=1e8)

    parser.add_argument("--emo_loss_weight", type=float, default=1.0)
    parser.add_argument("--val_loss_weight", type=float, default=1.3)
    parser.add_argument("--reg_loss_type", type=str, default="smoothl1", choices=["mse", "smoothl1"])
    parser.add_argument("--huber_beta", type=float, default=0.8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--modality_dropout", type=float, default=0.18)
    parser.add_argument("--use_modality_dropout", action="store_true", default=True)
    parser.add_argument("--modality_dropout_warmup", type=int, default=15)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_factor", type=float, default=0.5)

    parser.add_argument("--use_vae", action="store_true", default=True)
    parser.add_argument("--kl_weight", type=float, default=0.01)
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument("--cross_kl_weight", type=float, default=0.01)
    parser.add_argument("--use_proxy_attention", action="store_true", default=True)
    parser.add_argument("--fusion_temperature", type=float, default=1.0)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--use_dynamic_kl", action="store_true", default=True)
    parser.add_argument("--kl_warmup_epochs", type=int, default=20)
    parser.add_argument("--use_valence_prior", action="store_true", default=True)
    parser.add_argument("--valence_consistency_weight", type=float, default=0.12)
    parser.add_argument("--valence_center_reg_weight", type=float, default=0.005)
    parser.add_argument("--feature_noise_std", type=float, default=0.03)
    parser.add_argument("--feature_noise_prob", type=float, default=0.35)
    parser.add_argument("--feature_noise_warmup", type=int, default=5)
    return parser.parse_args()


def resolve_paths(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.output_dir, "models")
    os.makedirs(args.checkpoint_dir, exist_ok=True)


def get_checkpoint_path(checkpoint_dir: str, fold_idx: int, seed: int) -> str:
    return os.path.join(checkpoint_dir, f"attention_robust_v7_seed{seed}_fold{fold_idx}.pt")


def main():
    args = parse_args()
    resolve_paths(args)
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    if args.feat_type == "utt":
        args.feat_scale = 1
    elif args.feat_type == "frm_align":
        args.feat_scale = 6
    elif args.feat_type == "frm_unalign":
        args.feat_scale = 12

    run_tag = time.strftime("%Y%m%d_%H%M%S")

    print("====== Reading Data ======")
    dataloader_class = get_dataloaders(args)
    train_loaders, eval_loaders, test_loaders = dataloader_class.get_loaders()
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()
    print(f"folds: {len(train_loaders)}, tests: {len(test_loaders)}")

    fold_test_results: Dict[str, List[Dict]] = {f"test{i + 1}": [] for i in range(len(test_loaders))}
    train_start = time.time()

    if args.inference_only:
        print("====== Inference only: loading saved checkpoints ======")
        for fold_idx in range(1, len(train_loaders) + 1):
            ckpt_path = get_checkpoint_path(args.checkpoint_dir, fold_idx, args.seed)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

            model = get_models(args).cuda()
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=True)
            reg_loss = SmoothL1Loss(beta=args.huber_beta).cuda() if args.reg_loss_type == "smoothl1" else MSELoss().cuda()
            cls_loss = CELoss().cuda()

            this_fold_tests = evaluate_all_tests(args, model, reg_loss, cls_loss, dataloader_class, test_loaders)
            for split in this_fold_tests:
                fold_test_results[split].append(this_fold_tests[split])

            print(f"loaded fold {fold_idx}, ckpt: {ckpt_path}")
            del model
            torch.cuda.empty_cache()
    else:
        print("====== Training + checkpointing ======")
        for fold_idx in range(len(train_loaders)):
            print(f">>>>> Fold {fold_idx + 1}/{len(train_loaders)} >>>>>")
            train_loader = train_loaders[fold_idx]
            eval_loader = eval_loaders[fold_idx]

            model = get_models(args).cuda()
            reg_loss = SmoothL1Loss(beta=args.huber_beta).cuda() if args.reg_loss_type == "smoothl1" else MSELoss().cuda()
            cls_loss = CELoss().cuda()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=args.lr_factor,
                patience=args.lr_patience,
                verbose=False,
                min_lr=1e-6,
            )
            early_stopping = EarlyStopping(
                patience=args.early_stopping_patience,
                min_delta=0.001,
                mode="max",
            )

            best_metric = -1e9
            best_epoch = -1
            best_state = None

            for epoch in range(args.epochs):
                if hasattr(model.model, "set_epoch"):
                    model.model.set_epoch(epoch)

                _ = run_epoch(args, model, reg_loss, cls_loss, train_loader, dataloader_class, optimizer=optimizer, train=True)
                eval_results = run_epoch(args, model, reg_loss, cls_loss, eval_loader, dataloader_class, train=False)
                eval_metric = gain_metric_from_results(eval_results, args.metric_name)
                scheduler.step(eval_metric)
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"fold:{fold_idx + 1} epoch:{epoch + 1} eval_metric:{eval_metric:.4f} lr:{current_lr:.6f}")

                if eval_metric >= best_metric:
                    best_metric = eval_metric
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if early_stopping(eval_metric, epoch):
                    print(f"fold:{fold_idx + 1} early stop at epoch {epoch + 1}, best epoch {best_epoch + 1}")
                    break

            model.load_state_dict(best_state)

            ckpt_path = get_checkpoint_path(args.checkpoint_dir, fold_idx + 1, args.seed)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch + 1,
                    "best_metric": best_metric,
                    "seed": args.seed,
                    "fold": fold_idx + 1,
                },
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")

            this_fold_tests = evaluate_all_tests(args, model, reg_loss, cls_loss, dataloader_class, test_loaders)
            for split in this_fold_tests:
                fold_test_results[split].append(this_fold_tests[split])
            print(
                f"fold:{fold_idx + 1} "
                + " ".join(
                    [
                        f"{split}_acc:{this_fold_tests[split]['emoacc']:.4f}"
                        for split in sorted(this_fold_tests.keys())
                    ]
                )
            )

            del model
            del optimizer
            torch.cuda.empty_cache()

    duration = time.time() - train_start
    print(f"training/inference stage done in {duration:.1f}s")

    print("====== Ensembling and saving all test outputs ======")
    all_split_summaries = []
    split_prediction_paths = {}
    split_dfs = {}

    for split in sorted(fold_test_results.keys()):
        split_df, split_summary = ensemble_split_results(split, fold_test_results[split], dataloader_class)
        split_dfs[split] = split_df
        all_split_summaries.append(split_summary)
        pred_path = os.path.join(args.output_dir, f"{split}_ensemble_predictions_{run_tag}.csv")
        split_df.to_csv(pred_path, index=False)
        split_prediction_paths[split] = pred_path

    summary_df = pd.DataFrame(all_split_summaries)
    summary_df["duration_seconds"] = duration
    summary_path = os.path.join(args.output_dir, f"all_tests_metrics_{run_tag}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("saved split prediction files:")
    for split in sorted(split_prediction_paths.keys()):
        print(f"  {split}: {split_prediction_paths[split]}")
    print(f"saved summary: {summary_path}")

    if args.human_csv:
        print("====== Human vs machine (test1 subset) ======")
        human_summary, human_detail_path, human_summary_path = save_human_compare(
            args.human_csv,
            split_dfs["test1"],
            args.output_dir,
            run_tag,
        )
        for key, value in human_summary.items():
            print(f"{key}: {value}")
        print(f"human detail: {human_detail_path}")
        print(f"human summary: {human_summary_path}")


if __name__ == "__main__":
    main()
