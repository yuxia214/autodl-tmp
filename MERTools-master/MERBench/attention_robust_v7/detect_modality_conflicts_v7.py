#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MERBENCH_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if MERBENCH_ROOT not in sys.path:
    sys.path.insert(0, MERBENCH_ROOT)

from toolkit.dataloader import get_dataloaders
from toolkit.globals import emos_mer, idx2emo_mer
from toolkit.models import get_models


ALL_MODALITY_KEYS = ("audios", "texts", "videos")
SINGLE_MODALITY_MAP = {
    "audio_only": ("audios",),
    "text_only": ("texts",),
    "video_only": ("videos",),
}
ALL_INFER_MODES = ("full", "audio_only", "text_only", "video_only")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def parse_splits(raw: str) -> List[str]:
    text = raw.strip().lower()
    if text == "all":
        return ["test1", "test2", "test3"]

    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    valid = {"test1", "test2", "test3"}
    if not items:
        raise ValueError("--test_splits is empty")
    for item in items:
        if item not in valid:
            raise ValueError(f"invalid split: {item}, valid: test1,test2,test3,all")
    return items


def build_masked_batch(batch: Dict[str, torch.Tensor], keep_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    keep_set = set(keep_keys)
    out = {}
    for key in ALL_MODALITY_KEYS:
        value = batch[key]
        out[key] = value if key in keep_set else torch.zeros_like(value)
    return out


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def confidence_and_margin(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = softmax_numpy(logits)
    confidence = np.max(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return confidence, margin


def sign_bucket(values: np.ndarray, eps: float) -> np.ndarray:
    out = np.full(values.shape, "neu", dtype=object)
    out[values > eps] = "pos"
    out[values < -eps] = "neg"
    return out


def get_checkpoint_path(checkpoint_dir: str, fold_idx: int, seed: int) -> str:
    return os.path.join(checkpoint_dir, f"attention_robust_v7_seed{seed}_fold{fold_idx}.pt")


def infer_one_loader(model, dataloader, device: torch.device) -> Dict[str, np.ndarray]:
    names: List[str] = []
    true_emo: List[np.ndarray] = []
    true_val: List[np.ndarray] = []

    emo_logits = {mode: [] for mode in ALL_INFER_MODES}
    val_preds = {mode: [] for mode in ALL_INFER_MODES}

    model.eval()
    with torch.no_grad():
        for batch, emos, vals, bnames in dataloader:
            names.extend(bnames)
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)

            true_emo.append(emos.numpy())
            true_val.append(vals.numpy())

            _, e_full, v_full, _ = model(batch)
            emo_logits["full"].append(e_full.detach().cpu().numpy())
            val_preds["full"].append(v_full.detach().cpu().numpy().reshape(-1))

            for mode, keep in SINGLE_MODALITY_MAP.items():
                masked_batch = build_masked_batch(batch, keep)
                _, e_out, v_out, _ = model(masked_batch)
                emo_logits[mode].append(e_out.detach().cpu().numpy())
                val_preds[mode].append(v_out.detach().cpu().numpy().reshape(-1))

    result: Dict[str, np.ndarray] = {
        "names": np.array(names, dtype=object),
        "true_emo": np.concatenate(true_emo, axis=0).astype(np.int64),
        "true_val": np.concatenate(true_val, axis=0).astype(np.float32),
    }
    for mode in ALL_INFER_MODES:
        result[f"{mode}_emo_logits"] = np.concatenate(emo_logits[mode], axis=0).astype(np.float32)
        result[f"{mode}_val_pred"] = np.concatenate(val_preds[mode], axis=0).astype(np.float32)
    return result


def ensemble_fold_outputs(fold_outputs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not fold_outputs:
        raise RuntimeError("empty fold outputs")

    base_names = fold_outputs[0]["names"]
    base_true_emo = fold_outputs[0]["true_emo"]
    base_true_val = fold_outputs[0]["true_val"]

    for idx, each in enumerate(fold_outputs[1:], start=2):
        if not np.array_equal(base_names, each["names"]):
            raise RuntimeError(f"name order mismatch across folds, fold {idx}")
        if not np.array_equal(base_true_emo, each["true_emo"]):
            raise RuntimeError(f"emotion labels mismatch across folds, fold {idx}")
        if not np.allclose(base_true_val, each["true_val"]):
            raise RuntimeError(f"valence labels mismatch across folds, fold {idx}")

    out: Dict[str, np.ndarray] = {
        "names": base_names,
        "true_emo": base_true_emo,
        "true_val": base_true_val,
    }

    for mode in ALL_INFER_MODES:
        emo_stack = np.stack([x[f"{mode}_emo_logits"] for x in fold_outputs], axis=0)
        val_stack = np.stack([x[f"{mode}_val_pred"] for x in fold_outputs], axis=0)
        out[f"{mode}_emo_logits"] = np.mean(emo_stack, axis=0)
        out[f"{mode}_val_pred"] = np.mean(val_stack, axis=0)

    return out


def build_conflict_dataframe(split_name: str, ensemble_out: Dict[str, np.ndarray], valence_eps: float) -> pd.DataFrame:
    names = ensemble_out["names"]
    true_emo_idx = ensemble_out["true_emo"]
    true_val = ensemble_out["true_val"]

    logits = {mode: ensemble_out[f"{mode}_emo_logits"] for mode in ALL_INFER_MODES}
    vals = {mode: ensemble_out[f"{mode}_val_pred"] for mode in ALL_INFER_MODES}

    preds_idx = {mode: np.argmax(logits[mode], axis=1).astype(np.int64) for mode in ALL_INFER_MODES}
    conf = {}
    margin = {}
    for mode in ALL_INFER_MODES:
        conf[mode], margin[mode] = confidence_and_margin(logits[mode])

    n = len(names)
    conflict_type = np.full(n, "no_conflict", dtype=object)
    outlier_modality = np.full(n, "none", dtype=object)
    majority_idx = np.full(n, -1, dtype=np.int64)
    unique_count = np.zeros(n, dtype=np.int64)

    a_idx = preds_idx["audio_only"]
    t_idx = preds_idx["text_only"]
    v_idx = preds_idx["video_only"]
    f_idx = preds_idx["full"]

    audio_text_same = a_idx == t_idx
    audio_video_same = a_idx == v_idx
    text_video_same = t_idx == v_idx

    for i in range(n):
        ai, ti, vi = int(a_idx[i]), int(t_idx[i]), int(v_idx[i])
        uniq = len({ai, ti, vi})
        unique_count[i] = uniq
        if uniq == 1:
            majority_idx[i] = ai
            continue

        if uniq == 2:
            conflict_type[i] = "one_vs_two"
            if ai == ti:
                majority_idx[i] = ai
                outlier_modality[i] = "video"
            elif ai == vi:
                majority_idx[i] = ai
                outlier_modality[i] = "text"
            else:
                majority_idx[i] = ti
                outlier_modality[i] = "audio"
            continue

        conflict_type[i] = "all_three_diff"

    true_labels = np.array([idx2emo_mer[int(x)] for x in true_emo_idx], dtype=object)

    mode_pred_labels = {}
    for mode in ALL_INFER_MODES:
        mode_pred_labels[mode] = np.array([idx2emo_mer[int(x)] for x in preds_idx[mode]], dtype=object)

    majority_label = np.array([idx2emo_mer[int(x)] if x >= 0 else "" for x in majority_idx], dtype=object)
    is_conflict = conflict_type != "no_conflict"
    full_match_majority = (majority_idx >= 0) & (f_idx == majority_idx)

    val_sign = {mode: sign_bucket(vals[mode], valence_eps) for mode in ALL_INFER_MODES}
    val_sign_unique_count = np.array(
        [
            len({val_sign["audio_only"][i], val_sign["text_only"][i], val_sign["video_only"][i]})
            for i in range(n)
        ],
        dtype=np.int64,
    )

    df = pd.DataFrame(
        {
            "split": split_name,
            "clip_name": names,
            "true_label": true_labels,
            "true_valence": true_val,
            "full_pred_label": mode_pred_labels["full"],
            "full_pred_valence": vals["full"],
            "full_confidence": conf["full"],
            "full_margin": margin["full"],
            "full_correct": (f_idx == true_emo_idx).astype(np.int64),
            "audio_pred_label": mode_pred_labels["audio_only"],
            "audio_pred_valence": vals["audio_only"],
            "audio_confidence": conf["audio_only"],
            "audio_margin": margin["audio_only"],
            "text_pred_label": mode_pred_labels["text_only"],
            "text_pred_valence": vals["text_only"],
            "text_confidence": conf["text_only"],
            "text_margin": margin["text_only"],
            "video_pred_label": mode_pred_labels["video_only"],
            "video_pred_valence": vals["video_only"],
            "video_confidence": conf["video_only"],
            "video_margin": margin["video_only"],
            "audio_text_same": audio_text_same.astype(np.int64),
            "audio_video_same": audio_video_same.astype(np.int64),
            "text_video_same": text_video_same.astype(np.int64),
            "modality_unique_count": unique_count,
            "conflict_type": conflict_type,
            "outlier_modality": outlier_modality,
            "majority_label": majority_label,
            "full_match_majority": full_match_majority.astype(np.int64),
            "is_conflict": is_conflict.astype(np.int64),
            "audio_text_vs_video_conflict": (audio_text_same & (~audio_video_same)).astype(np.int64),
            "audio_valence_sign": val_sign["audio_only"],
            "text_valence_sign": val_sign["text_only"],
            "video_valence_sign": val_sign["video_only"],
            "valence_sign_unique_count": val_sign_unique_count,
            "valence_sign_conflict": (val_sign_unique_count > 1).astype(np.int64),
        }
    )

    return df


def summarize_conflicts(split_name: str, df: pd.DataFrame) -> Dict[str, object]:
    samples = len(df)
    conflict_count = int(df["is_conflict"].sum())
    majority_rows = df[df["majority_label"] != ""]

    if len(majority_rows) > 0:
        full_match_majority_ratio = float(majority_rows["full_match_majority"].mean())
    else:
        full_match_majority_ratio = float("nan")

    return {
        "split": split_name,
        "samples": int(samples),
        "conflict_count": conflict_count,
        "conflict_ratio": float(conflict_count / samples) if samples > 0 else 0.0,
        "one_vs_two_count": int((df["conflict_type"] == "one_vs_two").sum()),
        "all_three_diff_count": int((df["conflict_type"] == "all_three_diff").sum()),
        "audio_outlier_count": int((df["outlier_modality"] == "audio").sum()),
        "text_outlier_count": int((df["outlier_modality"] == "text").sum()),
        "video_outlier_count": int((df["outlier_modality"] == "video").sum()),
        "audio_text_vs_video_count": int(df["audio_text_vs_video_conflict"].sum()),
        "valence_sign_conflict_count": int(df["valence_sign_conflict"].sum()),
        "full_match_majority_ratio": full_match_majority_ratio,
        "full_accuracy": float(df["full_correct"].mean()) if samples > 0 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Detect modality conflicts with AttentionRobustV7 checkpoints")

    parser.add_argument("--output_dir", type=str, default="./attention_robust_v7/outputs/modality_conflict")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./attention_robust_v7/outputs/human_compare/models",
        help="directory containing attention_robust_v7_seed{seed}_fold{k}.pt",
    )
    parser.add_argument("--test_splits", type=str, default="all", help="test1,test2,test3 or all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--valence_sign_eps", type=float, default=0.2)

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


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir, MERBENCH_ROOT)
    args.checkpoint_dir = resolve_path(args.checkpoint_dir, MERBENCH_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)

    selected_splits = parse_splits(args.test_splits)

    if args.feat_type == "utt":
        args.feat_scale = 1
    elif args.feat_type == "frm_align":
        args.feat_scale = 6
    elif args.feat_type == "frm_unalign":
        args.feat_scale = 12

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("====== Reading Data ======")
    dataloader_class = get_dataloaders(args)
    train_loaders, _, test_loaders = dataloader_class.get_loaders()
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()

    split_loader_map = {
        "test1": test_loaders[0],
        "test2": test_loaders[1],
        "test3": test_loaders[2],
    }

    fold_outputs_by_split: Dict[str, List[Dict[str, np.ndarray]]] = {k: [] for k in selected_splits}

    print("====== Inference with checkpoints ======")
    run_start = time.time()
    for fold_idx in range(1, args.num_folds + 1):
        ckpt_path = get_checkpoint_path(args.checkpoint_dir, fold_idx, args.seed)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        model = get_models(args).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)

        for split_name in selected_splits:
            loader = split_loader_map[split_name]
            fold_out = infer_one_loader(model, loader, device)
            fold_outputs_by_split[split_name].append(fold_out)
            print(f"fold {fold_idx} done: {split_name}, samples={len(fold_out['names'])}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    summary_rows = []

    print("====== Ensembling + conflict detection ======")
    for split_name in selected_splits:
        ens_out = ensemble_fold_outputs(fold_outputs_by_split[split_name])
        df = build_conflict_dataframe(split_name, ens_out, args.valence_sign_eps)

        all_path = os.path.join(args.output_dir, f"{split_name}_all_samples_{run_tag}.csv")
        conflicts_path = os.path.join(args.output_dir, f"{split_name}_conflicts_{run_tag}.csv")
        atv_path = os.path.join(args.output_dir, f"{split_name}_audio_text_vs_video_{run_tag}.csv")

        df.to_csv(all_path, index=False)
        df[df["is_conflict"] == 1].to_csv(conflicts_path, index=False)
        df[df["audio_text_vs_video_conflict"] == 1].to_csv(atv_path, index=False)

        summary = summarize_conflicts(split_name, df)
        summary_rows.append(summary)

        print(
            f"{split_name}: conflict={summary['conflict_count']}/{summary['samples']} "
            f"({summary['conflict_ratio']:.4f}), one_vs_two={summary['one_vs_two_count']}, "
            f"all_three_diff={summary['all_three_diff_count']}"
        )
        print(f"  all: {all_path}")
        print(f"  conflicts: {conflicts_path}")
        print(f"  audio_text_vs_video: {atv_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df["duration_seconds"] = time.time() - run_start
    summary_path = os.path.join(args.output_dir, f"summary_{run_tag}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("====== Done ======")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
