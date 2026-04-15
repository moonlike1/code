import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from leakage_detection_torch.fusion_liquid_model_v2_torch import DN_MS_LiquidNet_V2_Torch
from leakage_detection_torch.liquid_dataset_torch import LiquidLeakageDatasetTorch


def add_toggle_arg(parser: argparse.ArgumentParser, enable_flag: str, disable_flag: str, dest: str, default: bool):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(enable_flag, dest=dest, action="store_true")
    group.add_argument(disable_flag, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="../full_area_train_v2.h5")
    parser.add_argument("--test_path", type=str, default="../full_area_val_v2.h5")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--augment_mode", type=str, default="strong", choices=["basic", "strong"])

    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--seg_classes", type=int, default=2)
    parser.add_argument("--cls_classes", type=int, default=2)
    parser.add_argument("--binary_class", dest="binary_class", action="store_true", default=True)
    parser.add_argument("--no_binary_class", dest="binary_class", action="store_false")
    parser.add_argument("--k_scales", type=int, nargs="+", default=[10, 20, 40])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--seg_use_class_weights", action="store_true", default=False)
    parser.add_argument("--no_seg_class_weights", action="store_true", default=False)
    parser.add_argument(
        "--seg_class_weight_mode",
        type=str,
        default="sqrt_inv",
        choices=["sqrt_inv", "inverse", "effective_num"],
    )
    parser.add_argument("--seg_class_weight_beta", type=float, default=0.999999)
    parser.add_argument("--seg_loss", type=str, default="dice_ce", choices=["ce", "dice_ce", "focal_ce"])
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--dice_smooth", type=float, default=1.0)
    parser.add_argument("--dice_target", type=str, default="leak", choices=["leak", "all"])
    parser.add_argument("--focal_weight", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)

    add_toggle_arg(parser, "--use_noise_guidance", "--no_noise_guidance", "use_noise_guidance", True)
    add_toggle_arg(parser, "--use_progressive", "--no_progressive", "use_progressive", True)
    add_toggle_arg(parser, "--use_noise_leak_corr", "--no_noise_leak_corr", "use_noise_leak_corr", True)
    add_toggle_arg(parser, "--use_multi_scale", "--no_multi_scale", "use_multi_scale", True)
    add_toggle_arg(parser, "--use_uncertainty_fusion", "--no_uncertainty_fusion", "use_uncertainty_fusion", True)
    parser.add_argument("--use_simple_uncertainty", action="store_true", default=False)
    parser.add_argument(
        "--seg_only",
        action="store_true",
        default=False,
        help="Segmentation-only comparison mode: disable classification and uncertainty classification branch",
    )
    parser.add_argument("--disable_cls", dest="disable_cls", action="store_true", default=None)
    parser.add_argument("--enable_cls", dest="disable_cls", action="store_false")
    parser.add_argument(
        "--boundary_input_mode",
        type=str,
        default="features_fine_probs",
        choices=["features", "features_fine_probs"],
    )

    parser.add_argument("--output_dir", type=str, default="./output/v2_training_torch_dataset_v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_iou",
        choices=[
            "val_iou",
            "val_leak_iou",
            "val_global_leak_iou",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "val_cls_acc",
            "val_cls_macro_f1",
            "val_cls_balanced_acc",
            "val_cls_weighted_f1",
            "cls_acc",
            "cls_macro_f1",
            "cls_balanced_acc",
            "cls_weighted_f1",
        ],
    )
    parser.add_argument(
        "--train_sampler",
        type=str,
        default="shuffle",
        choices=["shuffle", "balanced_presence", "small_positive"],
    )
    parser.add_argument("--sampler_alpha", type=float, default=0.5)
    parser.add_argument("--sampler_min_positive_ratio", type=float, default=0.005)
    parser.add_argument("--sampler_max_weight", type=float, default=4.0)
    parser.add_argument("--sampler_empty_weight", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "multistep"])
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--scheduler_milestones", type=float, nargs="*", default=[0.6, 0.8])
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_start_factor", type=float, default=0.2)
    parser.add_argument("--topk_checkpoints", type=int, default=3)

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    def resolve_data_path(p):
        p = Path(p)
        if p.is_absolute() and p.exists():
            return str(p)
        if p.exists():
            return str(p.resolve())
        cand = (script_dir / p).resolve()
        if cand.exists():
            return str(cand)
        return str(p)

    args.train_path = resolve_data_path(args.train_path)
    args.test_path = resolve_data_path(args.test_path)

    if args.no_seg_class_weights:
        args.seg_use_class_weights = False

    if args.binary_class:
        args.cls_classes = 2
    elif args.cls_classes == 2:
        args.cls_classes = 4

    if args.disable_cls is None:
        args.disable_cls = bool(args.binary_class and args.cls_classes == 2)
        if args.disable_cls:
            args.cls_weight = 0.0

    if args.seg_only:
        args.disable_cls = True
        args.cls_weight = 0.0
        args.use_uncertainty_fusion = False
        args.use_simple_uncertainty = False

    if args.disable_cls:
        args.use_uncertainty_fusion = False

    if (args.best_metric.startswith("val_cls_") or args.best_metric.startswith("cls_")) and args.disable_cls:
        raise ValueError(f"best_metric={args.best_metric} 需要启用分类分支，当前 disable_cls=True")

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.topk_checkpoints = max(int(args.topk_checkpoints), 0)
    args.warmup_epochs = max(int(args.warmup_epochs), 0)
    args.warmup_start_factor = float(np.clip(args.warmup_start_factor, 0.0, 1.0))
    args.min_lr = max(float(args.min_lr), 0.0)

    return args


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_iou_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int = 2):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        ious.append((intersection / union) if union > 0 else 0.0)
    return ious


def compute_binary_confusion(pred: np.ndarray, target: np.ndarray, positive_class: int = 1):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    pred_pos = pred == positive_class
    target_pos = target == positive_class
    tp = int(np.logical_and(pred_pos, target_pos).sum())
    fp = int(np.logical_and(pred_pos, ~target_pos).sum())
    fn = int(np.logical_and(~pred_pos, target_pos).sum())
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, tn, fn


def compute_binary_metrics(tp: int, fp: int, tn: int, fn: int):
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    global_leak_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    global_bg_iou = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
    global_miou = 0.5 * (global_leak_iou + global_bg_iou)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "global_leak_iou": float(global_leak_iou),
        "global_bg_iou": float(global_bg_iou),
        "global_miou": float(global_miou),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_multiclass_classification_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int):
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    valid = (
        (pred >= 0)
        & (pred < int(num_classes))
        & (target >= 0)
        & (target < int(num_classes))
    )
    pred = pred[valid]
    target = target[valid]
    confusion = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    if pred.size > 0:
        np.add.at(confusion, (target, pred), 1)
    tp = np.diag(confusion).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    target_count = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, target_count, out=np.zeros_like(tp), where=target_count > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    total = float(confusion.sum())
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    balanced_acc = float(recall.mean()) if recall.size > 0 else 0.0
    macro_precision = float(precision.mean()) if precision.size > 0 else 0.0
    macro_recall = float(recall.mean()) if recall.size > 0 else 0.0
    macro_f1 = float(f1.mean()) if f1.size > 0 else 0.0
    weighted_f1 = float(np.average(f1, weights=target_count)) if target_count.sum() > 0 else 0.0
    return {
        "cls_acc": accuracy,
        "cls_macro_precision": macro_precision,
        "cls_macro_recall": macro_recall,
        "cls_macro_f1": macro_f1,
        "cls_balanced_acc": balanced_acc,
        "cls_weighted_f1": weighted_f1,
    }


def seg_nll_loss_from_probs(probs: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None, eps: float = 1e-7):
    probs = probs.clamp(min=eps, max=1.0 - eps)
    logp = torch.log(probs)
    gathered = logp.gather(dim=1, index=labels.unsqueeze(1))
    loss = -gathered.squeeze(1)
    if class_weights is not None:
        w = class_weights.to(device=labels.device, dtype=probs.dtype)[labels]
        loss = loss * w
    return loss.mean()


def dice_loss_from_probs(probs: torch.Tensor, labels: torch.Tensor, num_classes: int, smooth: float = 1.0, target: str = "leak"):
    labels_oh = F.one_hot(labels, num_classes=num_classes).permute(0, 2, 1).to(dtype=probs.dtype)
    intersection = (probs * labels_oh).sum(dim=(0, 2))
    denom = probs.sum(dim=(0, 2)) + labels_oh.sum(dim=(0, 2))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    if target == "leak" and num_classes >= 2:
        dice = dice[1:2]
    return 1.0 - dice.mean()


def focal_loss_from_probs(probs: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7):
    probs = probs.clamp(min=eps, max=1.0 - eps)
    pt = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    logpt = torch.log(pt)
    if probs.shape[1] == 2:
        alpha_t = torch.where(labels == 1, torch.tensor(alpha, device=probs.device, dtype=probs.dtype), torch.tensor(1.0 - alpha, device=probs.device, dtype=probs.dtype))
    else:
        alpha_t = 1.0
    loss = -alpha_t * torch.pow(1.0 - pt, gamma) * logpt
    return loss.mean()


def build_seg_loss_fn(args, seg_class_weights: torch.Tensor | None):
    def loss_fn(seg_probs: torch.Tensor, seg_labels: torch.Tensor):
        ce = seg_nll_loss_from_probs(seg_probs, seg_labels, class_weights=seg_class_weights)
        if args.seg_loss == "ce":
            return ce
        if args.seg_loss == "dice_ce":
            dice = dice_loss_from_probs(seg_probs, seg_labels, num_classes=args.seg_classes, smooth=args.dice_smooth, target=args.dice_target)
            return ce + args.dice_weight * dice
        if args.seg_loss == "focal_ce":
            focal = focal_loss_from_probs(seg_probs, seg_labels, gamma=args.focal_gamma, alpha=args.focal_alpha)
            return (1.0 - args.focal_weight) * ce + args.focal_weight * focal
        return ce

    return loss_fn


def compute_cls_class_weights(dataset, num_classes: int, max_samples: int = 2000) -> torch.Tensor:
    sample_count = min(len(dataset), int(max_samples))
    if sample_count <= 0:
        return torch.ones(num_classes, dtype=torch.float32)
    indices = np.random.choice(len(dataset), sample_count, replace=False)
    cls_counts = np.zeros(int(num_classes), dtype=np.float64)
    for idx in indices:
        _, _, cls_label = dataset[int(idx)]
        label = int(cls_label.item())
        if 0 <= label < int(num_classes):
            cls_counts[label] += 1.0
    cls_counts = np.maximum(cls_counts, 1.0)
    cls_weights = 1.0 / np.sqrt(cls_counts)
    cls_weights = cls_weights / cls_weights.sum() * float(len(cls_counts))
    return torch.tensor(cls_weights, dtype=torch.float32)


def serialize_metric_dict(metrics: dict[str, float | int] | None) -> dict[str, float | int] | None:
    if metrics is None:
        return None
    serialized = {}
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            serialized[key] = int(value)
        else:
            serialized[key] = float(value)
    return serialized


def metric_ranking_tuple(
    val_metrics: dict[str, float | int],
    primary_metric_key: str,
    tie_break_metric_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[float, ...]:
    ranking = [float(val_metrics[primary_metric_key])]
    for key in tie_break_metric_keys or ():
        ranking.append(float(val_metrics.get(key, float("-inf"))))
    return tuple(ranking)


def format_metric_ranking(
    primary_metric_key: str,
    tie_break_metric_keys: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    return [str(primary_metric_key)] + [str(key) for key in (tie_break_metric_keys or ())]


def is_better_ranking(candidate: tuple[float, ...], reference: tuple[float, ...] | None) -> bool:
    if reference is None:
        return True
    return candidate > reference


def resolve_scheduler_milestones(milestones: list[float], total_epochs: int) -> list[int]:
    resolved = []
    total_epochs = max(int(total_epochs), 1)
    for milestone in milestones:
        value = float(milestone)
        if value <= 0:
            continue
        if value < 1.0:
            epoch = int(round(total_epochs * value))
        else:
            epoch = int(round(value))
        epoch = min(max(epoch, 1), total_epochs)
        resolved.append(epoch)
    return sorted(set(resolved))


def compute_epoch_lr(epoch: int, args, scheduler_milestones: list[int]) -> float:
    base_lr = float(args.lr)
    if base_lr <= 0.0:
        return 0.0

    warmup_epochs = min(int(args.warmup_epochs), max(int(args.epochs) - 1, 0))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        factor = args.warmup_start_factor + (1.0 - args.warmup_start_factor) * progress
        return base_lr * factor

    if args.scheduler == "cosine":
        cosine_epochs = max(int(args.epochs) - warmup_epochs, 1)
        progress_numer = max(epoch - warmup_epochs - 1, 0)
        progress_denom = max(cosine_epochs - 1, 1)
        progress = min(progress_numer / progress_denom, 1.0)
        min_lr = min(float(args.min_lr), base_lr)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_factor

    if args.scheduler == "multistep":
        decay_steps = sum(epoch >= milestone for milestone in scheduler_milestones)
        return base_lr * (float(args.scheduler_gamma) ** decay_steps)

    return base_lr


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> float:
    lr = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def update_topk_checkpoints(
    model: torch.nn.Module,
    save_dir: str,
    topk_records: list[dict[str, object]],
    metric_name: str,
    score: float,
    ranking: tuple[float, ...],
    ranking_keys: list[str] | tuple[str, ...],
    epoch: int,
    val_metrics: dict[str, float | int],
    topk_limit: int,
) -> list[dict[str, object]]:
    if topk_limit <= 0:
        return topk_records

    lowest_ranking = None
    if len(topk_records) >= topk_limit:
        last_item = topk_records[-1]
        lowest_ranking = tuple(float(v) for v in last_item.get("ranking", [float(last_item["score"])]))
    qualifies = len(topk_records) < topk_limit or ranking > lowest_ranking
    if not qualifies:
        return topk_records

    metric_tag = metric_name.replace("val_", "").replace("/", "_")
    checkpoint_name = f"checkpoint_epoch{epoch:03d}_{metric_tag}_{score:.6f}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)

    record = {
        "epoch": int(epoch),
        "score": float(score),
        "metric": metric_name,
        "ranking_keys": list(ranking_keys),
        "ranking": [float(v) for v in ranking],
        "path": checkpoint_name,
        "val_metrics": serialize_metric_dict(val_metrics),
    }
    topk_records = [item for item in topk_records if item["path"] != checkpoint_name]
    topk_records.append(record)
    topk_records.sort(
        key=lambda item: (
            tuple(-float(v) for v in item.get("ranking", [float(item["score"])])),
            int(item["epoch"]),
        )
    )

    while len(topk_records) > topk_limit:
        removed = topk_records.pop(-1)
        removed_path = os.path.join(save_dir, str(removed["path"]))
        if os.path.exists(removed_path):
            os.remove(removed_path)

    save_json(
        os.path.join(save_dir, "topk_checkpoints.json"),
        {
            "metric": metric_name,
            "topk_limit": int(topk_limit),
            "checkpoints": topk_records,
        },
    )
    return topk_records


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    seg_loss_fn,
    seg_weight: float,
    cls_weight: float,
    cls_class_weights: torch.Tensor | None,
    disable_cls: bool,
    cls_num_classes: int,
    epoch: int,
    total_epochs: int,
):
    model.eval()
    losses = []
    ious = []
    leak_ious = []
    cls_losses = []
    cls_correct = 0
    cls_total = 0
    cls_targets = []
    cls_preds = []
    tp = fp = tn = fn = 0
    val_pbar = tqdm(loader, desc=f"Val {epoch}/{total_epochs}", dynamic_ncols=True, leave=False, file=sys.stdout)
    for x, seg_labels, cls_labels in val_pbar:
        x = x.to(device)
        seg_labels = seg_labels.to(device)
        cls_labels = cls_labels.to(device)
        seg_probs, cls_out = model(x, return_intermediate=False)
        seg_loss = seg_loss_fn(seg_probs, seg_labels)
        if disable_cls or cls_out is None:
            cls_loss = seg_loss.new_tensor(0.0)
            loss = seg_weight * seg_loss
        else:
            cls_loss = F.cross_entropy(cls_out, cls_labels, weight=cls_class_weights)
            loss = seg_weight * seg_loss + cls_weight * cls_loss
            cls_losses.append(cls_loss.item())
            cls_pred = cls_out.argmax(dim=1)
            cls_correct += int((cls_pred == cls_labels).sum().item())
            cls_total += int(cls_labels.numel())
            cls_targets.append(cls_labels.detach().cpu().numpy())
            cls_preds.append(cls_pred.detach().cpu().numpy())
        losses.append(loss.item())
        seg_pred = seg_probs.argmax(dim=1).cpu().numpy()
        seg_tgt = seg_labels.cpu().numpy()
        per_cls = compute_iou_per_class(seg_pred, seg_tgt, num_classes=2)
        ious.append(float(np.mean(per_cls)))
        leak_ious.append(float(per_cls[1]) if len(per_cls) > 1 else 0.0)
        batch_tp, batch_fp, batch_tn, batch_fn = compute_binary_confusion(seg_pred, seg_tgt, positive_class=1)
        tp += batch_tp
        fp += batch_fp
        tn += batch_tn
        fn += batch_fn
        running_binary = compute_binary_metrics(tp, fp, tn, fn)
        val_pbar.set_postfix(
            {
                "loss": f"{np.mean(losses):.4f}",
                "iou": f"{np.mean(ious):.4f}",
                "leak_iou": f"{np.mean(leak_ious):.4f}",
                "precision": f"{running_binary['precision']:.4f}",
                "recall": f"{running_binary['recall']:.4f}",
                "f1": f"{running_binary['f1']:.4f}",
                "g_leak_iou": f"{running_binary['global_leak_iou']:.4f}",
                **({"cls_acc": f"{(cls_correct / cls_total):.4f}"} if cls_total > 0 else {}),
            }
        )
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "leak_iou": float(np.mean(leak_ious)) if leak_ious else 0.0,
    }
    metrics.update(compute_binary_metrics(tp, fp, tn, fn))
    if cls_total > 0:
        metrics["cls_loss"] = float(np.mean(cls_losses))
        metrics.update(
            compute_multiclass_classification_metrics(
                pred=np.concatenate(cls_preds, axis=0),
                target=np.concatenate(cls_targets, axis=0),
                num_classes=int(cls_num_classes),
            )
        )
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    best_metric_key = args.best_metric[4:] if args.best_metric.startswith("val_") else args.best_metric
    scheduler_milestones = resolve_scheduler_milestones(args.scheduler_milestones, args.epochs)
    if args.scheduler == "multistep" and not scheduler_milestones:
        raise ValueError("scheduler=multistep 时至少需要一个有效的 scheduler_milestones")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = LiquidLeakageDatasetTorch(
        data_path=args.train_path,
        num_points=args.num_points,
        augment=args.augment,
        normalize=True,
        augment_mode=args.augment_mode,
        binary_class=args.binary_class,
    )
    val_dataset = LiquidLeakageDatasetTorch(
        data_path=args.test_path,
        num_points=args.num_points,
        augment=False,
        normalize=True,
        augment_mode=args.augment_mode,
        binary_class=args.binary_class,
    )

    seg_class_weights = None
    if args.seg_use_class_weights:
        seg_class_weights = train_dataset.compute_seg_class_weights(
            num_classes=args.seg_classes,
            mode=args.seg_class_weight_mode,
            beta=args.seg_class_weight_beta,
        ).to(device)
    cls_class_weights = None
    if not args.disable_cls:
        cls_class_weights = compute_cls_class_weights(train_dataset, num_classes=args.cls_classes).to(device)

    sampler_stats = None
    train_sampler = None
    train_shuffle = True
    if args.train_sampler != "shuffle":
        train_sample_weights = train_dataset.compute_sample_weights(
            mode=args.train_sampler,
            alpha=args.sampler_alpha,
            min_positive_ratio=args.sampler_min_positive_ratio,
            max_weight=args.sampler_max_weight,
            empty_weight=args.sampler_empty_weight,
        )
        train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
        train_shuffle = False
        positive_mask = train_dataset.positive_ratios > 0
        positive_weights = train_sample_weights[positive_mask]
        negative_weights = train_sample_weights[~positive_mask]
        sampler_stats = {
            "positive_samples": int(positive_mask.sum()),
            "empty_samples": int((~positive_mask).sum()),
            "positive_ratio_mean": float(np.mean(train_dataset.positive_ratios)),
            "positive_ratio_median": float(np.median(train_dataset.positive_ratios)),
            "positive_weight_mean": float(positive_weights.mean().item()) if len(positive_weights) > 0 else 0.0,
            "positive_weight_max": float(positive_weights.max().item()) if len(positive_weights) > 0 else 0.0,
            "empty_weight_mean": float(negative_weights.mean().item()) if len(negative_weights) > 0 else 0.0,
        }
        print(
            "train_sampler="
            f"{args.train_sampler} positive_samples={sampler_stats['positive_samples']} "
            f"empty_samples={sampler_stats['empty_samples']} "
            f"positive_weight_mean={sampler_stats['positive_weight_mean']:.4f} "
            f"positive_weight_max={sampler_stats['positive_weight_max']:.4f} "
            f"empty_weight_mean={sampler_stats['empty_weight_mean']:.4f}",
            flush=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    seg_loss_fn = build_seg_loss_fn(args, seg_class_weights)

    model = DN_MS_LiquidNet_V2_Torch(
        in_channels=args.in_channels,
        seg_classes=args.seg_classes,
        cls_classes=args.cls_classes,
        k_scales=args.k_scales,
        use_noise_guidance=args.use_noise_guidance,
        use_progressive=args.use_progressive,
        use_noise_leak_corr=args.use_noise_leak_corr,
        use_multi_scale=args.use_multi_scale,
        use_uncertainty_fusion=args.use_uncertainty_fusion,
        use_simple_uncertainty=args.use_simple_uncertainty,
        disable_cls=args.disable_cls,
        boundary_input_mode=args.boundary_input_mode,
    ).to(device)

    args_path = os.path.join(save_dir, "args.json")
    run_args = dict(vars(args))
    run_args["timestamp"] = timestamp
    run_args["save_dir"] = save_dir
    run_args["effective_k_scales"] = [int(v) for v in model.k_scales]
    run_args["scheduler_milestones_resolved"] = scheduler_milestones
    run_args["seg_class_weights"] = None if seg_class_weights is None else [float(v) for v in seg_class_weights.detach().cpu().tolist()]
    run_args["cls_class_weights"] = None if cls_class_weights is None else [float(v) for v in cls_class_weights.detach().cpu().tolist()]
    run_args["train_positive_ratio_mean"] = float(np.mean(train_dataset.positive_ratios))
    run_args["train_positive_ratio_median"] = float(np.median(train_dataset.positive_ratios))
    run_args["train_positive_samples"] = int(np.sum(train_dataset.positive_ratios > 0))
    run_args["sampler_stats"] = sampler_stats
    save_json(args_path, run_args)
    print(f"args_path={args_path}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_iou = 0.0
    best_score = None
    best_ranking = None
    best_epoch = 0
    history = {
        "train_loss": [],
        "lr": [],
        "train_iou": [],
        "train_leak_iou": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_global_leak_iou": [],
        "train_global_bg_iou": [],
        "train_global_miou": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_iou": [],
        "val_leak_iou": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_global_leak_iou": [],
        "val_global_bg_iou": [],
        "val_global_miou": [],
        "val_accuracy": [],
        "val_epochs": [],
    }
    if not args.disable_cls:
        history["train_cls_loss"] = []
        history["train_cls_acc"] = []
        history["val_cls_loss"] = []
        history["val_cls_acc"] = []
        history["val_cls_macro_f1"] = []
        history["val_cls_balanced_acc"] = []
        history["val_cls_weighted_f1"] = []
    best_cls_acc = 0.0 if not args.disable_cls else None
    best_val_metrics = None
    topk_records: list[dict[str, object]] = []
    tie_break_metric_keys = ["global_miou", "global_leak_iou"]
    best_metric_ranking_keys = format_metric_ranking(best_metric_key, tie_break_metric_keys)

    start_time = time.time()
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", dynamic_ncols=True, file=sys.stdout)
    for epoch in epoch_bar:
        current_lr = set_optimizer_lr(optimizer, compute_epoch_lr(epoch, args, scheduler_milestones))
        history["lr"].append(current_lr)
        model.train()
        epoch_losses = []
        epoch_ious = []
        epoch_leak_ious = []
        epoch_cls_losses = []
        epoch_cls_accs = []
        epoch_tp = epoch_fp = epoch_tn = epoch_fn = 0
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", dynamic_ncols=True, leave=False, file=sys.stdout)
        for x, seg_labels, cls_labels in train_pbar:
            x = x.to(device)
            seg_labels = seg_labels.to(device)
            cls_labels = cls_labels.to(device)

            seg_probs, cls_out = model(x, return_intermediate=False)
            seg_loss = seg_loss_fn(seg_probs, seg_labels)
            if args.disable_cls or cls_out is None:
                cls_loss = seg_loss.new_tensor(0.0)
                loss = args.seg_weight * seg_loss
            else:
                cls_loss = F.cross_entropy(cls_out, cls_labels, weight=cls_class_weights)
                loss = args.seg_weight * seg_loss + args.cls_weight * cls_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            seg_pred = seg_probs.argmax(dim=1).detach().cpu().numpy()
            seg_tgt = seg_labels.detach().cpu().numpy()
            per_cls = compute_iou_per_class(seg_pred, seg_tgt, num_classes=2)
            epoch_ious.append(float(np.mean(per_cls)))
            epoch_leak_ious.append(float(per_cls[1]) if len(per_cls) > 1 else 0.0)
            batch_tp, batch_fp, batch_tn, batch_fn = compute_binary_confusion(seg_pred, seg_tgt, positive_class=1)
            epoch_tp += batch_tp
            epoch_fp += batch_fp
            epoch_tn += batch_tn
            epoch_fn += batch_fn
            running_binary = compute_binary_metrics(epoch_tp, epoch_fp, epoch_tn, epoch_fn)
            postfix = {
                "lr": f"{current_lr:.6f}",
                "loss": f"{np.mean(epoch_losses):.4f}",
                "iou": f"{np.mean(epoch_ious):.4f}",
                "leak_iou": f"{np.mean(epoch_leak_ious):.4f}",
                "precision": f"{running_binary['precision']:.4f}",
                "recall": f"{running_binary['recall']:.4f}",
                "f1": f"{running_binary['f1']:.4f}",
                "g_leak_iou": f"{running_binary['global_leak_iou']:.4f}",
            }
            if not args.disable_cls and cls_out is not None:
                cls_pred = cls_out.argmax(dim=1)
                epoch_cls_losses.append(cls_loss.item())
                epoch_cls_accs.append(float((cls_pred == cls_labels).float().mean().item()))
                postfix["cls_acc"] = f"{np.mean(epoch_cls_accs):.4f}"
            train_pbar.set_postfix(
                postfix
            )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_iou = float(np.mean(epoch_ious)) if epoch_ious else 0.0
        train_leak_iou = float(np.mean(epoch_leak_ious)) if epoch_leak_ious else 0.0
        train_binary = compute_binary_metrics(epoch_tp, epoch_fp, epoch_tn, epoch_fn)
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["train_leak_iou"].append(train_leak_iou)
        history["train_precision"].append(train_binary["precision"])
        history["train_recall"].append(train_binary["recall"])
        history["train_f1"].append(train_binary["f1"])
        history["train_global_leak_iou"].append(train_binary["global_leak_iou"])
        history["train_global_bg_iou"].append(train_binary["global_bg_iou"])
        history["train_global_miou"].append(train_binary["global_miou"])
        history["train_accuracy"].append(train_binary["accuracy"])

        postfix = {
            "lr": f"{current_lr:.6f}",
            "train_loss": f"{train_loss:.4f}",
            "train_iou": f"{train_iou:.4f}",
            "train_leak_iou": f"{train_leak_iou:.4f}",
            "train_precision": f"{train_binary['precision']:.4f}",
            "train_recall": f"{train_binary['recall']:.4f}",
            "train_f1": f"{train_binary['f1']:.4f}",
            "train_g_leak_iou": f"{train_binary['global_leak_iou']:.4f}",
        }
        msg = (
            f"epoch={epoch}"
            f" lr={current_lr:.8f}"
            f" train_loss={train_loss:.6f}"
            f" train_iou={train_iou:.6f}"
            f" train_leak_iou={train_leak_iou:.6f}"
            f" train_precision={train_binary['precision']:.6f}"
            f" train_recall={train_binary['recall']:.6f}"
            f" train_f1={train_binary['f1']:.6f}"
            f" train_global_leak_iou={train_binary['global_leak_iou']:.6f}"
        )
        if not args.disable_cls and epoch_cls_accs:
            train_cls_loss = float(np.mean(epoch_cls_losses))
            train_cls_acc = float(np.mean(epoch_cls_accs))
            history["train_cls_loss"].append(train_cls_loss)
            history["train_cls_acc"].append(train_cls_acc)
            postfix["train_cls_acc"] = f"{train_cls_acc:.4f}"
            msg += f" train_cls_loss={train_cls_loss:.6f} train_cls_acc={train_cls_acc:.6f}"

        if epoch % args.val_interval == 0:
            val_metrics = validate(
                model,
                val_loader,
                device,
                seg_loss_fn,
                args.seg_weight,
                args.cls_weight,
                cls_class_weights,
                args.disable_cls,
                args.cls_classes,
                epoch,
                args.epochs,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_iou"].append(val_metrics["iou"])
            history["val_leak_iou"].append(val_metrics["leak_iou"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_global_leak_iou"].append(val_metrics["global_leak_iou"])
            history["val_global_bg_iou"].append(val_metrics["global_bg_iou"])
            history["val_global_miou"].append(val_metrics["global_miou"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_epochs"].append(epoch)

            postfix["val_loss"] = f"{val_metrics['loss']:.4f}"
            postfix["val_iou"] = f"{val_metrics['iou']:.4f}"
            postfix["val_leak_iou"] = f"{val_metrics['leak_iou']:.4f}"
            postfix["val_precision"] = f"{val_metrics['precision']:.4f}"
            postfix["val_recall"] = f"{val_metrics['recall']:.4f}"
            postfix["val_f1"] = f"{val_metrics['f1']:.4f}"
            postfix["val_g_leak_iou"] = f"{val_metrics['global_leak_iou']:.4f}"
            if not args.disable_cls and "cls_acc" in val_metrics:
                history["val_cls_loss"].append(val_metrics["cls_loss"])
                history["val_cls_acc"].append(val_metrics["cls_acc"])
                history["val_cls_macro_f1"].append(val_metrics["cls_macro_f1"])
                history["val_cls_balanced_acc"].append(val_metrics["cls_balanced_acc"])
                history["val_cls_weighted_f1"].append(val_metrics["cls_weighted_f1"])
                postfix["val_cls_acc"] = f"{val_metrics['cls_acc']:.4f}"
                postfix["val_cls_mf1"] = f"{val_metrics['cls_macro_f1']:.4f}"
            msg += (
                f" val_loss={val_metrics['loss']:.6f}"
                f" val_iou={val_metrics['iou']:.6f}"
                f" val_leak_iou={val_metrics['leak_iou']:.6f}"
                f" val_precision={val_metrics['precision']:.6f}"
                f" val_recall={val_metrics['recall']:.6f}"
                f" val_f1={val_metrics['f1']:.6f}"
                f" val_global_leak_iou={val_metrics['global_leak_iou']:.6f}"
            )
            if not args.disable_cls and "cls_acc" in val_metrics:
                msg += (
                    f" val_cls_loss={val_metrics['cls_loss']:.6f}"
                    f" val_cls_acc={val_metrics['cls_acc']:.6f}"
                    f" val_cls_macro_f1={val_metrics['cls_macro_f1']:.6f}"
                    f" val_cls_balanced_acc={val_metrics['cls_balanced_acc']:.6f}"
                )

            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]

            selected_score = float(val_metrics[best_metric_key])
            selected_ranking = metric_ranking_tuple(
                val_metrics,
                primary_metric_key=best_metric_key,
                tie_break_metric_keys=tie_break_metric_keys,
            )
            if is_better_ranking(selected_ranking, best_ranking):
                best_score = selected_score
                best_ranking = selected_ranking
                best_epoch = epoch
                if not args.disable_cls and "cls_acc" in val_metrics:
                    best_cls_acc = val_metrics["cls_acc"]
                best_val_metrics = serialize_metric_dict(val_metrics)
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                save_json(
                    os.path.join(save_dir, "best_checkpoint.json"),
                    {
                        "epoch": int(epoch),
                        "score": float(best_score),
                        "metric": args.best_metric,
                        "ranking_keys": best_metric_ranking_keys,
                        "ranking": [float(v) for v in selected_ranking],
                        "path": "best_model.pt",
                        "val_metrics": best_val_metrics,
                    },
                )

            topk_records = update_topk_checkpoints(
                model=model,
                save_dir=save_dir,
                topk_records=topk_records,
                metric_name=args.best_metric,
                score=selected_score,
                ranking=selected_ranking,
                ranking_keys=best_metric_ranking_keys,
                epoch=epoch,
                val_metrics=val_metrics,
                topk_limit=args.topk_checkpoints,
            )

        epoch_bar.set_postfix(postfix)
        print(msg, flush=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))

    history["best_iou"] = best_iou
    history["best_score"] = None if best_score is None else float(best_score)
    history["best_ranking_keys"] = best_metric_ranking_keys
    history["best_ranking"] = None if best_ranking is None else [float(v) for v in best_ranking]
    if best_cls_acc is not None:
        history["best_cls_acc"] = best_cls_acc
    if best_val_metrics is not None:
        for key in ("cls_macro_f1", "cls_balanced_acc", "cls_weighted_f1"):
            if key in best_val_metrics:
                history[f"best_{key}"] = float(best_val_metrics[key])
    history["best_epoch"] = best_epoch
    history["best_metric"] = args.best_metric
    history["best_val_metrics"] = best_val_metrics
    history["topk_checkpoints"] = topk_records
    history["train_time_sec"] = float(time.time() - start_time)

    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"save_dir={save_dir}")
    print(
        f"best_metric={args.best_metric} "
        f"best_score={(best_score if best_score is not None else 0.0):.6f} "
        f"best_epoch={best_epoch}"
    )
    print(f"best_iou={best_iou:.6f}")
    if best_val_metrics is not None:
        print(
            "best_precision="
            f"{best_val_metrics['precision']:.6f} "
            f"best_recall={best_val_metrics['recall']:.6f} "
            f"best_f1={best_val_metrics['f1']:.6f} "
            f"best_global_leak_iou={best_val_metrics['global_leak_iou']:.6f}"
        )
        if "cls_macro_f1" in best_val_metrics:
            print(
                "best_cls_macro_f1="
                f"{best_val_metrics['cls_macro_f1']:.6f} "
                f"best_cls_balanced_acc={best_val_metrics['cls_balanced_acc']:.6f} "
                f"best_cls_weighted_f1={best_val_metrics['cls_weighted_f1']:.6f}"
            )
    if topk_records:
        print(
            "topk_checkpoints="
            + ", ".join(
                f"epoch{int(item['epoch'])}:{float(item['score']):.6f}"
                for item in topk_records
            )
        )
    if best_cls_acc is not None:
        print(f"best_cls_acc={best_cls_acc:.6f}")
    print(f"train_time_sec={history['train_time_sec']:.1f}")


if __name__ == "__main__":
    main()
