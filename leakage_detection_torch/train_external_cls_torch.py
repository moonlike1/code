#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakage_detection_torch.external_cls_models import build_external_cls_model
from leakage_detection_torch.liquid_dataset_torch import LiquidLeakageDatasetTorch


DENSE_CLASS_NAMES = {
    0: "清水",
    1: "盐水",
    2: "混合物",
    3: "无渗漏",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外部分类 baseline 单次训练")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["pointnet", "pointnet2", "dgcnn"])
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--augment_mode", type=str, default="strong", choices=["basic", "strong"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_cls_macro_f1",
        choices=["val_cls_macro_f1", "val_cls_balanced_acc", "val_cls_weighted_f1", "val_cls_acc"],
    )
    return parser.parse_args()


def save_json(path: str | Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_cls_class_weights(dataset, num_classes: int) -> torch.Tensor:
    labels = dataset.cls_labels.astype(np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * float(len(counts))
    return torch.tensor(weights, dtype=torch.float32)


def compute_multiclass_classification_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    valid = (pred >= 0) & (pred < int(num_classes)) & (target >= 0) & (target < int(num_classes))
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
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    total = float(confusion.sum())
    return {
        "cls_acc": float(tp.sum() / total) if total > 0 else 0.0,
        "cls_macro_precision": float(precision.mean()) if precision.size > 0 else 0.0,
        "cls_macro_recall": float(recall.mean()) if recall.size > 0 else 0.0,
        "cls_macro_f1": float(f1.mean()) if f1.size > 0 else 0.0,
        "cls_balanced_acc": float(recall.mean()) if recall.size > 0 else 0.0,
        "cls_weighted_f1": float(np.average(f1, weights=target_count)) if target_count.sum() > 0 else 0.0,
    }


def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cls_class_weights: torch.Tensor | None,
    num_classes: int,
) -> dict[str, object]:
    model.eval()
    losses = []
    all_pred = []
    all_target = []
    with torch.no_grad():
        for x, _, cls_labels in loader:
            x = x.to(device)
            cls_labels = cls_labels.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, cls_labels, weight=cls_class_weights)
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            all_pred.append(pred)
            all_target.append(cls_labels.detach().cpu().numpy())

    pred = np.concatenate(all_pred, axis=0) if all_pred else np.empty((0,), dtype=np.int64)
    target = np.concatenate(all_target, axis=0) if all_target else np.empty((0,), dtype=np.int64)
    metrics = compute_multiclass_classification_metrics(pred=pred, target=target, num_classes=num_classes)
    metrics["cls_loss"] = float(np.mean(losses)) if losses else 0.0
    labels = list(range(int(num_classes)))
    target_names = [DENSE_CLASS_NAMES[idx] for idx in labels]
    report = classification_report(target, pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(target, pred, labels=labels)
    return {
        "metrics": metrics,
        "pred": pred,
        "target": target,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def ranking_tuple(metrics: dict[str, float], best_metric: str) -> tuple[float, float, float, float]:
    key = best_metric[4:] if best_metric.startswith("val_") else best_metric
    return (
        float(metrics.get(key, float("-inf"))),
        float(metrics.get("cls_balanced_acc", float("-inf"))),
        float(metrics.get("cls_acc", float("-inf"))),
        -float(metrics.get("cls_loss", float("inf"))),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = LiquidLeakageDatasetTorch(
        data_path=args.train_path,
        num_points=args.num_points,
        augment=bool(args.augment),
        normalize=True,
        augment_mode=args.augment_mode,
        binary_class=False,
    )
    val_dataset = LiquidLeakageDatasetTorch(
        data_path=args.test_path,
        num_points=args.num_points,
        augment=False,
        normalize=True,
        augment_mode=args.augment_mode,
        binary_class=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cls_class_weights = compute_cls_class_weights(train_dataset, num_classes=4).to(device)
    model = build_external_cls_model(args.model_name, in_channels=4, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args_payload = dict(vars(args))
    args_payload["timestamp"] = timestamp
    args_payload["save_dir"] = str(save_dir)
    args_payload["cls_classes"] = 4
    args_payload["cls_class_weights"] = [float(v) for v in cls_class_weights.detach().cpu().tolist()]
    args_payload["train_positive_ratio_mean"] = float(np.mean(train_dataset.positive_ratios))
    args_payload["train_positive_ratio_median"] = float(np.median(train_dataset.positive_ratios))
    save_json(save_dir / "args.json", args_payload)

    history: dict[str, object] = {
        "train_loss": [],
        "train_cls_acc": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_cls_acc": [],
        "val_cls_macro_f1": [],
        "val_cls_balanced_acc": [],
        "val_cls_weighted_f1": [],
        "val_epochs": [],
    }

    best_ranking = None
    best_score = None
    best_epoch = 0
    best_val_metrics = None
    start_time = time.time()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", dynamic_ncols=True, file=sys.stdout)
    for epoch in epoch_bar:
        model.train()
        batch_losses = []
        batch_accs = []
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", dynamic_ncols=True, leave=False, file=sys.stdout)
        for x, _, cls_labels in train_pbar:
            x = x.to(device)
            cls_labels = cls_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, cls_labels, weight=cls_class_weights)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            acc = float((pred == cls_labels).float().mean().item())
            batch_losses.append(float(loss.item()))
            batch_accs.append(acc)
            train_pbar.set_postfix({"loss": f"{np.mean(batch_losses):.4f}", "cls_acc": f"{np.mean(batch_accs):.4f}"})

        history["train_loss"].append(float(np.mean(batch_losses)) if batch_losses else 0.0)
        history["train_cls_acc"].append(float(np.mean(batch_accs)) if batch_accs else 0.0)

        if epoch % args.val_interval != 0 and epoch != args.epochs:
            epoch_bar.set_postfix({"train_loss": f"{history['train_loss'][-1]:.4f}", "train_cls_acc": f"{history['train_cls_acc'][-1]:.4f}"})
            continue

        eval_result = evaluate_classifier(model, val_loader, device, cls_class_weights, num_classes=4)
        val_metrics = eval_result["metrics"]

        history["val_epochs"].append(int(epoch))
        history["val_loss"].append(float(val_metrics["cls_loss"]))
        history["val_cls_loss"].append(float(val_metrics["cls_loss"]))
        history["val_cls_acc"].append(float(val_metrics["cls_acc"]))
        history["val_cls_macro_f1"].append(float(val_metrics["cls_macro_f1"]))
        history["val_cls_balanced_acc"].append(float(val_metrics["cls_balanced_acc"]))
        history["val_cls_weighted_f1"].append(float(val_metrics["cls_weighted_f1"]))

        current_ranking = ranking_tuple(val_metrics, args.best_metric)
        if best_ranking is None or current_ranking > best_ranking:
            best_ranking = current_ranking
            best_score = float(current_ranking[0])
            best_epoch = int(epoch)
            best_val_metrics = dict(val_metrics)
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            save_json(
                save_dir / "best_checkpoint.json",
                {
                    "epoch": best_epoch,
                    "metric": args.best_metric,
                    "score": best_score,
                    "ranking_keys": ["cls_macro_f1", "cls_balanced_acc", "cls_acc", "neg_cls_loss"],
                    "ranking": [float(v) for v in current_ranking],
                    "val_metrics": best_val_metrics,
                },
            )

        epoch_bar.set_postfix(
            {
                "train_loss": f"{history['train_loss'][-1]:.4f}",
                "train_cls_acc": f"{history['train_cls_acc'][-1]:.4f}",
                "val_cls_acc": f"{val_metrics['cls_acc']:.4f}",
                "val_cls_mf1": f"{val_metrics['cls_macro_f1']:.4f}",
            }
        )

    torch.save(model.state_dict(), save_dir / "final_model.pt")

    best_model = build_external_cls_model(args.model_name, in_channels=4, num_classes=4).to(device)
    best_model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))
    final_eval = evaluate_classifier(best_model, val_loader, device, cls_class_weights, num_classes=4)
    eval_summary = {
        "run_dir": str(save_dir),
        "num_val_samples": int(len(final_eval["target"])),
        "overall_accuracy": float(final_eval["metrics"]["cls_acc"]),
        "balanced_accuracy": float(final_eval["metrics"]["cls_balanced_acc"]),
        "macro_precision": float(final_eval["metrics"]["cls_macro_precision"]),
        "macro_recall": float(final_eval["metrics"]["cls_macro_recall"]),
        "macro_f1": float(final_eval["metrics"]["cls_macro_f1"]),
        "weighted_f1": float(final_eval["metrics"]["cls_weighted_f1"]),
        "confusion_matrix": final_eval["confusion_matrix"],
        "dense_class_names": DENSE_CLASS_NAMES,
        "classification_report": final_eval["classification_report"],
    }
    save_json(save_dir / "classification_eval_best_model.json", eval_summary)

    history["best_epoch"] = int(best_epoch)
    history["best_metric"] = args.best_metric
    history["best_score"] = float(best_score) if best_score is not None else None
    history["best_ranking_keys"] = ["cls_macro_f1", "cls_balanced_acc", "cls_acc", "neg_cls_loss"]
    history["best_ranking"] = [float(v) for v in best_ranking] if best_ranking is not None else None
    history["best_val_metrics"] = best_val_metrics or final_eval["metrics"]
    history["training_time_sec"] = float(time.time() - start_time)
    save_json(save_dir / "history.json", history)

    print(f"args_path={save_dir / 'args.json'}", flush=True)
    print(f"summary_path={save_dir / 'classification_eval_best_model.json'}", flush=True)


if __name__ == "__main__":
    main()
