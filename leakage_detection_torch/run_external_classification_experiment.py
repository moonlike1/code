#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from leakage_detection_torch.run_4class_classification_experiment import (
    DENSE_CLASS_NAMES,
    RAW_TO_DENSE_LABEL,
    build_reuse_signature,
    build_split_iterator,
    load_group_labels,
    load_json,
    load_reuse_signature,
    load_raw_dataset,
    remap_four_class_labels,
    save_json,
    save_reuse_signature,
    summarize_split,
    write_split_h5,
    resolve_optional_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外部分类 baseline 实验脚本")
    parser.add_argument("--data_path", type=str, default="/ai/0309/cloud/liquid_leakage_dataset_with_groups.h5")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["pointnet", "pointnet2", "dgcnn"])
    parser.add_argument(
        "--split_mode",
        type=str,
        default="group_kfold",
        choices=["holdout", "kfold", "group_holdout", "group_kfold"],
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_only", action="store_true", default=False)
    parser.add_argument("--group_labels_path", type=str, default="")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--augment_mode", type=str, default="strong", choices=["basic", "strong"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_cls_macro_f1",
        choices=["val_cls_macro_f1", "val_cls_balanced_acc", "val_cls_weighted_f1", "val_cls_acc"],
    )
    return parser.parse_args()


def is_complete_training_run(run_dir: Path) -> bool:
    required_files = ["args.json", "history.json", "best_model.pt", "classification_eval_best_model.json"]
    return run_dir.is_dir() and all((run_dir / file_name).exists() for file_name in required_files)


def run_matches_request(run_dir: Path, args, train_path: Path, val_path: Path) -> bool:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return False
    payload = load_json(args_path)
    expected = {
        "train_path": str(train_path),
        "test_path": str(val_path),
        "model_name": str(args.model_name),
        "num_points": int(args.num_points),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "augment": bool(args.augment),
        "augment_mode": str(args.augment_mode),
        "device": str(args.device),
        "val_interval": int(args.val_interval),
        "best_metric": str(args.best_metric),
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            return False

    saved_signature = load_reuse_signature(run_dir)
    if saved_signature is not None:
        return saved_signature == build_reuse_signature(args, train_path, val_path)

    if resolve_optional_path(getattr(args, "group_labels_path", "")):
        return False
    return True


def find_latest_run_dir(training_root: Path, args=None, train_path: Path | None = None, val_path: Path | None = None) -> Path:
    candidates = [p for p in training_root.iterdir() if is_complete_training_run(p)]
    if args is not None and train_path is not None and val_path is not None:
        candidates = [p for p in candidates if run_matches_request(p, args, train_path, val_path)]
    if not candidates:
        raise FileNotFoundError(f"训练目录下未找到符合条件的运行结果: {training_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def build_train_command(args, train_path: Path, val_path: Path, training_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "leakage_detection_torch.train_external_cls_torch",
        "--train_path",
        str(train_path),
        "--test_path",
        str(val_path),
        "--output_dir",
        str(training_root),
        "--model_name",
        str(args.model_name),
        "--num_points",
        str(args.num_points),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--device",
        str(args.device),
        "--val_interval",
        str(args.val_interval),
        "--best_metric",
        str(args.best_metric),
        "--seed",
        str(args.seed),
    ]
    if args.augment:
        cmd.extend(["--augment", "--augment_mode", str(args.augment_mode)])
    return cmd


def load_training_summary(run_dir: Path) -> dict:
    history_path = run_dir / "history.json"
    history = load_json(history_path)
    return {
        "run_dir": str(run_dir),
        "best_epoch": int(history.get("best_epoch", 0)),
        "best_metric": history.get("best_metric"),
        "best_score": float(history.get("best_score")) if history.get("best_score") is not None else None,
        "best_ranking_keys": history.get("best_ranking_keys"),
        "best_ranking": history.get("best_ranking"),
        "best_val_metrics": history.get("best_val_metrics", {}) or {},
    }


def aggregate_fold_results(fold_results: list[dict]) -> dict:
    if not fold_results:
        return {}
    accs = [float(item["eval"]["overall_accuracy"]) for item in fold_results]
    balanced_accs = [float(item["eval"]["balanced_accuracy"]) for item in fold_results]
    macro_f1s = [float(item["eval"]["classification_report"]["macro avg"]["f1-score"]) for item in fold_results]
    weighted_f1s = [float(item["eval"]["classification_report"]["weighted avg"]["f1-score"]) for item in fold_results]
    best_epochs = [float(item["training"]["best_epoch"]) for item in fold_results if item.get("training", {}).get("best_epoch") is not None]
    return {
        "num_folds": int(len(fold_results)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(balanced_accs)),
        "balanced_accuracy_std": float(np.std(balanced_accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
        "best_epoch_mean": float(np.mean(best_epochs)) if best_epochs else None,
        "best_epoch_std": float(np.std(best_epochs)) if best_epochs else None,
    }


def main() -> None:
    args = parse_args()
    root_dir = Path(args.output_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    raw_data_path = Path(args.data_path).resolve()
    arrays, attrs = load_raw_dataset(raw_data_path)
    raw_labels = arrays["cls_labels"].astype(np.int64)
    remap_four_class_labels(raw_labels)
    group_labels = load_group_labels(args.group_labels_path, arrays, len(raw_labels))

    config = {
        "task_name": "external-4-class-liquid-type-classification",
        "data_path": str(raw_data_path),
        "model_name": str(args.model_name),
        "split_mode": str(args.split_mode),
        "test_size": float(args.test_size),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "group_labels_path": str(Path(args.group_labels_path).resolve()) if args.group_labels_path else "",
        "dense_class_names": DENSE_CLASS_NAMES,
        "raw_to_dense_label_map": RAW_TO_DENSE_LABEL,
        "train_args": {
            "num_points": int(args.num_points),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "augment": bool(args.augment),
            "augment_mode": str(args.augment_mode),
            "device": str(args.device),
            "val_interval": int(args.val_interval),
            "best_metric": str(args.best_metric),
        },
    }
    save_json(root_dir / "experiment_config.json", config)

    fold_results = []
    splits_root = root_dir / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)

    for split_id, train_idx, val_idx in build_split_iterator(raw_labels, args, groups=group_labels):
        fold_name = "holdout" if args.split_mode in {"holdout", "group_holdout"} else f"fold_{split_id:02d}"
        fold_dir = root_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_path = splits_root / f"{fold_name}_train.h5"
        val_path = splits_root / f"{fold_name}_val.h5"
        write_split_h5(train_path, arrays, attrs, train_idx, "train", split_id, group_labels=group_labels)
        write_split_h5(val_path, arrays, attrs, val_idx, "val", split_id, group_labels=group_labels)

        split_summary = {
            "fold_name": fold_name,
            "split_id": int(split_id),
            "train": summarize_split(train_idx, raw_labels, group_labels=group_labels),
            "val": summarize_split(val_idx, raw_labels, group_labels=group_labels),
            "train_path": str(train_path),
            "val_path": str(val_path),
        }
        save_json(fold_dir / "split_summary.json", split_summary)

        fold_record = {
            "fold_name": fold_name,
            "split_id": int(split_id),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "split_summary": split_summary,
        }

        if not args.prepare_only:
            training_root = fold_dir / "training"
            training_root.mkdir(parents=True, exist_ok=True)
            reused_existing = True
            try:
                run_dir = find_latest_run_dir(training_root, args=args, train_path=train_path, val_path=val_path)
                print(f"复用已完成训练结果: {run_dir}", flush=True)
            except FileNotFoundError:
                reused_existing = False
                cmd = build_train_command(args, train_path, val_path, training_root)
                print("运行命令:", " ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), check=True)
                run_dir = find_latest_run_dir(training_root, args=args, train_path=train_path, val_path=val_path)
            if not reused_existing:
                save_reuse_signature(run_dir, args, train_path, val_path)
            fold_record["run_dir"] = str(run_dir)
            fold_record["eval"] = load_json(run_dir / "classification_eval_best_model.json")
            fold_record["training"] = load_training_summary(run_dir)

        fold_results.append(fold_record)
        if args.split_mode in {"holdout", "group_holdout"}:
            break

    summary = {
        "task_name": "external-4-class-liquid-type-classification",
        "raw_data_path": str(raw_data_path),
        "model_name": str(args.model_name),
        "split_mode": str(args.split_mode),
        "prepare_only": bool(args.prepare_only),
        "dense_class_names": DENSE_CLASS_NAMES,
        "raw_to_dense_label_map": RAW_TO_DENSE_LABEL,
        "fold_results": fold_results,
        "aggregate_metrics": aggregate_fold_results(fold_results) if not args.prepare_only else {},
    }
    save_json(root_dir / "experiment_summary.json", summary)
    print(summary["aggregate_metrics"])
    print(f"summary_path={root_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
