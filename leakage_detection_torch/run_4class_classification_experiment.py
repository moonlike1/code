#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 类液体类型分类实验脚本

用途：
1. 从原始 `liquid_leakage_dataset.h5` 构建干净的 4 类分类实验切分
2. 统一类别定义为：清水 / 盐水 / 混合物 / 无渗漏
3. 可选执行单次 holdout 或 k-fold 交叉验证训练
4. 自动评估最佳模型并汇总 confusion matrix 与逐类指标

示例：
    python -m leakage_detection_torch.run_4class_classification_experiment \
      --data_path /ai/0309/cloud/liquid_leakage_dataset.h5 \
      --output_dir /ai/0309/cloud/output/classification_feasibility/four_class_holdout \
      --split_mode holdout

    python -m leakage_detection_torch.run_4class_classification_experiment \
      --data_path /ai/0309/cloud/liquid_leakage_dataset.h5 \
      --output_dir /ai/0309/cloud/output/classification_feasibility/four_class_cv \
      --split_mode kfold \
      --n_splits 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from leakage_detection_torch.fusion_liquid_model_v2_torch import DN_MS_LiquidNet_V2_Torch
from leakage_detection_torch.liquid_dataset_torch import LiquidLeakageDatasetTorch


RAW_TO_DENSE_LABEL = {
    0: 0,  # 清水
    1: 1,  # 盐水
    4: 2,  # 混合物
    5: 3,  # 无渗漏
}

DENSE_CLASS_NAMES = {
    0: "清水",
    1: "盐水",
    2: "混合物",
    3: "无渗漏",
}

REUSE_SIGNATURE_FILENAME = "reuse_signature.json"


def add_toggle_arg(parser: argparse.ArgumentParser, enable_flag: str, disable_flag: str, dest: str, default: bool):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(enable_flag, dest=dest, action="store_true")
    group.add_argument(disable_flag, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args():
    parser = argparse.ArgumentParser(description="4 类液体类型分类实验")
    parser.add_argument("--data_path", type=str, default="/ai/0309/cloud/liquid_leakage_dataset.h5")
    parser.add_argument("--output_dir", type=str, default="/ai/0309/cloud/output/classification_feasibility/four_class_experiment")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="holdout",
        choices=["holdout", "kfold", "group_holdout", "group_kfold"],
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_only", action="store_true", default=False)
    parser.add_argument(
        "--group_labels_path",
        type=str,
        default="",
        help="可选外部分组标签文件，支持 .json / .txt / .csv / .npy / .npz。",
    )

    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--augment_mode", type=str, default="strong", choices=["basic", "strong"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--train_sampler", type=str, default="shuffle", choices=["shuffle", "balanced_presence", "small_positive"])
    parser.add_argument("--boundary_input_mode", type=str, default="features_fine_probs", choices=["features", "features_fine_probs"])
    add_toggle_arg(parser, "--use_noise_guidance", "--no_noise_guidance", "use_noise_guidance", True)
    add_toggle_arg(parser, "--use_progressive", "--no_progressive", "use_progressive", True)
    add_toggle_arg(parser, "--use_noise_leak_corr", "--no_noise_leak_corr", "use_noise_leak_corr", True)
    add_toggle_arg(parser, "--use_multi_scale", "--no_multi_scale", "use_multi_scale", True)
    add_toggle_arg(parser, "--use_uncertainty_fusion", "--no_uncertainty_fusion", "use_uncertainty_fusion", True)
    parser.add_argument("--use_simple_uncertainty", action="store_true", default=False)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_cls_macro_f1",
        choices=[
            "val_cls_macro_f1",
            "val_cls_balanced_acc",
            "val_cls_weighted_f1",
            "val_cls_acc",
            "val_global_leak_iou",
            "val_f1",
        ],
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_dataset(data_path: Path):
    with h5py.File(data_path, "r") as f:
        arrays = {k: f[k][:] for k in f.keys()}
        attrs = {}
        for key, value in f.attrs.items():
            try:
                attrs[key] = value.tolist() if hasattr(value, "tolist") else value
            except Exception:
                attrs[key] = str(value)
    return arrays, attrs


def resolve_optional_path(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve()) if path_str else ""


def load_group_labels_from_path(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "groups" in payload:
                return np.asarray(payload["groups"])
            if "group_labels" in payload:
                return np.asarray(payload["group_labels"])
            raise KeyError("JSON 中未找到 groups 或 group_labels 字段。")
        return np.asarray(payload)

    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=True))

    if suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        if "groups" in payload:
            return np.asarray(payload["groups"])
        if "group_labels" in payload:
            return np.asarray(payload["group_labels"])
        first_key = list(payload.keys())[0]
        return np.asarray(payload[first_key])

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if lines and "," in lines[0]:
        header = [item.strip() for item in lines[0].split(",")]
        if "group" in header:
            group_idx = header.index("group")
            return np.asarray([line.split(",")[group_idx].strip() for line in lines[1:]])
        return np.asarray([line.split(",")[-1].strip() for line in lines[1:]])
    return np.asarray(lines)


def load_group_labels(group_labels_path: str, arrays: dict, num_samples: int) -> np.ndarray | None:
    groups = None

    if group_labels_path:
        path = Path(group_labels_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"找不到 group labels 文件: {path}")
        groups = load_group_labels_from_path(path)

    if groups is None:
        for key in ("group_labels", "group_ids", "groups", "session_ids", "source_ids", "record_ids"):
            if key in arrays:
                groups = np.asarray(arrays[key])
                break

    if groups is None:
        return None

    groups = np.asarray(groups).reshape(-1)
    if len(groups) != int(num_samples):
        raise ValueError(f"group labels 数量不匹配：期望 {num_samples}，实际 {len(groups)}。")
    return groups


def remap_four_class_labels(raw_labels: np.ndarray) -> np.ndarray:
    raw_labels = np.asarray(raw_labels).astype(np.int64)
    unique_labels = sorted(int(v) for v in np.unique(raw_labels))
    invalid = [v for v in unique_labels if v not in RAW_TO_DENSE_LABEL]
    if invalid:
        raise ValueError(f"发现当前 4 类任务未定义的原始标签: {invalid}")
    mapper = np.vectorize(lambda x: RAW_TO_DENSE_LABEL[int(x)], otypes=[np.int64])
    return mapper(raw_labels)


def build_split_iterator(labels: np.ndarray, args, groups: np.ndarray | None = None):
    dummy_x = np.zeros(len(labels), dtype=np.float32)
    if args.split_mode == "holdout":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        for split_id, (train_idx, val_idx) in enumerate(splitter.split(dummy_x, labels), start=1):
            yield split_id, train_idx, val_idx
        return

    if args.split_mode == "kfold":
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        for split_id, (train_idx, val_idx) in enumerate(splitter.split(dummy_x, labels), start=1):
            yield split_id, train_idx, val_idx
        return

    if groups is None:
        raise ValueError(
            f"split_mode={args.split_mode} 需要 group labels。请通过 --group_labels_path 提供外部分组文件，"
            "或者在母数据集中补充 group_labels / group_ids / session_ids 字段。"
        )

    if args.split_mode == "group_holdout":
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        for split_id, (train_idx, val_idx) in enumerate(
            splitter.split(dummy_x, labels, groups=groups),
            start=1,
        ):
            yield split_id, train_idx, val_idx
        return

    if args.split_mode == "group_kfold":
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
            split_iter = splitter.split(dummy_x, labels, groups=groups)
        else:
            splitter = GroupKFold(n_splits=args.n_splits)
            split_iter = splitter.split(dummy_x, labels, groups=groups)
        for split_id, (train_idx, val_idx) in enumerate(split_iter, start=1):
            yield split_id, train_idx, val_idx
        return

    raise ValueError(f"未支持的 split_mode: {args.split_mode}")


def write_split_h5(
    out_path: Path,
    arrays: dict,
    attrs: dict,
    indices: np.ndarray,
    split_name: str,
    split_id: int,
    group_labels: np.ndarray | None = None,
) -> None:
    point_clouds = arrays["point_clouds"][indices]
    seg_labels = arrays["seg_labels"][indices]
    cls_labels_raw = arrays["cls_labels"][indices].astype(np.int64)
    cls_labels_dense = remap_four_class_labels(cls_labels_raw)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("point_clouds", data=point_clouds, compression="gzip")
        f.create_dataset("seg_labels", data=seg_labels, compression="gzip")
        f.create_dataset("cls_labels", data=cls_labels_dense, compression="gzip")
        f.create_dataset("cls_labels_raw", data=cls_labels_raw, compression="gzip")
        f.create_dataset("source_indices", data=indices.astype(np.int32), compression="gzip")
        if group_labels is not None:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            group_values = np.asarray(group_labels[indices], dtype=object)
            f.create_dataset("group_labels", data=group_values, dtype=string_dtype, compression="gzip")
        for key, value in attrs.items():
            try:
                f.attrs[key] = value
            except Exception:
                f.attrs[key] = str(value)
        f.attrs["classification_task"] = "4-class-liquid-type"
        f.attrs["dense_class_names_json"] = json.dumps(DENSE_CLASS_NAMES, ensure_ascii=False)
        f.attrs["raw_to_dense_label_map_json"] = json.dumps(RAW_TO_DENSE_LABEL, ensure_ascii=False)
        f.attrs["split_name"] = split_name
        f.attrs["split_id"] = int(split_id)


def summarize_split(indices: np.ndarray, raw_labels: np.ndarray, group_labels: np.ndarray | None = None):
    dense_labels = remap_four_class_labels(raw_labels[indices])
    unique_labels, counts = np.unique(dense_labels, return_counts=True)
    summary = {
        "num_samples": int(len(indices)),
        "class_counts_dense": {int(k): int(v) for k, v in zip(unique_labels, counts)},
        "class_names_dense": DENSE_CLASS_NAMES,
    }
    if group_labels is not None:
        summary["num_groups"] = int(len(np.unique(group_labels[indices])))
    return summary


def build_train_command(args, train_path: Path, val_path: Path, fold_output_dir: Path):
    cmd = [
        sys.executable,
        "-m",
        "leakage_detection_torch.train_v2_torch",
        "--train_path",
        str(train_path),
        "--test_path",
        str(val_path),
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
        "--seg_weight",
        str(args.seg_weight),
        "--cls_weight",
        str(args.cls_weight),
        "--no_binary_class",
        "--cls_classes",
        "4",
        "--device",
        str(args.device),
        "--output_dir",
        str(fold_output_dir),
        "--val_interval",
        str(args.val_interval),
        "--train_sampler",
        str(args.train_sampler),
        "--boundary_input_mode",
        str(args.boundary_input_mode),
        "--best_metric",
        str(args.best_metric),
    ]
    if args.augment:
        cmd.extend(["--augment", "--augment_mode", str(args.augment_mode)])
    if not args.use_noise_guidance:
        cmd.append("--no_noise_guidance")
    if not args.use_progressive:
        cmd.append("--no_progressive")
    if not args.use_noise_leak_corr:
        cmd.append("--no_noise_leak_corr")
    if not args.use_multi_scale:
        cmd.append("--no_multi_scale")
    if not args.use_uncertainty_fusion:
        cmd.append("--no_uncertainty_fusion")
    if args.use_simple_uncertainty:
        cmd.append("--use_simple_uncertainty")
    return cmd


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reuse_signature(args, train_path: Path, val_path: Path) -> dict:
    signature = {
        "group_labels_path": resolve_optional_path(getattr(args, "group_labels_path", "")),
        "train_path": str(train_path),
        "test_path": str(val_path),
        "train_sha256": compute_file_sha256(train_path),
        "test_sha256": compute_file_sha256(val_path),
    }
    model_name = getattr(args, "model_name", None)
    if model_name is not None:
        signature["model_name"] = str(model_name)
    return signature


def load_reuse_signature(run_dir: Path) -> dict | None:
    signature_path = run_dir / REUSE_SIGNATURE_FILENAME
    if not signature_path.exists():
        return None
    return load_json(signature_path)


def save_reuse_signature(run_dir: Path, args, train_path: Path, val_path: Path) -> None:
    save_json(run_dir / REUSE_SIGNATURE_FILENAME, build_reuse_signature(args, train_path, val_path))


def is_complete_training_run(run_dir: Path) -> bool:
    required_files = ["args.json", "history.json", "best_model.pt"]
    return run_dir.is_dir() and all((run_dir / file_name).exists() for file_name in required_files)


def run_matches_request(run_dir: Path, args, train_path: Path, val_path: Path) -> bool:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return False

    saved_args = load_json(args_path)
    expected_pairs = {
        "train_path": str(train_path),
        "test_path": str(val_path),
        "num_points": int(args.num_points),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seg_weight": float(args.seg_weight),
        "cls_weight": float(args.cls_weight),
        "val_interval": int(args.val_interval),
        "train_sampler": str(args.train_sampler),
        "boundary_input_mode": str(args.boundary_input_mode),
        "best_metric": str(args.best_metric),
        "augment": bool(args.augment),
        "augment_mode": str(args.augment_mode),
        "binary_class": False,
        "cls_classes": 4,
    }
    for key, expected_value in expected_pairs.items():
        actual_value = saved_args.get(key)
        if actual_value != expected_value:
            return False

    saved_signature = load_reuse_signature(run_dir)
    if saved_signature is not None:
        return saved_signature == build_reuse_signature(args, train_path, val_path)

    if resolve_optional_path(getattr(args, "group_labels_path", "")):
        return False
    return True


def find_latest_run_dir(
    training_root: Path,
    args=None,
    train_path: Path | None = None,
    val_path: Path | None = None,
    require_complete: bool = False,
) -> Path:
    run_dirs = [p for p in training_root.iterdir() if p.is_dir()]
    if require_complete:
        run_dirs = [p for p in run_dirs if is_complete_training_run(p)]
    if args is not None and train_path is not None and val_path is not None:
        run_dirs = [p for p in run_dirs if run_matches_request(p, args, train_path, val_path)]
    if not run_dirs:
        raise FileNotFoundError(f"训练目录下未找到符合条件的运行结果: {training_root}")
    return sorted(run_dirs, key=lambda p: p.name)[-1]


def load_training_summary(run_dir: Path) -> dict:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return {}

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


def evaluate_best_model(run_dir: Path):
    args_path = run_dir / "args.json"
    with open(args_path, "r", encoding="utf-8") as f:
        run_args = json.load(f)

    model = DN_MS_LiquidNet_V2_Torch(
        in_channels=int(run_args["in_channels"]),
        seg_classes=int(run_args["seg_classes"]),
        cls_classes=int(run_args["cls_classes"]),
        k_scales=[int(v) for v in run_args["effective_k_scales"]],
        use_noise_guidance=bool(run_args["use_noise_guidance"]),
        use_progressive=bool(run_args["use_progressive"]),
        use_noise_leak_corr=bool(run_args["use_noise_leak_corr"]),
        use_multi_scale=bool(run_args["use_multi_scale"]),
        use_uncertainty_fusion=bool(run_args["use_uncertainty_fusion"]),
        use_simple_uncertainty=bool(run_args["use_simple_uncertainty"]),
        disable_cls=bool(run_args["disable_cls"]),
        boundary_input_mode=str(run_args["boundary_input_mode"]),
    )
    state_dict = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dataset = LiquidLeakageDatasetTorch(
        data_path=run_args["test_path"],
        num_points=int(run_args["num_points"]),
        augment=False,
        normalize=True,
        augment_mode=str(run_args["augment_mode"]),
        binary_class=bool(run_args["binary_class"]),
    )
    loader = DataLoader(dataset, batch_size=int(run_args["batch_size"]), shuffle=False, num_workers=0)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, _, cls_labels in loader:
            _, cls_out = model(x, return_intermediate=False)
            pred = cls_out.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(cls_labels.numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = list(DENSE_CLASS_NAMES.keys())
    target_names = [DENSE_CLASS_NAMES[k] for k in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    eval_summary = {
        "run_dir": str(run_dir),
        "num_val_samples": int(len(y_true)),
        "overall_accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(report["macro avg"]["recall"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "dense_class_names": DENSE_CLASS_NAMES,
        "classification_report": report,
    }
    save_json(run_dir / "classification_eval_best_model.json", eval_summary)
    return eval_summary


def aggregate_fold_results(fold_results: list[dict]):
    if not fold_results:
        return {}
    accs = [float(item["eval"]["overall_accuracy"]) for item in fold_results]
    balanced_accs = [float(item["eval"]["balanced_accuracy"]) for item in fold_results]
    macro_f1s = [float(item["eval"]["classification_report"]["macro avg"]["f1-score"]) for item in fold_results]
    weighted_f1s = [float(item["eval"]["classification_report"]["weighted avg"]["f1-score"]) for item in fold_results]
    aggregate = {
        "num_folds": int(len(fold_results)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(balanced_accs)),
        "balanced_accuracy_std": float(np.std(balanced_accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_std": float(np.std(weighted_f1s)),
    }
    metric_key_map = {
        "global_miou": "global_miou",
        "global_leak_iou": "global_leak_iou",
        "precision": "seg_precision",
        "recall": "seg_recall",
        "f1": "seg_f1",
    }
    for training_key, aggregate_prefix in metric_key_map.items():
        values = []
        for item in fold_results:
            best_val_metrics = item.get("training", {}).get("best_val_metrics", {})
            if training_key in best_val_metrics:
                values.append(float(best_val_metrics[training_key]))
        if values:
            aggregate[f"{aggregate_prefix}_mean"] = float(np.mean(values))
            aggregate[f"{aggregate_prefix}_std"] = float(np.std(values))

    best_epochs = [
        float(item.get("training", {}).get("best_epoch"))
        for item in fold_results
        if item.get("training", {}).get("best_epoch") is not None
    ]
    if best_epochs:
        aggregate["best_epoch_mean"] = float(np.mean(best_epochs))
        aggregate["best_epoch_std"] = float(np.std(best_epochs))
    return aggregate


def main():
    args = parse_args()
    root_dir = Path(args.output_dir).resolve()
    ensure_dir(root_dir)

    raw_data_path = Path(args.data_path).resolve()
    arrays, attrs = load_raw_dataset(raw_data_path)
    raw_labels = arrays["cls_labels"].astype(np.int64)
    remap_four_class_labels(raw_labels)
    group_labels = load_group_labels(
        group_labels_path=args.group_labels_path,
        arrays=arrays,
        num_samples=len(raw_labels),
    )

    config = {
        "task_name": "4-class-liquid-type-classification",
        "data_path": str(raw_data_path),
        "split_mode": args.split_mode,
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
            "seg_weight": float(args.seg_weight),
            "cls_weight": float(args.cls_weight),
            "augment": bool(args.augment),
            "augment_mode": str(args.augment_mode),
            "device": str(args.device),
            "val_interval": int(args.val_interval),
            "train_sampler": str(args.train_sampler),
            "best_metric": str(args.best_metric),
            "use_noise_guidance": bool(args.use_noise_guidance),
            "use_progressive": bool(args.use_progressive),
            "use_noise_leak_corr": bool(args.use_noise_leak_corr),
            "use_multi_scale": bool(args.use_multi_scale),
            "use_uncertainty_fusion": bool(args.use_uncertainty_fusion),
            "use_simple_uncertainty": bool(args.use_simple_uncertainty),
        },
    }
    save_json(root_dir / "experiment_config.json", config)

    fold_results = []
    splits_root = root_dir / "splits"
    ensure_dir(splits_root)

    for split_id, train_idx, val_idx in build_split_iterator(raw_labels, args, groups=group_labels):
        fold_name = "holdout" if args.split_mode in {"holdout", "group_holdout"} else f"fold_{split_id:02d}"
        fold_dir = root_dir / fold_name
        ensure_dir(fold_dir)

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
            ensure_dir(training_root)
            reused_existing = True
            try:
                run_dir = find_latest_run_dir(
                    training_root,
                    args=args,
                    train_path=train_path,
                    val_path=val_path,
                    require_complete=True,
                )
                print(f"复用已完成训练结果: {run_dir}", flush=True)
            except FileNotFoundError:
                reused_existing = False
                cmd = build_train_command(args, train_path, val_path, training_root)
                print("运行命令:", " ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), check=True)
                run_dir = find_latest_run_dir(
                    training_root,
                    args=args,
                    train_path=train_path,
                    val_path=val_path,
                    require_complete=True,
                )
            if not reused_existing:
                save_reuse_signature(run_dir, args, train_path, val_path)

            eval_path = run_dir / "classification_eval_best_model.json"
            if eval_path.exists():
                eval_summary = load_json(eval_path)
            else:
                eval_summary = evaluate_best_model(run_dir)
            fold_record["run_dir"] = str(run_dir)
            fold_record["eval"] = eval_summary
            fold_record["training"] = load_training_summary(run_dir)

        fold_results.append(fold_record)

        if args.split_mode in {"holdout", "group_holdout"}:
            break

    summary = {
        "task_name": "4-class-liquid-type-classification",
        "raw_data_path": str(raw_data_path),
        "split_mode": args.split_mode,
        "prepare_only": bool(args.prepare_only),
        "dense_class_names": DENSE_CLASS_NAMES,
        "raw_to_dense_label_map": RAW_TO_DENSE_LABEL,
        "fold_results": fold_results,
        "aggregate_metrics": aggregate_fold_results(fold_results) if not args.prepare_only else {},
    }
    save_json(root_dir / "experiment_summary.json", summary)
    print(json.dumps(summary["aggregate_metrics"], ensure_ascii=False, indent=2))
    print(f"summary_path={root_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    main()
