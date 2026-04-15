#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from leakage_detection_torch.export_predicted_pointclouds import (
    build_block_stem,
    choose_device,
    compute_bbox,
    export_one_block,
    infer_predictions,
    load_dataset,
    load_model,
    load_run_args,
    normalize_point_cloud,
    select_indices,
)


try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - runtime fallback only
    cKDTree = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense block inference by voting on source-resolution point clouds."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260327_223510/best_model.pt",
        help="Path to best_model.pt or final_model.pt",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Optional path to args.json. Defaults to <checkpoint_dir>/args.json",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional path to H5 dataset. Defaults to test_path from args.json",
    )
    parser.add_argument(
        "--source_base_dir",
        type=str,
        default="",
        help="Base directory that contains Area_*_down_sampled folders. Defaults to project root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Defaults to <checkpoint_dir>/dense_pointcloud_visualizations",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for both sampled-block ranking inference and dense vote inference.",
    )
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument(
        "--area_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of area ids to process",
    )
    parser.add_argument(
        "--block_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit H5 block indices to process. When set, top-k ranking is skipped.",
    )
    parser.add_argument(
        "--save_all_blocks",
        action="store_true",
        default=False,
        help="Process every sampled block in the selected areas instead of ranking by predicted score.",
    )
    parser.add_argument(
        "--min_pred_ratio",
        type=float,
        default=0.002,
        help="Minimum sampled-block predicted positive ratio when ranking is enabled.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=6,
        help="Maximum number of sampled blocks to densify when ranking is enabled. 0 means no limit.",
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=0,
        help="Override the number of dense voting passes. 0 enables adaptive coverage-based voting.",
    )
    parser.add_argument(
        "--coverage_target",
        type=float,
        default=0.95,
        help="Target source-point coverage used to derive adaptive vote counts.",
    )
    parser.add_argument("--min_votes", type=int, default=16)
    parser.add_argument("--max_votes", type=int, default=48)
    parser.add_argument(
        "--knn_k",
        type=int,
        default=8,
        help="Number of neighbors used to fill unsampled source points.",
    )
    parser.add_argument(
        "--disable_knn_fill",
        action="store_true",
        default=False,
        help="Leave unsampled source points at probability 0 instead of filling them from nearby sampled points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dense sampling votes.",
    )
    parser.add_argument(
        "--pointcloud_dirname",
        type=str,
        default="dense_pointcloud_blocks",
        help="Directory name for exported dense point clouds",
    )
    parser.add_argument(
        "--base_gray",
        type=float,
        default=0.55,
        help="Base grayscale value for non-leak points",
    )
    parser.add_argument(
        "--gray_scale",
        type=float,
        default=0.35,
        help="Intensity contribution added to the background grayscale",
    )
    parser.add_argument(
        "--bbox_line_points",
        type=int,
        default=64,
        help="Number of sampled points per bbox edge for the combined PLY preview",
    )
    parser.add_argument(
        "--bbox_min_points",
        type=int,
        default=16,
        help="Minimum number of predicted positive points required to emit a bbox",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        default=True,
        help="Save JSON metadata for exported dense point clouds",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = Path(args.config_path).resolve() if args.config_path else checkpoint_path.with_name("args.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        run_args = json.load(f)

    dataset_path = Path(args.dataset_path).resolve() if args.dataset_path else Path(run_args["test_path"]).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    source_base_dir = Path(args.source_base_dir).resolve() if args.source_base_dir else ROOT_DIR
    if not source_base_dir.exists():
        raise FileNotFoundError(f"Source base dir not found: {source_base_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint_path.parent / "dense_pointcloud_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, config_path, dataset_path, source_base_dir, output_dir


def load_dataset_attrs(dataset_path: Path) -> dict[str, float | int]:
    with h5py.File(dataset_path, "r") as f:
        return {
            "cell_size_x": float(f.attrs.get("cell_size_x", 2.0)),
            "cell_size_y": float(f.attrs.get("cell_size_y", 2.0)),
            "num_points": int(f.attrs.get("num_points", 4096)),
        }


def load_txt_points(path: Path, label_val: int) -> np.ndarray:
    data = np.fromfile(path, sep=" ", dtype=np.float32)
    if data.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    if data.size % 4 == 0:
        data = data.reshape(-1, 4)
    elif data.size % 3 == 0:
        data = data.reshape(-1, 3)
        data = np.hstack([data, np.zeros((len(data), 1), dtype=np.float32)])
    else:
        raise ValueError(f"Unsupported point format: {path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    labels = np.full((len(data), 1), float(label_val), dtype=np.float32)
    return np.hstack([data[:, :4], labels]).astype(np.float32)


def pack_grid_key(grid_x: np.ndarray | int, grid_y: np.ndarray | int) -> np.ndarray | np.int64:
    gx = np.asarray(grid_x, dtype=np.int64)
    gy = np.asarray(grid_y, dtype=np.int64)
    return (gx << 32) | (gy & np.int64(0xFFFFFFFF))


def load_area_source_blocks(
    source_base_dir: Path,
    area_id: int,
    cell_size_x: float,
    cell_size_y: float,
) -> dict[str, np.ndarray | float]:
    area_root = source_base_dir / f"Area_{area_id}_down_sampled" / f"Area_{area_id}_down_sampled"
    bg_path = area_root / "background.txt"
    leak_path = area_root / "leakage.txt"
    if (not bg_path.exists()) or (not leak_path.exists()):
        raise FileNotFoundError(f"Missing source files for area {area_id}: {bg_path} {leak_path}")

    print(f"[area {area_id}] loading source points from {area_root}", flush=True)
    bg_points = load_txt_points(bg_path, 0)
    leak_points = load_txt_points(leak_path, 1)
    all_points = np.vstack([bg_points, leak_points]).astype(np.float32, copy=False)
    if len(all_points) == 0:
        raise RuntimeError(f"Area {area_id} has no source points.")

    x_min = float(np.min(all_points[:, 0]))
    y_min = float(np.min(all_points[:, 1]))
    grid_x = np.floor((all_points[:, 0] - x_min) / max(float(cell_size_x), 1e-6)).astype(np.int32)
    grid_y = np.floor((all_points[:, 1] - y_min) / max(float(cell_size_y), 1e-6)).astype(np.int32)
    grid_key = pack_grid_key(grid_x, grid_y)
    order = np.argsort(grid_key, kind="mergesort")
    return {
        "all_points": all_points[order],
        "grid_key": grid_key[order],
        "x_min": x_min,
        "y_min": y_min,
    }


def extract_source_block(area_source: dict[str, np.ndarray | float], grid_x: int, grid_y: int) -> np.ndarray:
    grid_key = np.asarray(area_source["grid_key"], dtype=np.int64)
    all_points = np.asarray(area_source["all_points"], dtype=np.float32)
    key = int(pack_grid_key(grid_x, grid_y))
    left = int(np.searchsorted(grid_key, key, side="left"))
    right = int(np.searchsorted(grid_key, key, side="right"))
    if right <= left:
        raise KeyError(f"Source block not found for grid=({grid_x}, {grid_y})")
    return all_points[left:right]


def choose_vote_count(
    raw_point_count: int,
    sample_points: int,
    coverage_target: float,
    min_votes: int,
    max_votes: int,
    override_votes: int,
) -> int:
    if int(override_votes) > 0:
        return max(int(override_votes), 1)
    if raw_point_count <= sample_points:
        return 1
    sample_ratio = min(float(sample_points) / float(raw_point_count), 1.0 - 1e-6)
    target = float(np.clip(coverage_target, 1e-3, 0.999999))
    votes = math.ceil(math.log(max(1.0 - target, 1e-6)) / math.log(max(1.0 - sample_ratio, 1e-6)))
    votes = max(votes, max(int(min_votes), 1))
    votes = min(votes, max(int(max_votes), 1))
    return max(votes, 1)


def estimate_coverage(raw_point_count: int, sample_points: int, num_votes: int) -> float:
    if raw_point_count <= 0:
        return 0.0
    if raw_point_count <= sample_points:
        return 1.0
    remain = max(1.0 - float(sample_points) / float(raw_point_count), 0.0)
    return float(1.0 - remain ** max(int(num_votes), 0))


def infer_dense_block_probabilities(
    model,
    dense_points: np.ndarray,
    device: torch.device,
    sample_points: int,
    in_channels: int,
    num_votes: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dense_points = np.asarray(dense_points, dtype=np.float32)
    if dense_points.ndim != 2 or dense_points.shape[1] < in_channels:
        raise ValueError(f"Expected dense_points shape [N, >= {in_channels}], got {dense_points.shape}")

    num_dense_points = int(len(dense_points))
    if num_dense_points == 0:
        raise ValueError("dense_points is empty")

    prob_sum = np.zeros((num_dense_points,), dtype=np.float32)
    vote_count = np.zeros((num_dense_points,), dtype=np.int32)
    vote_batch_size = max(int(batch_size), 1)
    with torch.inference_mode():
        for vote_start in range(0, max(int(num_votes), 1), vote_batch_size):
            vote_end = min(vote_start + vote_batch_size, int(num_votes))
            sampled_batches: list[np.ndarray] = []
            sampled_indices: list[np.ndarray] = []
            for _ in range(vote_start, vote_end):
                replace = num_dense_points < int(sample_points)
                choice = rng.choice(num_dense_points, size=int(sample_points), replace=replace)
                sampled_indices.append(choice.astype(np.int64, copy=False))
                sampled_batches.append(dense_points[choice, :in_channels].astype(np.float32, copy=False))

            normalized = np.stack([normalize_point_cloud(pc) for pc in sampled_batches], axis=0)
            batch_tensor = torch.from_numpy(normalized.transpose(0, 2, 1)).to(device)
            seg_output, _ = model(batch_tensor, return_intermediate=False)
            leak_prob = seg_output[:, 1, :].detach().cpu().numpy().astype(np.float32)

            for block_indices, block_prob in zip(sampled_indices, leak_prob):
                np.add.at(prob_sum, block_indices, block_prob)
                np.add.at(vote_count, block_indices, 1)

            print(
                f"  dense votes {vote_end}/{int(num_votes)} | "
                f"coverage={(vote_count > 0).mean() * 100.0:.2f}%",
                flush=True,
            )

    mean_prob = np.zeros((num_dense_points,), dtype=np.float32)
    known_mask = vote_count > 0
    if np.any(known_mask):
        mean_prob[known_mask] = prob_sum[known_mask] / vote_count[known_mask].astype(np.float32)
    return mean_prob, vote_count


def fill_unsampled_probabilities(
    dense_points_xyz: np.ndarray,
    leak_prob: np.ndarray,
    vote_count: np.ndarray,
    knn_k: int,
) -> tuple[np.ndarray, int, str]:
    known_mask = vote_count > 0
    unknown_mask = ~known_mask
    unknown_count = int(np.sum(unknown_mask))
    if unknown_count == 0:
        return leak_prob.astype(np.float32, copy=False), 0, "none"

    if not np.any(known_mask):
        filled = np.full_like(leak_prob, float(np.mean(leak_prob, dtype=np.float64)), dtype=np.float32)
        return filled, unknown_count, "global_mean"

    if cKDTree is None:
        fallback = leak_prob.astype(np.float32, copy=True)
        fallback[unknown_mask] = float(np.mean(leak_prob[known_mask], dtype=np.float64))
        return fallback, unknown_count, "global_mean_no_scipy"

    known_idx = np.where(known_mask)[0]
    unknown_idx = np.where(unknown_mask)[0]
    tree = cKDTree(np.asarray(dense_points_xyz[known_idx], dtype=np.float32))
    k = min(max(int(knn_k), 1), int(len(known_idx)))
    dist, nn = tree.query(np.asarray(dense_points_xyz[unknown_idx], dtype=np.float32), k=k)
    if k == 1:
        dist = dist[:, None]
        nn = nn[:, None]
    weights = 1.0 / np.maximum(np.asarray(dist, dtype=np.float32), 1e-6)
    neighbor_prob = leak_prob[known_idx[np.asarray(nn, dtype=np.int64)]]
    filled_values = np.sum(weights * neighbor_prob, axis=1) / np.sum(weights, axis=1)
    filled = leak_prob.astype(np.float32, copy=True)
    filled[unknown_idx] = filled_values.astype(np.float32)
    return filled, unknown_count, f"knn_{k}"


def infer_selected_sampled_blocks(
    model,
    sampled_point_clouds: np.ndarray,
    device: torch.device,
    batch_size: int,
    in_channels: int,
) -> np.ndarray:
    return infer_predictions(
        model=model,
        point_clouds=sampled_point_clouds,
        device=device,
        batch_size=max(int(batch_size), 1),
        in_channels=int(in_channels),
    )


def save_dense_npz(
    sample_dir: Path,
    dense_point_cloud: np.ndarray,
    dense_gt: np.ndarray,
    dense_prob: np.ndarray,
    dense_pred_mask: np.ndarray,
    vote_count: np.ndarray,
) -> str:
    npz_path = sample_dir / "dense_predictions.npz"
    np.savez_compressed(
        npz_path,
        point_cloud=dense_point_cloud.astype(np.float32),
        gt_label=dense_gt.astype(np.uint8),
        leak_prob=dense_prob.astype(np.float32),
        pred_label=dense_pred_mask.astype(np.uint8),
        vote_count=vote_count.astype(np.int32),
    )
    return str(npz_path)


def main() -> None:
    args = parse_args()
    checkpoint_path, config_path, dataset_path, source_base_dir, output_dir = resolve_paths(args)
    run_args = load_run_args(config_path)
    device = choose_device(args.device)
    rng = np.random.default_rng(int(args.seed))

    print("=" * 80)
    print("Running dense block inference")
    print("=" * 80)
    print(f"checkpoint={checkpoint_path}")
    print(f"config_path={config_path}")
    print(f"dataset_path={dataset_path}")
    print(f"source_base_dir={source_base_dir}")
    print(f"output_dir={output_dir}")
    print(f"device={device}")

    data = load_dataset(dataset_path)
    dataset_attrs = load_dataset_attrs(dataset_path)
    model = load_model(run_args, checkpoint_path, device)
    sampled_num_points = int(dataset_attrs.get("num_points", data["point_clouds"].shape[1]))
    in_channels = int(run_args.get("in_channels", 4))
    cell_size_x = float(dataset_attrs.get("cell_size_x", 2.0))
    cell_size_y = float(dataset_attrs.get("cell_size_y", 2.0))

    allowed_areas = set(
        int(v) for v in (args.area_ids if args.area_ids is not None else np.unique(data["area_ids"]))
    )

    sampled_prob_all: np.ndarray | None = None
    selected_indices: list[int]
    if args.block_indices is not None and len(args.block_indices) > 0:
        selected_indices = [int(v) for v in args.block_indices]
        invalid = [v for v in selected_indices if v < 0 or v >= len(data["point_clouds"])]
        if invalid:
            raise IndexError(f"Invalid block_indices: {invalid}")
        mismatched = [v for v in selected_indices if int(data["area_ids"][v]) not in allowed_areas]
        if mismatched:
            raise ValueError(f"block_indices outside selected area_ids: {mismatched}")
        sampled_prob_selected = infer_selected_sampled_blocks(
            model=model,
            sampled_point_clouds=data["point_clouds"][selected_indices],
            device=device,
            batch_size=max(int(args.batch_size), 1),
            in_channels=in_channels,
        )
        sampled_prob_all = np.zeros(
            (len(data["point_clouds"]), data["point_clouds"].shape[1]),
            dtype=np.float32,
        )
        for local_idx, global_idx in enumerate(selected_indices):
            sampled_prob_all[global_idx] = sampled_prob_selected[local_idx]
    else:
        sampled_prob_all = infer_selected_sampled_blocks(
            model=model,
            sampled_point_clouds=data["point_clouds"],
            device=device,
            batch_size=max(int(args.batch_size), 1),
            in_channels=in_channels,
        )
        sampled_pred_mask_all = sampled_prob_all >= float(args.prob_threshold)
        sampled_pred_ratio_all = sampled_pred_mask_all.mean(axis=1).astype(np.float32)
        sampled_max_prob_all = sampled_prob_all.max(axis=1).astype(np.float32)
        selected_indices = select_indices(
            pred_ratio=sampled_pred_ratio_all,
            max_prob=sampled_max_prob_all,
            area_ids=data["area_ids"],
            allowed_areas=allowed_areas,
            save_all_blocks=bool(args.save_all_blocks),
            min_pred_ratio=float(args.min_pred_ratio),
            top_k=int(args.top_k),
        )

    if not selected_indices:
        raise RuntimeError("No blocks selected for dense inference.")

    rank_map = {int(idx): rank for rank, idx in enumerate(selected_indices, start=1)}
    pointcloud_root = output_dir / args.pointcloud_dirname
    pointcloud_root.mkdir(parents=True, exist_ok=True)

    grouped_indices: dict[int, list[int]] = {}
    for idx in selected_indices:
        grouped_indices.setdefault(int(data["area_ids"][idx]), []).append(int(idx))

    summary_records: list[dict] = []
    for area_id in sorted(grouped_indices):
        area_source = load_area_source_blocks(
            source_base_dir=source_base_dir,
            area_id=int(area_id),
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
        )

        for idx in grouped_indices[area_id]:
            sampled_prob = np.asarray(sampled_prob_all[idx], dtype=np.float32)
            sampled_pred_mask = sampled_prob >= float(args.prob_threshold)
            sampled_pred_ratio = float(np.mean(sampled_pred_mask))
            sampled_max_prob = float(np.max(sampled_prob))
            sampled_gt_ratio = float(np.mean(data["seg_labels"][idx]))

            dense_block = extract_source_block(
                area_source=area_source,
                grid_x=int(data["grid_x"][idx]),
                grid_y=int(data["grid_y"][idx]),
            )
            dense_point_cloud = np.asarray(dense_block[:, :4], dtype=np.float32)
            dense_gt = np.asarray(dense_block[:, 4], dtype=np.int64)
            source_raw_count = int(len(dense_point_cloud))
            if source_raw_count != int(data["raw_point_counts"][idx]):
                raise RuntimeError(
                    f"Recovered raw count mismatch for idx={idx}: "
                    f"{source_raw_count} vs h5 {int(data['raw_point_counts'][idx])}"
                )

            num_votes = choose_vote_count(
                raw_point_count=source_raw_count,
                sample_points=sampled_num_points,
                coverage_target=float(args.coverage_target),
                min_votes=int(args.min_votes),
                max_votes=int(args.max_votes),
                override_votes=int(args.num_votes),
            )
            expected_coverage = estimate_coverage(
                raw_point_count=source_raw_count,
                sample_points=sampled_num_points,
                num_votes=num_votes,
            )

            print(
                f"[rank {rank_map[idx]:03d}] area={int(area_id)} grid=({int(data['grid_x'][idx])}, {int(data['grid_y'][idx])}) "
                f"raw={source_raw_count} votes={num_votes} expected_coverage={expected_coverage * 100.0:.2f}%",
                flush=True,
            )
            dense_prob, vote_count = infer_dense_block_probabilities(
                model=model,
                dense_points=dense_point_cloud,
                device=device,
                sample_points=sampled_num_points,
                in_channels=in_channels,
                num_votes=num_votes,
                batch_size=max(int(args.batch_size), 1),
                rng=rng,
            )
            observed_coverage = float(np.mean(vote_count > 0))
            fill_unknown_count = 0
            fill_method = "disabled"
            if not bool(args.disable_knn_fill):
                dense_prob, fill_unknown_count, fill_method = fill_unsampled_probabilities(
                    dense_points_xyz=dense_point_cloud[:, :3],
                    leak_prob=dense_prob,
                    vote_count=vote_count,
                    knn_k=int(args.knn_k),
                )

            dense_pred_mask = dense_prob >= float(args.prob_threshold)
            dense_pred_ratio = float(np.mean(dense_pred_mask))
            dense_gt_ratio = float(np.mean(dense_gt == 1))
            bbox = compute_bbox(dense_point_cloud[dense_pred_mask, :3], min_points=int(args.bbox_min_points))
            block_stem = build_block_stem(
                area_id=int(area_id),
                grid_x=int(data["grid_x"][idx]),
                grid_y=int(data["grid_y"][idx]),
                pred_ratio=dense_pred_ratio,
                gt_ratio=dense_gt_ratio,
            )
            files = export_one_block(
                output_root=pointcloud_root,
                block_stem=block_stem,
                point_cloud=dense_point_cloud,
                leak_prob=dense_prob,
                seg_labels=dense_gt,
                pred_mask=dense_pred_mask,
                bbox=bbox,
                bbox_line_points=int(args.bbox_line_points),
                base_gray=float(args.base_gray),
                gray_scale=float(args.gray_scale),
            )
            sample_dir = Path(files["sample_dir"])
            dense_npz_path = save_dense_npz(
                sample_dir=sample_dir,
                dense_point_cloud=dense_point_cloud,
                dense_gt=dense_gt,
                dense_prob=dense_prob,
                dense_pred_mask=dense_pred_mask,
                vote_count=vote_count,
            )
            files["dense_npz"] = dense_npz_path

            record = {
                "rank": int(rank_map[idx]),
                "index": int(idx),
                "area_id": int(area_id),
                "grid_x": int(data["grid_x"][idx]),
                "grid_y": int(data["grid_y"][idx]),
                "sampled_num_points": int(sampled_num_points),
                "raw_point_count": int(source_raw_count),
                "num_votes": int(num_votes),
                "expected_coverage": float(expected_coverage),
                "observed_coverage_before_fill": float(observed_coverage),
                "fill_unknown_count": int(fill_unknown_count),
                "fill_method": str(fill_method),
                "mean_vote_count_sampled_points": float(np.mean(vote_count[vote_count > 0])) if np.any(vote_count > 0) else 0.0,
                "sampled_pred_ratio": float(sampled_pred_ratio),
                "sampled_gt_ratio": float(sampled_gt_ratio),
                "sampled_max_prob": float(sampled_max_prob),
                "dense_pred_ratio": float(dense_pred_ratio),
                "dense_gt_ratio": float(dense_gt_ratio),
                "dense_pred_positive_points": int(np.sum(dense_pred_mask)),
                "dense_gt_positive_points": int(np.sum(dense_gt == 1)),
                "bbox": bbox,
                "files": files,
            }
            summary_records.append(record)
            print(
                f"  saved {block_stem} | dense_pred={dense_pred_ratio * 100.0:.2f}% | "
                f"dense_gt={dense_gt_ratio * 100.0:.2f}% | bbox={'yes' if bbox is not None else 'no'}",
                flush=True,
            )

        del area_source

    summary_records = sorted(summary_records, key=lambda item: int(item["rank"]))
    if args.save_summary:
        summary = {
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "source_base_dir": str(source_base_dir),
            "device": str(device),
            "prob_threshold": float(args.prob_threshold),
            "coverage_target": float(args.coverage_target),
            "num_votes_override": int(args.num_votes),
            "min_votes": int(args.min_votes),
            "max_votes": int(args.max_votes),
            "knn_k": int(args.knn_k),
            "disable_knn_fill": bool(args.disable_knn_fill),
            "selected_count": int(len(summary_records)),
            "selected_areas": sorted(int(v) for v in allowed_areas),
            "pointcloud_root": str(pointcloud_root),
            "samples": summary_records,
        }
        summary_path = output_dir / "dense_pointcloud_inference_summary.json"
        with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved {summary_path}", flush=True)

    print("=" * 80)
    print(f"dense_exported_blocks={len(summary_records)}")
    print(f"pointcloud_root={pointcloud_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
