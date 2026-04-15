#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from leakage_detection_torch.fusion_liquid_model_v2_torch import DN_MS_LiquidNet_V2_Torch


LEAK_BLUE = np.array([30, 74, 255], dtype=np.uint8)
BBOX_BLUE = np.array([70, 115, 255], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export predicted leakage point clouds with blue leak points and 3D bbox wireframes."
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
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Defaults to <checkpoint_dir>/pointcloud_visualizations",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument(
        "--area_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of area ids to export",
    )
    parser.add_argument(
        "--save_all_blocks",
        action="store_true",
        default=False,
        help="Export every block in the selected areas instead of only positive predictions",
    )
    parser.add_argument(
        "--min_pred_ratio",
        type=float,
        default=0.002,
        help="Minimum predicted positive ratio used when --save_all_blocks is disabled",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Optional cap after ranking blocks by confidence. 0 means no limit",
    )
    parser.add_argument(
        "--pointcloud_dirname",
        type=str,
        default="pointcloud_blocks",
        help="Directory name for exported per-block point clouds",
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
        default=4,
        help="Minimum number of predicted positive points required to emit a bbox",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        default=True,
        help="Save JSON metadata for exported point clouds",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
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

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint_path.parent / "pointcloud_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, config_path, dataset_path, output_dir


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU。", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def load_run_args(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(run_args: dict, checkpoint_path: Path, device: torch.device) -> DN_MS_LiquidNet_V2_Torch:
    model = DN_MS_LiquidNet_V2_Torch(
        in_channels=int(run_args.get("in_channels", 4)),
        seg_classes=int(run_args.get("seg_classes", 2)),
        cls_classes=int(run_args.get("cls_classes", 2)),
        k_scales=[int(v) for v in run_args.get("k_scales", [10, 20, 40])],
        use_noise_guidance=bool(run_args.get("use_noise_guidance", True)),
        use_progressive=bool(run_args.get("use_progressive", True)),
        use_noise_leak_corr=bool(run_args.get("use_noise_leak_corr", True)),
        use_multi_scale=bool(run_args.get("use_multi_scale", True)),
        use_uncertainty_fusion=bool(run_args.get("use_uncertainty_fusion", False)),
        use_simple_uncertainty=bool(run_args.get("use_simple_uncertainty", False)),
        disable_cls=bool(run_args.get("disable_cls", False)),
        boundary_input_mode=str(run_args.get("boundary_input_mode", "features_fine_probs")),
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(dataset_path, "r") as f:
        data = {
            "point_clouds": f["point_clouds"][:].astype(np.float32),
            "seg_labels": f["seg_labels"][:].astype(np.int64),
            "cls_labels": f["cls_labels"][:].astype(np.int64),
            "area_ids": f["area_ids"][:].astype(np.int64),
            "grid_x": f["grid_x"][:].astype(np.int64),
            "grid_y": f["grid_y"][:].astype(np.int64),
            "positive_ratios": f["positive_ratios"][:].astype(np.float32),
            "raw_point_counts": f["raw_point_counts"][:].astype(np.int64)
            if "raw_point_counts" in f
            else np.full((len(f["point_clouds"]),), f["point_clouds"].shape[1], dtype=np.int64),
            "sampled_with_replacement": f["sampled_with_replacement"][:].astype(np.bool_)
            if "sampled_with_replacement" in f
            else np.zeros((len(f["point_clouds"]),), dtype=np.bool_),
        }
    return data


def normalize_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    point_cloud = point_cloud.copy()
    xyz = point_cloud[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    if max_dist > 0:
        xyz = xyz / max_dist
    point_cloud[:, :3] = xyz

    if point_cloud.shape[1] > 3:
        for i in range(3, point_cloud.shape[1]):
            feature = point_cloud[:, i]
            min_val = feature.min()
            max_val = feature.max()
            if max_val > min_val:
                point_cloud[:, i] = (feature - min_val) / (max_val - min_val)
    return point_cloud.astype(np.float32)


def infer_predictions(
    model: DN_MS_LiquidNet_V2_Torch,
    point_clouds: np.ndarray,
    device: torch.device,
    batch_size: int,
    in_channels: int,
) -> np.ndarray:
    all_probs: list[np.ndarray] = []
    total = len(point_clouds)
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            raw_batch = point_clouds[start:end, :, :in_channels]
            norm_batch = np.stack([normalize_point_cloud(pc) for pc in raw_batch], axis=0)
            batch_tensor = torch.from_numpy(norm_batch.transpose(0, 2, 1)).to(device)
            seg_output, _ = model(batch_tensor, return_intermediate=False)
            leak_prob = seg_output[:, 1, :].detach().cpu().numpy().astype(np.float32)
            all_probs.append(leak_prob)
            print(f"infer {end}/{total}", flush=True)
    return np.concatenate(all_probs, axis=0)


def robust_minmax(values: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.percentile(values, low_q))
    hi = float(np.percentile(values, high_q))
    if hi <= lo:
        return np.clip(values - values.min(), 0.0, None).astype(np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_block_stem(
    area_id: int,
    grid_x: int,
    grid_y: int,
    pred_ratio: float,
    gt_ratio: float,
) -> str:
    return (
        f"area_{area_id:02d}_gx_{grid_x:03d}_gy_{grid_y:02d}"
        f"_pred_{pred_ratio * 100.0:05.2f}_gt_{gt_ratio * 100.0:05.2f}"
    )


def colorize_points(
    point_cloud: np.ndarray,
    pred_mask: np.ndarray,
    base_gray: float,
    gray_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if point_cloud.shape[1] > 3:
        intensity = point_cloud[:, 3]
        intensity_norm = robust_minmax(intensity)
    else:
        intensity_norm = np.ones((len(point_cloud),), dtype=np.float32) * 0.5

    gray = np.clip(float(base_gray) + float(gray_scale) * intensity_norm, 0.0, 1.0)
    rgb = np.repeat((gray * 255.0).astype(np.uint8)[:, None], 3, axis=1)
    rgb[pred_mask] = LEAK_BLUE
    return rgb, intensity_norm.astype(np.float32)


def compute_bbox(points_xyz: np.ndarray, min_points: int) -> dict[str, list[float]] | None:
    if len(points_xyz) < max(int(min_points), 1):
        return None
    min_xyz = points_xyz.min(axis=0).astype(np.float32)
    max_xyz = points_xyz.max(axis=0).astype(np.float32)
    span = max_xyz - min_xyz
    degenerate = span < 1e-6
    if np.any(degenerate):
        pad = np.where(degenerate, 1e-3, 0.0).astype(np.float32)
        min_xyz = min_xyz - pad
        max_xyz = max_xyz + pad
    return {
        "min_xyz": [float(v) for v in min_xyz.tolist()],
        "max_xyz": [float(v) for v in max_xyz.tolist()],
        "size_xyz": [float(v) for v in (max_xyz - min_xyz).tolist()],
        "center_xyz": [float(v) for v in ((min_xyz + max_xyz) * 0.5).tolist()],
    }


def generate_bbox_edge_points(
    bbox: dict[str, list[float]],
    samples_per_edge: int,
) -> np.ndarray:
    min_xyz = np.asarray(bbox["min_xyz"], dtype=np.float32)
    max_xyz = np.asarray(bbox["max_xyz"], dtype=np.float32)
    x0, y0, z0 = min_xyz.tolist()
    x1, y1, z1 = max_xyz.tolist()
    corners = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    t = np.linspace(0.0, 1.0, num=max(int(samples_per_edge), 2), dtype=np.float32)[:, None]
    sampled = []
    for start_idx, end_idx in edges:
        start = corners[start_idx]
        end = corners[end_idx]
        sampled.append(start[None, :] * (1.0 - t) + end[None, :] * t)
    return np.concatenate(sampled, axis=0).astype(np.float32)


def write_bbox_obj(path: Path, bbox: dict[str, list[float]]) -> None:
    min_xyz = np.asarray(bbox["min_xyz"], dtype=np.float32)
    max_xyz = np.asarray(bbox["max_xyz"], dtype=np.float32)
    x0, y0, z0 = min_xyz.tolist()
    x1, y1, z1 = max_xyz.tolist()
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 5),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# bbox wireframe\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for s, e in edges:
            f.write(f"l {s} {e}\n")


def write_ascii_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    leak_prob: np.ndarray,
    pred_label: np.ndarray,
    gt_label: np.ndarray,
    intensity: np.ndarray,
    is_bbox: np.ndarray,
) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    leak_prob = np.asarray(leak_prob, dtype=np.float32)
    pred_label = np.asarray(pred_label, dtype=np.uint8)
    gt_label = np.asarray(gt_label, dtype=np.uint8)
    intensity = np.asarray(intensity, dtype=np.float32)
    is_bbox = np.asarray(is_bbox, dtype=np.uint8)

    if not (
        len(xyz)
        == len(rgb)
        == len(leak_prob)
        == len(pred_label)
        == len(gt_label)
        == len(intensity)
        == len(is_bbox)
    ):
        raise ValueError("All PLY attributes must have the same length")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment exported_by export_predicted_pointclouds.py\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float leak_prob\n")
        f.write("property uchar pred_label\n")
        f.write("property uchar gt_label\n")
        f.write("property float intensity\n")
        f.write("property uchar is_bbox\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{int(r)} {int(g)} {int(b)} "
                f"{float(leak_prob[i]):.6f} {int(pred_label[i])} {int(gt_label[i])} "
                f"{float(intensity[i]):.6f} {int(is_bbox[i])}\n"
            )


def select_indices(
    pred_ratio: np.ndarray,
    max_prob: np.ndarray,
    area_ids: np.ndarray,
    allowed_areas: set[int],
    save_all_blocks: bool,
    min_pred_ratio: float,
    top_k: int,
) -> list[int]:
    area_mask = np.array([int(v) in allowed_areas for v in area_ids], dtype=bool)
    if save_all_blocks:
        candidate_idx = np.where(area_mask)[0]
    else:
        candidate_idx = np.where(area_mask & (pred_ratio >= float(min_pred_ratio)))[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.where(area_mask)[0]

    if len(candidate_idx) == 0:
        return []

    score = pred_ratio * 0.7 + max_prob * 0.3
    ranked = candidate_idx[np.argsort(score[candidate_idx])[::-1]]
    if int(top_k) > 0:
        ranked = ranked[: int(top_k)]
    return [int(v) for v in ranked.tolist()]


def export_one_block(
    output_root: Path,
    block_stem: str,
    point_cloud: np.ndarray,
    leak_prob: np.ndarray,
    seg_labels: np.ndarray,
    pred_mask: np.ndarray,
    bbox: dict[str, list[float]] | None,
    bbox_line_points: int,
    base_gray: float,
    gray_scale: float,
) -> dict[str, str]:
    sample_dir = output_root / block_stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    points_xyz = point_cloud[:, :3].astype(np.float32)
    rgb, intensity_norm = colorize_points(
        point_cloud=point_cloud,
        pred_mask=pred_mask,
        base_gray=base_gray,
        gray_scale=gray_scale,
    )

    pred_label = pred_mask.astype(np.uint8)
    gt_label = np.asarray(seg_labels, dtype=np.uint8)
    is_bbox = np.zeros((len(points_xyz),), dtype=np.uint8)

    scene_points_path = sample_dir / "scene_points.ply"
    write_ascii_ply(
        path=scene_points_path,
        xyz=points_xyz,
        rgb=rgb,
        leak_prob=leak_prob,
        pred_label=pred_label,
        gt_label=gt_label,
        intensity=intensity_norm,
        is_bbox=is_bbox,
    )

    scene_with_bbox_path = sample_dir / "scene_with_bbox.ply"
    if bbox is None:
        write_ascii_ply(
            path=scene_with_bbox_path,
            xyz=points_xyz,
            rgb=rgb,
            leak_prob=leak_prob,
            pred_label=pred_label,
            gt_label=gt_label,
            intensity=intensity_norm,
            is_bbox=is_bbox,
        )
        return {
            "sample_dir": str(sample_dir),
            "scene_points_ply": str(scene_points_path),
            "scene_with_bbox_ply": str(scene_with_bbox_path),
            "bbox_obj": "",
            "bbox_json": "",
        }

    bbox_edge_points = generate_bbox_edge_points(
        bbox=bbox,
        samples_per_edge=max(int(bbox_line_points), 2),
    )
    bbox_rgb = np.repeat(BBOX_BLUE[None, :], len(bbox_edge_points), axis=0)
    bbox_leak_prob = np.ones((len(bbox_edge_points),), dtype=np.float32)
    bbox_pred_label = np.full((len(bbox_edge_points),), 255, dtype=np.uint8)
    bbox_gt_label = np.full((len(bbox_edge_points),), 255, dtype=np.uint8)
    bbox_intensity = np.ones((len(bbox_edge_points),), dtype=np.float32)
    bbox_mask = np.ones((len(bbox_edge_points),), dtype=np.uint8)

    write_ascii_ply(
        path=scene_with_bbox_path,
        xyz=np.concatenate([points_xyz, bbox_edge_points], axis=0),
        rgb=np.concatenate([rgb, bbox_rgb], axis=0),
        leak_prob=np.concatenate([leak_prob, bbox_leak_prob], axis=0),
        pred_label=np.concatenate([pred_label, bbox_pred_label], axis=0),
        gt_label=np.concatenate([gt_label, bbox_gt_label], axis=0),
        intensity=np.concatenate([intensity_norm, bbox_intensity], axis=0),
        is_bbox=np.concatenate([is_bbox, bbox_mask], axis=0),
    )

    bbox_obj_path = sample_dir / "bbox.obj"
    bbox_json_path = sample_dir / "bbox.json"
    write_bbox_obj(bbox_obj_path, bbox)
    with open(bbox_json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(bbox, f, indent=2, ensure_ascii=False)

    return {
        "sample_dir": str(sample_dir),
        "scene_points_ply": str(scene_points_path),
        "scene_with_bbox_ply": str(scene_with_bbox_path),
        "bbox_obj": str(bbox_obj_path),
        "bbox_json": str(bbox_json_path),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path, config_path, dataset_path, output_dir = resolve_paths(args)
    run_args = load_run_args(config_path)
    device = choose_device(args.device)

    print("=" * 80)
    print("Exporting predicted leakage point clouds")
    print("=" * 80)
    print(f"checkpoint={checkpoint_path}")
    print(f"config_path={config_path}")
    print(f"dataset_path={dataset_path}")
    print(f"output_dir={output_dir}")
    print(f"device={device}")

    data = load_dataset(dataset_path)
    model = load_model(run_args, checkpoint_path, device)
    leak_prob = infer_predictions(
        model=model,
        point_clouds=data["point_clouds"],
        device=device,
        batch_size=max(int(args.batch_size), 1),
        in_channels=int(run_args.get("in_channels", 4)),
    )

    pred_mask_all = leak_prob >= float(args.prob_threshold)
    pred_ratio = pred_mask_all.mean(axis=1).astype(np.float32)
    max_prob = leak_prob.max(axis=1).astype(np.float32)
    gt_ratio = data["seg_labels"].mean(axis=1).astype(np.float32)

    allowed_areas = set(
        int(v) for v in (args.area_ids if args.area_ids is not None else np.unique(data["area_ids"]))
    )
    selected_indices = select_indices(
        pred_ratio=pred_ratio,
        max_prob=max_prob,
        area_ids=data["area_ids"],
        allowed_areas=allowed_areas,
        save_all_blocks=bool(args.save_all_blocks),
        min_pred_ratio=float(args.min_pred_ratio),
        top_k=int(args.top_k),
    )

    pointcloud_root = output_dir / args.pointcloud_dirname
    pointcloud_root.mkdir(parents=True, exist_ok=True)

    summary_records: list[dict] = []
    for rank, idx in enumerate(selected_indices, start=1):
        point_cloud = data["point_clouds"][idx]
        seg_labels = data["seg_labels"][idx]
        pred_mask = pred_mask_all[idx]
        bbox = compute_bbox(point_cloud[pred_mask, :3], min_points=int(args.bbox_min_points))
        block_stem = build_block_stem(
            area_id=int(data["area_ids"][idx]),
            grid_x=int(data["grid_x"][idx]),
            grid_y=int(data["grid_y"][idx]),
            pred_ratio=float(pred_ratio[idx]),
            gt_ratio=float(gt_ratio[idx]),
        )
        files = export_one_block(
            output_root=pointcloud_root,
            block_stem=block_stem,
            point_cloud=point_cloud,
            leak_prob=leak_prob[idx],
            seg_labels=seg_labels,
            pred_mask=pred_mask,
            bbox=bbox,
            bbox_line_points=int(args.bbox_line_points),
            base_gray=float(args.base_gray),
            gray_scale=float(args.gray_scale),
        )
        record = {
            "rank": int(rank),
            "index": int(idx),
            "area_id": int(data["area_ids"][idx]),
            "grid_x": int(data["grid_x"][idx]),
            "grid_y": int(data["grid_y"][idx]),
            "pred_ratio": float(pred_ratio[idx]),
            "max_prob": float(max_prob[idx]),
            "gt_ratio": float(gt_ratio[idx]),
            "pred_positive_points": int(pred_mask.sum()),
            "gt_positive_points": int(seg_labels.sum()),
            "raw_point_count": int(data["raw_point_counts"][idx]),
            "sampled_with_replacement": bool(data["sampled_with_replacement"][idx]),
            "bbox": bbox,
            "files": files,
        }
        summary_records.append(record)
        print(
            f"[{rank:03d}/{len(selected_indices):03d}] "
            f"saved {block_stem} | pred_ratio={pred_ratio[idx] * 100.0:.2f}% | "
            f"bbox={'yes' if bbox is not None else 'no'}",
            flush=True,
        )

    if args.save_summary:
        summary = {
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "device": str(device),
            "prob_threshold": float(args.prob_threshold),
            "min_pred_ratio": float(args.min_pred_ratio),
            "save_all_blocks": bool(args.save_all_blocks),
            "top_k": int(args.top_k),
            "bbox_min_points": int(args.bbox_min_points),
            "bbox_line_points": int(args.bbox_line_points),
            "selected_count": int(len(summary_records)),
            "selected_areas": sorted(int(v) for v in allowed_areas),
            "pointcloud_root": str(pointcloud_root),
            "samples": summary_records,
        }
        summary_path = output_dir / "pointcloud_visualization_summary.json"
        with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved {summary_path}", flush=True)

    print("=" * 80)
    print(f"exported_blocks={len(summary_records)}")
    print(f"pointcloud_root={pointcloud_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
