#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from leakage_detection_torch.fusion_liquid_model_v2_torch import DN_MS_LiquidNet_V2_Torch


BLUE = np.array([30, 74, 255], dtype=np.uint8)
BBOX_BLUE = np.array([70, 115, 255], dtype=np.uint8)
CANVAS_BG = np.array([244, 244, 244], dtype=np.uint8)


def add_toggle_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize leakage segmentation predictions as grayscale overlays."
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
        help="Output directory. Defaults to <checkpoint_dir>/visualizations",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--projection",
        type=str,
        default="xz",
        choices=["xz", "yz", "xy"],
        help="Projection used to render each block",
    )
    parser.add_argument("--tile_width", type=int, default=256)
    parser.add_argument("--tile_height", type=int, default=576)
    parser.add_argument("--tile_gap", type=int, default=2)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--gallery_cols", type=int, default=4)
    parser.add_argument("--base_gray", type=float, default=0.60)
    parser.add_argument("--gray_gamma", type=float, default=0.75)
    parser.add_argument(
        "--min_gallery_ratio",
        type=float,
        default=0.002,
        help="Minimum predicted positive ratio to prioritize a block in the gallery",
    )
    parser.add_argument(
        "--area_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of area ids to render",
    )
    parser.add_argument("--draw_bbox", action="store_true", default=False)
    add_toggle_arg(parser, "save_overview", False, "Save stitched per-area overview images")
    add_toggle_arg(parser, "save_gallery", False, "Save multi-image gallery figure")
    add_toggle_arg(parser, "save_individual", True, "Save individual high-resolution block images")
    parser.add_argument(
        "--save_all_blocks",
        action="store_true",
        default=False,
        help="Save every rendered block as an individual PNG instead of only the selected top_k samples",
    )
    parser.add_argument(
        "--individual_dirname",
        type=str,
        default="individual_blocks",
        help="Directory name for individual block PNG files",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        default=True,
        help="Save JSON metadata for generated figures",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
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

    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint_path.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    return config_path, dataset_path, output_dir


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


def load_dataset(dataset_path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
        }
        attrs = {
            "cell_size_x": float(f.attrs.get("cell_size_x", 2.0)),
            "cell_size_y": float(f.attrs.get("cell_size_y", 2.0)),
            "num_points": int(f.attrs.get("num_points", data["point_clouds"].shape[1])),
        }
    return data, attrs


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


def compute_area_stats(data: dict[str, np.ndarray]) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for area_id in np.unique(data["area_ids"]):
        mask = data["area_ids"] == area_id
        area_points = data["point_clouds"][mask]
        stats[int(area_id)] = {
            "x_min": float(area_points[:, :, 0].min()),
            "x_max": float(area_points[:, :, 0].max()),
            "y_min": float(area_points[:, :, 1].min()),
            "y_max": float(area_points[:, :, 1].max()),
            "z_min": float(area_points[:, :, 2].min()),
            "z_max": float(area_points[:, :, 2].max()),
        }
    return stats


def clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def project_points(
    point_cloud: np.ndarray,
    projection: str,
    area_stat: dict[str, float],
    grid_x: int,
    grid_y: int,
    cell_size_x: float,
    cell_size_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    z_span = max(area_stat["z_max"] - area_stat["z_min"], 1e-6)
    if projection == "xz":
        origin_x = area_stat["x_min"] + grid_x * cell_size_x
        h = clip01((x - origin_x) / max(cell_size_x, 1e-6))
        v = clip01((z - area_stat["z_min"]) / z_span)
        return h, v

    if projection == "yz":
        origin_y = area_stat["y_min"] + grid_y * cell_size_y
        h = clip01((y - origin_y) / max(cell_size_y, 1e-6))
        v = clip01((z - area_stat["z_min"]) / z_span)
        return h, v

    origin_x = area_stat["x_min"] + grid_x * cell_size_x
    origin_y = area_stat["y_min"] + grid_y * cell_size_y
    h = clip01((x - origin_x) / max(cell_size_x, 1e-6))
    v = clip01((y - origin_y) / max(cell_size_y, 1e-6))
    return h, v


def draw_rectangle(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray, thickness: int = 2) -> None:
    h, w = image.shape[:2]
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return
    for t in range(max(thickness, 1)):
        yy0 = max(y0 - t, 0)
        yy1 = min(y1 + t, h - 1)
        xx0 = max(x0 - t, 0)
        xx1 = min(x1 + t, w - 1)
        image[yy0, xx0 : xx1 + 1] = color
        image[yy1, xx0 : xx1 + 1] = color
        image[yy0 : yy1 + 1, xx0] = color
        image[yy0 : yy1 + 1, xx1] = color


def rasterize_block(
    point_cloud: np.ndarray,
    leak_prob: np.ndarray,
    area_stat: dict[str, float],
    grid_x: int,
    grid_y: int,
    cell_size_x: float,
    cell_size_y: float,
    projection: str,
    tile_width: int,
    tile_height: int,
    point_radius: int,
    prob_threshold: float,
    draw_bbox: bool,
    base_gray: float,
    gray_gamma: float,
) -> np.ndarray:
    h_norm, v_norm = project_points(
        point_cloud=point_cloud,
        projection=projection,
        area_stat=area_stat,
        grid_x=grid_x,
        grid_y=grid_y,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
    )
    x_pix = np.rint(h_norm * (tile_width - 1)).astype(np.int32)
    y_pix = np.rint((1.0 - v_norm) * (tile_height - 1)).astype(np.int32)

    intensity = point_cloud[:, 3] if point_cloud.shape[1] > 3 else np.ones((len(point_cloud),), dtype=np.float32)
    intensity_norm = robust_minmax(intensity)

    gray_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
    gray_count = np.zeros((tile_height, tile_width), dtype=np.float32)
    mask_map = np.zeros((tile_height, tile_width), dtype=np.float32)

    radius = max(int(point_radius), 0)
    for dy in range(-radius, radius + 1):
        yy = np.clip(y_pix + dy, 0, tile_height - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(x_pix + dx, 0, tile_width - 1)
            np.add.at(gray_sum, (yy, xx), intensity_norm)
            np.add.at(gray_count, (yy, xx), 1.0)
            np.maximum.at(mask_map, (yy, xx), leak_prob)

    valid = gray_count > 0
    gray_map = np.full((tile_height, tile_width), float(np.clip(base_gray, 0.0, 1.0)), dtype=np.float32)
    if np.any(valid):
        averaged = np.zeros_like(gray_map)
        averaged[valid] = gray_sum[valid] / gray_count[valid]
        occupancy = gray_count / max(float(gray_count.max()), 1.0)
        gamma = max(float(gray_gamma), 1e-3)
        averaged = np.power(np.clip(averaged, 0.0, 1.0), gamma)
        gray_map = np.clip(base_gray + 0.30 * averaged + 0.10 * occupancy, 0.0, 1.0)

    gray_rgb = (gray_map * 255.0).astype(np.uint8)
    image = np.repeat(gray_rgb[..., None], 3, axis=2).astype(np.float32)

    leak_strength = np.clip((mask_map - prob_threshold) / max(1.0 - prob_threshold, 1e-6), 0.0, 1.0)
    alpha = np.where(mask_map >= prob_threshold, 0.45 + 0.55 * leak_strength, 0.0).astype(np.float32)
    image = image * (1.0 - alpha[..., None]) + BLUE.astype(np.float32) * alpha[..., None]
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    if draw_bbox:
        positive = np.argwhere(mask_map >= prob_threshold)
        if positive.size > 0:
            y0, x0 = positive.min(axis=0)
            y1, x1 = positive.max(axis=0)
            draw_rectangle(image, int(x0), int(y0), int(x1), int(y1), BBOX_BLUE, thickness=2)

    return image


def build_block_filename(
    area_id: int,
    grid_x: int,
    grid_y: int,
    pred_ratio: float,
    gt_ratio: float,
) -> str:
    return (
        f"area_{area_id:02d}_gx_{grid_x:03d}_gy_{grid_y:02d}"
        f"_pred_{pred_ratio * 100.0:05.2f}_gt_{gt_ratio * 100.0:05.2f}.png"
    )


def compose_area_overview(
    rendered_blocks: dict[int, np.ndarray],
    sample_indices: list[int],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    tile_width: int,
    tile_height: int,
    tile_gap: int,
) -> np.ndarray:
    gx = grid_x[sample_indices]
    gy = grid_y[sample_indices]
    min_gx = int(gx.min())
    max_gx = int(gx.max())
    min_gy = int(gy.min())
    max_gy = int(gy.max())

    cols = max_gx - min_gx + 1
    rows = max_gy - min_gy + 1
    width = cols * tile_width + max(cols - 1, 0) * tile_gap
    height = rows * tile_height + max(rows - 1, 0) * tile_gap
    canvas = np.full((height, width, 3), CANVAS_BG, dtype=np.uint8)

    for idx in sample_indices:
        col = int(grid_x[idx] - min_gx)
        row = int(max_gy - grid_y[idx])
        x0 = col * (tile_width + tile_gap)
        y0 = row * (tile_height + tile_gap)
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = rendered_blocks[idx]
    return canvas


def select_gallery_indices(
    pred_ratio: np.ndarray,
    max_prob: np.ndarray,
    area_ids: np.ndarray,
    allowed_areas: set[int],
    top_k: int,
    min_gallery_ratio: float,
) -> list[int]:
    area_mask = np.array([int(a) in allowed_areas for a in area_ids], dtype=bool)
    if not np.any(area_mask):
        return []

    scores = pred_ratio * 0.7 + max_prob * 0.3
    candidate_idx = np.where(area_mask & (pred_ratio >= float(min_gallery_ratio)))[0]
    if len(candidate_idx) < top_k:
        candidate_idx = np.where(area_mask)[0]
    sorted_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    return [int(v) for v in sorted_idx[:top_k]]


def save_image(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image).save(path)


def save_individual_blocks(
    output_dir: Path,
    rendered_blocks: dict[int, np.ndarray],
    selected_indices: list[int],
    area_ids: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    pred_ratio: np.ndarray,
    gt_ratio: np.ndarray,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    for idx in selected_indices:
        file_name = build_block_filename(
            area_id=int(area_ids[idx]),
            grid_x=int(grid_x[idx]),
            grid_y=int(grid_y[idx]),
            pred_ratio=float(pred_ratio[idx]),
            gt_ratio=float(gt_ratio[idx]),
        )
        out_path = output_dir / file_name
        save_image(rendered_blocks[idx], out_path)
        saved_files.append(str(out_path))
        print(f"saved {out_path}", flush=True)
    return saved_files


def save_gallery(
    output_path: Path,
    rendered_blocks: dict[int, np.ndarray],
    selected_indices: list[int],
    area_ids: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    pred_ratio: np.ndarray,
    max_prob: np.ndarray,
    gt_ratio: np.ndarray,
    cols: int,
) -> None:
    if not selected_indices:
        return

    cols = max(int(cols), 1)
    rows = int(math.ceil(len(selected_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 4.2), dpi=150)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes[:, None]

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, selected_indices):
        ax.imshow(rendered_blocks[idx])
        ax.set_title(
            f"Area {int(area_ids[idx])} | grid=({int(grid_x[idx])}, {int(grid_y[idx])})\n"
            f"pred={pred_ratio[idx] * 100.0:.2f}% | max={max_prob[idx]:.3f} | gt={gt_ratio[idx] * 100.0:.2f}%",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle("Top Predicted Leakage Blocks", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path, dataset_path, output_dir = resolve_paths(args)
    run_args = load_run_args(config_path)
    device = choose_device(args.device)

    print("=" * 80)
    print("Visualizing leakage segmentation predictions")
    print("=" * 80)
    print(f"checkpoint={checkpoint_path}")
    print(f"config_path={config_path}")
    print(f"dataset_path={dataset_path}")
    print(f"output_dir={output_dir}")
    print(f"device={device}")

    data, attrs = load_dataset(dataset_path)
    area_stats = compute_area_stats(data)
    model = load_model(run_args, checkpoint_path, device)
    leak_prob = infer_predictions(
        model=model,
        point_clouds=data["point_clouds"],
        device=device,
        batch_size=max(int(args.batch_size), 1),
        in_channels=int(run_args.get("in_channels", 4)),
    )

    pred_mask = leak_prob >= float(args.prob_threshold)
    pred_ratio = pred_mask.mean(axis=1).astype(np.float32)
    max_prob = leak_prob.max(axis=1).astype(np.float32)
    gt_ratio = data["seg_labels"].mean(axis=1).astype(np.float32)

    allowed_areas = set(int(v) for v in (args.area_ids if args.area_ids is not None else np.unique(data["area_ids"])))
    rendered_blocks: dict[int, np.ndarray] = {}

    for idx in range(len(data["point_clouds"])):
        area_id = int(data["area_ids"][idx])
        if area_id not in allowed_areas:
            continue
        rendered_blocks[idx] = rasterize_block(
            point_cloud=data["point_clouds"][idx],
            leak_prob=leak_prob[idx],
            area_stat=area_stats[area_id],
            grid_x=int(data["grid_x"][idx]),
            grid_y=int(data["grid_y"][idx]),
            cell_size_x=float(attrs["cell_size_x"]),
            cell_size_y=float(attrs["cell_size_y"]),
            projection=args.projection,
            tile_width=int(args.tile_width),
            tile_height=int(args.tile_height),
            point_radius=int(args.point_radius),
            prob_threshold=float(args.prob_threshold),
            draw_bbox=bool(args.draw_bbox),
            base_gray=float(args.base_gray),
            gray_gamma=float(args.gray_gamma),
        )

    overview_files: list[str] = []
    if args.save_overview:
        for area_id in sorted(allowed_areas):
            sample_indices = [idx for idx in rendered_blocks.keys() if int(data["area_ids"][idx]) == area_id]
            if not sample_indices:
                continue
            overview = compose_area_overview(
                rendered_blocks=rendered_blocks,
                sample_indices=sample_indices,
                grid_x=data["grid_x"],
                grid_y=data["grid_y"],
                tile_width=int(args.tile_width),
                tile_height=int(args.tile_height),
                tile_gap=int(args.tile_gap),
            )
            overview_path = output_dir / f"area_{area_id:02d}_overview.png"
            save_image(overview, overview_path)
            overview_files.append(str(overview_path))
            print(f"saved {overview_path}", flush=True)

    selected_indices = select_gallery_indices(
        pred_ratio=pred_ratio,
        max_prob=max_prob,
        area_ids=data["area_ids"],
        allowed_areas=allowed_areas,
        top_k=max(int(args.top_k), 1),
        min_gallery_ratio=float(args.min_gallery_ratio),
    )
    gallery_path = output_dir / "top_predictions.png"
    if args.save_gallery:
        save_gallery(
            output_path=gallery_path,
            rendered_blocks=rendered_blocks,
            selected_indices=selected_indices,
            area_ids=data["area_ids"],
            grid_x=data["grid_x"],
            grid_y=data["grid_y"],
            pred_ratio=pred_ratio,
            max_prob=max_prob,
            gt_ratio=gt_ratio,
            cols=max(int(args.gallery_cols), 1),
        )
        if selected_indices:
            print(f"saved {gallery_path}", flush=True)

    individual_files: list[str] = []
    if args.save_individual:
        individual_indices = sorted(rendered_blocks.keys()) if args.save_all_blocks else selected_indices
        individual_dir = output_dir / args.individual_dirname
        individual_files = save_individual_blocks(
            output_dir=individual_dir,
            rendered_blocks=rendered_blocks,
            selected_indices=individual_indices,
            area_ids=data["area_ids"],
            grid_x=data["grid_x"],
            grid_y=data["grid_y"],
            pred_ratio=pred_ratio,
            gt_ratio=gt_ratio,
        )

    if args.save_summary:
        summary = {
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "device": str(device),
            "projection": args.projection,
            "prob_threshold": float(args.prob_threshold),
            "tile_width": int(args.tile_width),
            "tile_height": int(args.tile_height),
            "tile_gap": int(args.tile_gap),
            "point_radius": int(args.point_radius),
            "base_gray": float(args.base_gray),
            "gray_gamma": float(args.gray_gamma),
            "draw_bbox": bool(args.draw_bbox),
            "overview_files": overview_files,
            "gallery_file": str(gallery_path) if args.save_gallery and selected_indices else "",
            "individual_files": individual_files,
            "selected_samples": [
                {
                    "index": int(idx),
                    "area_id": int(data["area_ids"][idx]),
                    "grid_x": int(data["grid_x"][idx]),
                    "grid_y": int(data["grid_y"][idx]),
                    "pred_ratio": float(pred_ratio[idx]),
                    "max_prob": float(max_prob[idx]),
                    "gt_ratio": float(gt_ratio[idx]),
                    "raw_point_count": int(data["raw_point_counts"][idx]),
                }
                for idx in selected_indices
            ],
        }
        summary_path = output_dir / "visualization_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved {summary_path}", flush=True)

    print("=" * 80)
    print(f"generated {len(overview_files)} overview image(s)")
    print(f"selected_gallery_blocks={len(selected_indices)}")
    print(f"saved_individual_blocks={len(individual_files)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
