#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


try:
    from scipy import ndimage
except Exception:  # pragma: no cover - runtime fallback only
    ndimage = None


from leakage_detection_torch.export_predicted_pointclouds import robust_minmax


BLUE = np.array([30, 74, 255], dtype=np.uint8)
PRED_BOX = np.array([255, 166, 0], dtype=np.uint8)
GT_BOX = np.array([66, 214, 96], dtype=np.uint8)
CANVAS_BG = np.array([248, 248, 248], dtype=np.uint8)
TEXT_COLOR = (32, 32, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render dense block 2D projection overlays with flat component-wise bbox visualizations."
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        required=True,
        help="Path to dense_pointcloud_inference_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Defaults to <summary_dir>/projection_visualizations",
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional summary sample indices (0-based within summary['samples']) to render.",
    )
    parser.add_argument(
        "--projections",
        type=str,
        nargs="+",
        default=["x_theta", "xz"],
        choices=["xz", "yz", "xy", "x_theta"],
        help="Projection planes to render",
    )
    parser.add_argument("--tile_width", type=int, default=420)
    parser.add_argument("--tile_height", type=int, default=320)
    parser.add_argument("--tile_gap", type=int, default=16)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--prob_threshold", type=float, default=-1.0, help="Use summary threshold when < 0.")
    parser.add_argument("--base_gray", type=float, default=0.58)
    parser.add_argument("--gray_gamma", type=float, default=0.78)
    parser.add_argument(
        "--background_fill_mode",
        type=str,
        default="nearest",
        choices=["nearest", "none"],
        help="How to fill empty background pixels before smoothing.",
    )
    parser.add_argument(
        "--background_sigma",
        type=float,
        default=1.4,
        help="Gaussian sigma used to smooth the reconstructed background intensity.",
    )
    parser.add_argument(
        "--occupancy_sigma",
        type=float,
        default=1.2,
        help="Gaussian sigma used to smooth the occupancy map for background shading.",
    )
    parser.add_argument(
        "--min_component_pixels",
        type=int,
        default=36,
        help="Minimum connected-component pixel count required to emit a 2D bbox.",
    )
    parser.add_argument("--box_padding", type=int, default=4)
    parser.add_argument("--box_thickness", type=int, default=3)
    parser.add_argument("--dash_length", type=int, default=10)
    parser.add_argument("--dash_gap", type=int, default=6)
    parser.add_argument(
        "--mask_close_radius",
        type=int,
        default=2,
        help="Binary closing radius applied before component extraction.",
    )
    parser.add_argument(
        "--mask_dilate_radius",
        type=int,
        default=1,
        help="Binary dilation radius applied after closing to bridge tiny gaps.",
    )
    parser.add_argument(
        "--box_merge_gap",
        type=int,
        default=12,
        help="Merge boxes whose expanded regions overlap within this pixel gap.",
    )
    parser.add_argument(
        "--save_compare_only",
        action="store_true",
        default=False,
        help="Only save compare images and sample panels, skip pred/gt individual tiles.",
    )
    parser.add_argument(
        "--save_gallery",
        action="store_true",
        default=True,
        help="Save per-projection compare galleries across all rendered samples.",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        default=True,
        help="Save JSON metadata for generated projection images.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    summary_path = Path(args.summary_path).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else summary_path.parent / "projection_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return summary_path, output_dir


def load_summary(summary_path: Path) -> dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def load_dense_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"dense npz not found: {path}")
    data = np.load(path)
    return {
        "point_cloud": data["point_cloud"].astype(np.float32),
        "gt_label": data["gt_label"].astype(np.uint8),
        "leak_prob": data["leak_prob"].astype(np.float32),
        "pred_label": data["pred_label"].astype(np.uint8),
        "vote_count": data["vote_count"].astype(np.int32),
    }


def fit_circle_yz(point_cloud: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(point_cloud[:, 1], dtype=np.float64)
    z = np.asarray(point_cloud[:, 2], dtype=np.float64)
    A = np.column_stack([2.0 * y, 2.0 * z, np.ones_like(y)])
    b = y * y + z * z
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cy, cz, c = sol
    radius = math.sqrt(max(float(cy * cy + cz * cz + c), 0.0))
    return float(cy), float(cz), float(radius)


def project_local_points(point_cloud: np.ndarray, projection: str) -> tuple[np.ndarray, np.ndarray]:
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    if projection == "x_theta":
        center_y, center_z, _ = fit_circle_yz(point_cloud)
        theta = np.arctan2(z - center_z, y - center_y).astype(np.float32)
        h = clip01((x - x.min()) / max(float(x.max() - x.min()), 1e-6))
        v = clip01((theta - theta.min()) / max(float(theta.max() - theta.min()), 1e-6))
        return h, v

    if projection == "xz":
        h = clip01((x - x.min()) / max(float(x.max() - x.min()), 1e-6))
        v = clip01((z - z.min()) / max(float(z.max() - z.min()), 1e-6))
        return h, v
    if projection == "yz":
        h = clip01((y - y.min()) / max(float(y.max() - y.min()), 1e-6))
        v = clip01((z - z.min()) / max(float(z.max() - z.min()), 1e-6))
        return h, v
    h = clip01((x - x.min()) / max(float(x.max() - x.min()), 1e-6))
    v = clip01((y - y.min()) / max(float(y.max() - y.min()), 1e-6))
    return h, v


def build_projection_maps(
    point_cloud: np.ndarray,
    pred_prob: np.ndarray,
    gt_label: np.ndarray,
    projection: str,
    tile_width: int,
    tile_height: int,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_norm, v_norm = project_local_points(point_cloud, projection)
    x_pix = np.rint(h_norm * max(int(tile_width) - 1, 1)).astype(np.int32)
    y_pix = np.rint((1.0 - v_norm) * max(int(tile_height) - 1, 1)).astype(np.int32)

    intensity = point_cloud[:, 3] if point_cloud.shape[1] > 3 else np.ones((len(point_cloud),), dtype=np.float32)
    intensity_norm = robust_minmax(intensity)

    gray_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
    gray_count = np.zeros((tile_height, tile_width), dtype=np.float32)
    pred_map = np.zeros((tile_height, tile_width), dtype=np.float32)
    gt_map = np.zeros((tile_height, tile_width), dtype=np.float32)

    radius = max(int(point_radius), 0)
    gt_label = gt_label.astype(np.float32, copy=False)
    pred_prob = pred_prob.astype(np.float32, copy=False)
    for dy in range(-radius, radius + 1):
        yy = np.clip(y_pix + dy, 0, tile_height - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(x_pix + dx, 0, tile_width - 1)
            np.add.at(gray_sum, (yy, xx), intensity_norm)
            np.add.at(gray_count, (yy, xx), 1.0)
            np.maximum.at(pred_map, (yy, xx), pred_prob)
            np.maximum.at(gt_map, (yy, xx), gt_label)

    return gray_sum, gray_count, pred_map, gt_map


def fill_invalid_pixels(values: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    if np.all(valid) or mode == "none":
        return values.astype(np.float32, copy=False)
    if not np.any(valid):
        return np.full_like(values, float(np.mean(values, dtype=np.float64)), dtype=np.float32)
    if ndimage is None:
        filled = values.astype(np.float32, copy=True)
        filled[~valid] = float(np.mean(values[valid], dtype=np.float64))
        return filled

    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = values[tuple(indices)].astype(np.float32, copy=False)
    return filled


def compose_gray_background(
    gray_sum: np.ndarray,
    gray_count: np.ndarray,
    base_gray: float,
    gray_gamma: float,
    background_fill_mode: str,
    background_sigma: float,
    occupancy_sigma: float,
) -> np.ndarray:
    valid = gray_count > 0
    gray_map = np.full(gray_sum.shape, float(np.clip(base_gray, 0.0, 1.0)), dtype=np.float32)
    if np.any(valid):
        averaged = np.zeros_like(gray_map, dtype=np.float32)
        averaged[valid] = gray_sum[valid] / gray_count[valid]
        averaged = fill_invalid_pixels(
            values=averaged,
            valid=valid,
            mode=str(background_fill_mode),
        )
        if ndimage is not None and float(background_sigma) > 1e-6:
            averaged = ndimage.gaussian_filter(averaged, sigma=float(background_sigma), mode="nearest")

        occupancy = gray_count.astype(np.float32, copy=False)
        if ndimage is not None and float(occupancy_sigma) > 1e-6:
            occupancy = ndimage.gaussian_filter(occupancy, sigma=float(occupancy_sigma), mode="nearest")
        occupancy = occupancy / max(float(occupancy.max()), 1.0)
        gamma = max(float(gray_gamma), 1e-3)
        averaged = np.power(np.clip(averaged, 0.0, 1.0), gamma)
        gray_map = np.clip(base_gray + 0.28 * averaged + 0.12 * occupancy, 0.0, 1.0)
    gray_rgb = (gray_map * 255.0).astype(np.uint8)
    return np.repeat(gray_rgb[..., None], 3, axis=2)


def apply_blue_overlay(base_image: np.ndarray, mask_strength: np.ndarray, alpha_min: float, alpha_max: float) -> np.ndarray:
    strength = np.clip(mask_strength.astype(np.float32), 0.0, 1.0)
    alpha = np.where(strength > 0, alpha_min + (alpha_max - alpha_min) * strength, 0.0).astype(np.float32)
    composed = base_image.astype(np.float32) * (1.0 - alpha[..., None]) + BLUE.astype(np.float32) * alpha[..., None]
    return np.clip(composed, 0.0, 255.0).astype(np.uint8)


def extract_component_boxes(mask: np.ndarray, min_component_pixels: int, padding: int) -> list[tuple[int, int, int, int]]:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if not np.any(binary):
        return []

    boxes: list[tuple[int, int, int, int]] = []
    if ndimage is not None:
        structure = np.ones((3, 3), dtype=np.uint8)
        labels, count = ndimage.label(binary, structure=structure)
        objects = ndimage.find_objects(labels)
        for component_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            component_mask = labels[slices] == component_id
            pixel_count = int(np.sum(component_mask))
            if pixel_count < max(int(min_component_pixels), 1):
                continue
            y_slice, x_slice = slices
            x0 = int(x_slice.start) - int(padding)
            x1 = int(x_slice.stop - 1) + int(padding)
            y0 = int(y_slice.start) - int(padding)
            y1 = int(y_slice.stop - 1) + int(padding)
            boxes.append((x0, y0, x1, y1))
        return boxes

    visited = np.zeros_like(binary, dtype=bool)
    h, w = binary.shape
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_y = max_y = y
            min_x = max_x = x
            pixel_count = 0
            while stack:
                cy, cx = stack.pop()
                pixel_count += 1
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] > 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if pixel_count >= max(int(min_component_pixels), 1):
                boxes.append((min_x - int(padding), min_y - int(padding), max_x + int(padding), max_y + int(padding)))
    return boxes


def build_disk_structure(radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def smooth_binary_mask(mask: np.ndarray, close_radius: int, dilate_radius: int) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=bool)
    if not np.any(binary):
        return binary
    if ndimage is None:
        return binary
    if int(close_radius) > 0:
        binary = ndimage.binary_closing(binary, structure=build_disk_structure(int(close_radius)))
    if int(dilate_radius) > 0:
        binary = ndimage.binary_dilation(binary, structure=build_disk_structure(int(dilate_radius)))
    return np.asarray(binary, dtype=bool)


def boxes_should_merge(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    merge_gap: int,
) -> bool:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    gap = max(int(merge_gap), 0)
    return not (
        ax1 + gap < bx0
        or bx1 + gap < ax0
        or ay1 + gap < by0
        or by1 + gap < ay0
    )


def merge_adjacent_boxes(
    boxes: list[tuple[int, int, int, int]],
    merge_gap: int,
) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return list(boxes)

    pending = [tuple(int(v) for v in box) for box in boxes]
    merged = True
    while merged:
        merged = False
        next_boxes: list[tuple[int, int, int, int]] = []
        used = [False] * len(pending)
        for i, box in enumerate(pending):
            if used[i]:
                continue
            x0, y0, x1, y1 = box
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(i + 1, len(pending)):
                    if used[j]:
                        continue
                    if boxes_should_merge((x0, y0, x1, y1), pending[j], merge_gap):
                        bx0, by0, bx1, by1 = pending[j]
                        x0 = min(x0, bx0)
                        y0 = min(y0, by0)
                        x1 = max(x1, bx1)
                        y1 = max(y1, by1)
                        used[j] = True
                        changed = True
                        merged = True
            next_boxes.append((x0, y0, x1, y1))
        pending = next_boxes
    pending.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    return pending


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
    dash_length: int,
    dash_gap: int,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    total = math.hypot(x1 - x0, y1 - y0)
    if total <= 1e-6:
        return
    step = max(int(dash_length) + int(dash_gap), 1)
    dx = (x1 - x0) / total
    dy = (y1 - y0) / total
    start = 0.0
    while start < total:
        end = min(start + max(int(dash_length), 1), total)
        sx = int(round(x0 + dx * start))
        sy = int(round(y0 + dy * start))
        ex = int(round(x0 + dx * end))
        ey = int(round(y0 + dy * end))
        draw.line((sx, sy, ex, ey), fill=color, width=max(int(width), 1))
        start += step


def draw_dashed_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: np.ndarray,
    width: int,
    dash_length: int,
    dash_gap: int,
) -> np.ndarray:
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    w, h = pil_image.size
    color_tuple = tuple(int(v) for v in color.tolist())
    for x0, y0, x1, y1 in boxes:
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        draw_dashed_line(draw, (x0, y0), (x1, y0), color_tuple, width, dash_length, dash_gap)
        draw_dashed_line(draw, (x1, y0), (x1, y1), color_tuple, width, dash_length, dash_gap)
        draw_dashed_line(draw, (x1, y1), (x0, y1), color_tuple, width, dash_length, dash_gap)
        draw_dashed_line(draw, (x0, y1), (x0, y0), color_tuple, width, dash_length, dash_gap)
    return np.asarray(pil_image)


def add_title_band(image: np.ndarray, title: str, band_height: int = 30) -> np.ndarray:
    band_height = max(int(band_height), 18)
    canvas = np.full((image.shape[0] + band_height, image.shape[1], 3), CANVAS_BG, dtype=np.uint8)
    canvas[band_height:] = image
    pil_image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=TEXT_COLOR, font=font)
    return np.asarray(pil_image)


def concat_h(images: list[np.ndarray], gap: int) -> np.ndarray:
    if not images:
        raise ValueError("images is empty")
    gap = max(int(gap), 0)
    height = max(img.shape[0] for img in images)
    width = sum(img.shape[1] for img in images) + gap * max(len(images) - 1, 0)
    canvas = np.full((height, width, 3), CANVAS_BG, dtype=np.uint8)
    x = 0
    for img in images:
        y = (height - img.shape[0]) // 2
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
        x += img.shape[1] + gap
    return canvas


def concat_v(images: list[np.ndarray], gap: int) -> np.ndarray:
    if not images:
        raise ValueError("images is empty")
    gap = max(int(gap), 0)
    height = sum(img.shape[0] for img in images) + gap * max(len(images) - 1, 0)
    width = max(img.shape[1] for img in images)
    canvas = np.full((height, width, 3), CANVAS_BG, dtype=np.uint8)
    y = 0
    for img in images:
        x = (width - img.shape[1]) // 2
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
        y += img.shape[0] + gap
    return canvas


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def render_projection_triplet(
    point_cloud: np.ndarray,
    pred_prob: np.ndarray,
    gt_label: np.ndarray,
    projection: str,
    tile_width: int,
    tile_height: int,
    point_radius: int,
    prob_threshold: float,
    base_gray: float,
    gray_gamma: float,
    min_component_pixels: int,
    box_padding: int,
    box_thickness: int,
    dash_length: int,
    dash_gap: int,
    mask_close_radius: int,
    mask_dilate_radius: int,
    box_merge_gap: int,
    background_fill_mode: str,
    background_sigma: float,
    occupancy_sigma: float,
) -> dict[str, np.ndarray | list[tuple[int, int, int, int]]]:
    gray_sum, gray_count, pred_map, gt_map = build_projection_maps(
        point_cloud=point_cloud,
        pred_prob=pred_prob,
        gt_label=gt_label,
        projection=projection,
        tile_width=tile_width,
        tile_height=tile_height,
        point_radius=point_radius,
    )
    base = compose_gray_background(
        gray_sum=gray_sum,
        gray_count=gray_count,
        base_gray=base_gray,
        gray_gamma=gray_gamma,
        background_fill_mode=background_fill_mode,
        background_sigma=background_sigma,
        occupancy_sigma=occupancy_sigma,
    )
    pred_strength = np.clip((pred_map - prob_threshold) / max(1.0 - prob_threshold, 1e-6), 0.0, 1.0)
    gt_strength = np.clip(gt_map, 0.0, 1.0)
    pred_binary = pred_map >= prob_threshold
    gt_binary = gt_map >= 0.5

    pred_binary = smooth_binary_mask(pred_binary, close_radius=mask_close_radius, dilate_radius=mask_dilate_radius)
    gt_binary = smooth_binary_mask(gt_binary, close_radius=mask_close_radius, dilate_radius=mask_dilate_radius)
    pred_strength = pred_strength * pred_binary.astype(np.float32)
    gt_strength = gt_strength * gt_binary.astype(np.float32)

    pred_boxes = extract_component_boxes(pred_binary, min_component_pixels=min_component_pixels, padding=box_padding)
    gt_boxes = extract_component_boxes(gt_binary, min_component_pixels=min_component_pixels, padding=box_padding)
    pred_boxes = merge_adjacent_boxes(pred_boxes, merge_gap=box_merge_gap)
    gt_boxes = merge_adjacent_boxes(gt_boxes, merge_gap=box_merge_gap)

    pred_image = apply_blue_overlay(base, pred_strength, alpha_min=0.40, alpha_max=0.95)
    gt_image = apply_blue_overlay(base, gt_strength, alpha_min=0.40, alpha_max=0.95)
    compare_strength = np.maximum(pred_strength, gt_strength)
    compare_image = apply_blue_overlay(base, compare_strength, alpha_min=0.32, alpha_max=0.88)

    pred_image = draw_dashed_boxes(pred_image, pred_boxes, PRED_BOX, box_thickness, dash_length, dash_gap)
    gt_image = draw_dashed_boxes(gt_image, gt_boxes, GT_BOX, box_thickness, dash_length, dash_gap)
    compare_image = draw_dashed_boxes(compare_image, pred_boxes, PRED_BOX, box_thickness, dash_length, dash_gap)
    compare_image = draw_dashed_boxes(compare_image, gt_boxes, GT_BOX, box_thickness, dash_length, dash_gap)
    return {
        "pred_image": pred_image,
        "gt_image": gt_image,
        "compare_image": compare_image,
        "pred_boxes": pred_boxes,
        "gt_boxes": gt_boxes,
        "pred_mask_pixels": int(np.sum(pred_binary)),
        "gt_mask_pixels": int(np.sum(gt_binary)),
    }


def compose_sample_panel(
    sample_title: str,
    projection_rows: list[np.ndarray],
    tile_gap: int,
) -> np.ndarray:
    title_image = add_title_band(
        np.full((12, projection_rows[0].shape[1], 3), CANVAS_BG, dtype=np.uint8),
        sample_title,
        band_height=32,
    )
    body = concat_v(projection_rows, gap=max(int(tile_gap), 0))
    return concat_v([title_image, body], gap=4)


def compose_gallery(compare_tiles: list[np.ndarray], labels: list[str], cols: int, tile_gap: int) -> np.ndarray:
    if not compare_tiles:
        raise ValueError("compare_tiles is empty")
    labeled = [add_title_band(tile, label, band_height=26) for tile, label in zip(compare_tiles, labels)]
    cols = max(int(cols), 1)
    rows = int(math.ceil(len(labeled) / cols))
    blank = np.full_like(labeled[0], CANVAS_BG, dtype=np.uint8)
    row_images: list[np.ndarray] = []
    for row in range(rows):
        start = row * cols
        end = min(start + cols, len(labeled))
        cells = labeled[start:end]
        if len(cells) < cols:
            cells = cells + [blank] * (cols - len(cells))
        row_images.append(concat_h(cells, gap=tile_gap))
    return concat_v(row_images, gap=tile_gap)


def main() -> None:
    args = parse_args()
    summary_path, output_dir = resolve_paths(args)
    summary = load_summary(summary_path)
    prob_threshold = float(summary.get("prob_threshold", 0.5)) if args.prob_threshold < 0 else float(args.prob_threshold)
    sample_records = summary.get("samples", [])
    if not sample_records:
        raise RuntimeError("Summary contains no samples.")

    if args.sample_indices is not None and len(args.sample_indices) > 0:
        selected_local_indices = [int(v) for v in args.sample_indices]
    else:
        selected_local_indices = list(range(len(sample_records)))

    invalid = [v for v in selected_local_indices if v < 0 or v >= len(sample_records)]
    if invalid:
        raise IndexError(f"Invalid sample_indices: {invalid}")

    print("=" * 80)
    print("Rendering dense 2D projection visualizations")
    print("=" * 80)
    print(f"summary_path={summary_path}")
    print(f"output_dir={output_dir}")
    print(f"projections={args.projections}")
    print(f"prob_threshold={prob_threshold}")

    gallery_compare: dict[str, list[np.ndarray]] = {projection: [] for projection in args.projections}
    gallery_labels: dict[str, list[str]] = {projection: [] for projection in args.projections}
    summary_records_out: list[dict] = []

    for local_idx in selected_local_indices:
        sample = sample_records[local_idx]
        dense_npz_path = Path(sample["files"]["dense_npz"]).resolve()
        dense_data = load_dense_npz(dense_npz_path)
        point_cloud = dense_data["point_cloud"]
        pred_prob = dense_data["leak_prob"]
        gt_label = dense_data["gt_label"]
        center_y, center_z, radius = fit_circle_yz(point_cloud)
        sample_dir = output_dir / Path(sample["files"]["sample_dir"]).name
        sample_dir.mkdir(parents=True, exist_ok=True)

        projection_rows: list[np.ndarray] = []
        projection_files: dict[str, dict[str, str | int]] = {}
        for projection in args.projections:
            rendered = render_projection_triplet(
                point_cloud=point_cloud,
                pred_prob=pred_prob,
                gt_label=gt_label,
                projection=projection,
                tile_width=int(args.tile_width),
                tile_height=int(args.tile_height),
                point_radius=int(args.point_radius),
                prob_threshold=prob_threshold,
                base_gray=float(args.base_gray),
                gray_gamma=float(args.gray_gamma),
                min_component_pixels=int(args.min_component_pixels),
                box_padding=int(args.box_padding),
                box_thickness=int(args.box_thickness),
                dash_length=int(args.dash_length),
                dash_gap=int(args.dash_gap),
                mask_close_radius=int(args.mask_close_radius),
                mask_dilate_radius=int(args.mask_dilate_radius),
                box_merge_gap=int(args.box_merge_gap),
                background_fill_mode=str(args.background_fill_mode),
                background_sigma=float(args.background_sigma),
                occupancy_sigma=float(args.occupancy_sigma),
            )

            compare_path = sample_dir / f"compare_{projection}.png"
            save_image(rendered["compare_image"], compare_path)
            file_record: dict[str, str | int] = {
                "compare_image": str(compare_path),
                "pred_box_count": int(len(rendered["pred_boxes"])),
                "gt_box_count": int(len(rendered["gt_boxes"])),
                "pred_mask_pixels": int(rendered["pred_mask_pixels"]),
                "gt_mask_pixels": int(rendered["gt_mask_pixels"]),
            }

            if not bool(args.save_compare_only):
                pred_path = sample_dir / f"prediction_{projection}.png"
                gt_path = sample_dir / f"ground_truth_{projection}.png"
                save_image(rendered["pred_image"], pred_path)
                save_image(rendered["gt_image"], gt_path)
                file_record["prediction_image"] = str(pred_path)
                file_record["ground_truth_image"] = str(gt_path)

            pred_tile = add_title_band(rendered["pred_image"], f"{projection.upper()} | Prediction", band_height=28)
            gt_tile = add_title_band(rendered["gt_image"], f"{projection.upper()} | Ground Truth", band_height=28)
            compare_tile = add_title_band(rendered["compare_image"], f"{projection.upper()} | Compare", band_height=28)
            row = concat_h([pred_tile, gt_tile, compare_tile], gap=int(args.tile_gap))
            projection_rows.append(row)
            projection_files[projection] = file_record

            gallery_compare[projection].append(rendered["compare_image"])
            gallery_labels[projection].append(
                f"Rank {int(sample['rank'])} | A{int(sample['area_id'])} ({int(sample['grid_x'])}, {int(sample['grid_y'])})"
            )

        sample_title = (
            f"Rank {int(sample['rank'])} | Area {int(sample['area_id'])} | "
            f"grid=({int(sample['grid_x'])}, {int(sample['grid_y'])}) | "
            f"dense_pred={float(sample['dense_pred_ratio']) * 100.0:.2f}% | "
            f"dense_gt={float(sample['dense_gt_ratio']) * 100.0:.2f}%"
        )
        sample_panel = compose_sample_panel(sample_title=sample_title, projection_rows=projection_rows, tile_gap=int(args.tile_gap))
        panel_path = sample_dir / "panel_all.png"
        save_image(sample_panel, panel_path)

        summary_records_out.append(
            {
                "summary_sample_index": int(local_idx),
                "rank": int(sample["rank"]),
                "index": int(sample["index"]),
                "area_id": int(sample["area_id"]),
                "grid_x": int(sample["grid_x"]),
                "grid_y": int(sample["grid_y"]),
                "panel_image": str(panel_path),
                "projections": projection_files,
                "cylindrical_fit": {
                    "center_y": float(center_y),
                    "center_z": float(center_z),
                    "radius": float(radius),
                },
            }
        )
        print(
            f"saved sample panel | rank={int(sample['rank'])} area={int(sample['area_id'])} "
            f"grid=({int(sample['grid_x'])}, {int(sample['grid_y'])})",
            flush=True,
        )

    gallery_files: dict[str, str] = {}
    if bool(args.save_gallery):
        cols = min(max(len(summary_records_out), 1), 4)
        for projection in args.projections:
            gallery_image = compose_gallery(
                compare_tiles=gallery_compare[projection],
                labels=gallery_labels[projection],
                cols=cols,
                tile_gap=int(args.tile_gap),
            )
            gallery_path = output_dir / f"gallery_compare_{projection}.png"
            save_image(gallery_image, gallery_path)
            gallery_files[projection] = str(gallery_path)
            print(f"saved {gallery_path}", flush=True)

    if bool(args.save_summary):
        summary_out = {
            "source_summary_path": str(summary_path),
            "prob_threshold": float(prob_threshold),
            "projections": list(args.projections),
            "tile_width": int(args.tile_width),
            "tile_height": int(args.tile_height),
            "point_radius": int(args.point_radius),
            "min_component_pixels": int(args.min_component_pixels),
            "box_padding": int(args.box_padding),
            "box_thickness": int(args.box_thickness),
            "dash_length": int(args.dash_length),
            "dash_gap": int(args.dash_gap),
            "mask_close_radius": int(args.mask_close_radius),
            "mask_dilate_radius": int(args.mask_dilate_radius),
            "box_merge_gap": int(args.box_merge_gap),
            "background_fill_mode": str(args.background_fill_mode),
            "background_sigma": float(args.background_sigma),
            "occupancy_sigma": float(args.occupancy_sigma),
            "selected_count": int(len(summary_records_out)),
            "gallery_files": gallery_files,
            "samples": summary_records_out,
        }
        summary_out_path = output_dir / "dense_projection_visualization_summary.json"
        with open(summary_out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary_out, f, indent=2, ensure_ascii=False)
        print(f"saved {summary_out_path}", flush=True)

    print("=" * 80)
    print(f"rendered_samples={len(summary_records_out)}")
    print(f"output_dir={output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
