from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from leakage_detection_torch.render_paper_figure import crop_content, fit_contain, render_overview_image
from leakage_detection_torch.visualize_dense_projections import (
    apply_blue_overlay,
    fill_invalid_pixels,
    fit_circle_yz,
    smooth_binary_mask,
)


PANEL_BG = (255, 255, 255)
BORDER = (226, 229, 234)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PPT-ready local 3D + texture-like unwrap showcases."
    )
    parser.add_argument("--dense_summary_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None)
    parser.add_argument("--tile_width", type=int, default=3600)
    parser.add_argument("--tile_height", type=int, default=1820)
    parser.add_argument("--prob_threshold", type=float, default=-1.0)
    parser.add_argument("--canvas_width", type=int, default=3000)
    parser.add_argument("--canvas_height", type=int, default=1180)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--panel_gap", type=int, default=26)
    parser.add_argument("--left_width", type=int, default=940)
    parser.add_argument("--overview_width", type=int, default=1850)
    parser.add_argument("--overview_height", type=int, default=2250)
    parser.add_argument(
        "--overview_mode",
        type=str,
        default="scatter",
        choices=["scatter", "surfel"],
        help="3D overview rendering mode.",
    )
    parser.add_argument("--overview_bg_point_size", type=float, default=0.95)
    parser.add_argument("--overview_fg_point_size", type=float, default=2.9)
    parser.add_argument("--overview_elev", type=float, default=18.0)
    parser.add_argument("--overview_azim", type=float, default=-64.0)
    parser.add_argument("--overview_supersample", type=float, default=1.35)
    parser.add_argument("--overview_perspective", type=float, default=0.10)
    parser.add_argument(
        "--overview_axis_mode",
        type=str,
        default="xz_y",
        choices=["xy_z", "xz_y", "yz_x"],
        help="Projection axes used by surfel overview renderer.",
    )
    parser.add_argument("--overview_smooth_sigma", type=float, default=1.35)
    parser.add_argument("--overview_coarse_sigma", type=float, default=3.8)
    parser.add_argument("--overview_detail_sigma", type=float, default=6.2)
    parser.add_argument("--overview_support_sigma", type=float, default=2.2)
    parser.add_argument("--overview_detail_amount", type=float, default=0.46)
    parser.add_argument("--overview_gray_floor", type=float, default=0.72)
    parser.add_argument("--overview_gray_gain", type=float, default=0.24)
    parser.add_argument("--overview_pred_sigma", type=float, default=1.05)
    parser.add_argument("--overview_pred_focus_threshold", type=float, default=0.17)
    parser.add_argument("--overview_pred_min_strength", type=float, default=0.82)
    parser.add_argument("--overview_pred_close_radius", type=int, default=2)
    parser.add_argument("--overview_pred_dilate_radius", type=int, default=1)
    parser.add_argument("--overview_unsharp_radius", type=float, default=1.3)
    parser.add_argument("--overview_unsharp_percent", type=int, default=142)
    parser.add_argument("--overview_unsharp_threshold", type=int, default=2)
    parser.add_argument("--smooth_sigma", type=float, default=1.25)
    parser.add_argument("--coarse_sigma", type=float, default=3.4)
    parser.add_argument("--detail_sigma", type=float, default=7.0)
    parser.add_argument("--support_sigma", type=float, default=2.2)
    parser.add_argument("--detail_amount", type=float, default=0.44)
    parser.add_argument("--contrast_gamma", type=float, default=0.88)
    parser.add_argument("--gray_floor", type=float, default=0.76)
    parser.add_argument("--gray_gain", type=float, default=0.20)
    parser.add_argument("--support_threshold", type=float, default=0.0005)
    parser.add_argument("--surface_close_radius", type=int, default=2)
    parser.add_argument("--surface_dilate_radius", type=int, default=2)
    parser.add_argument("--pred_close_radius", type=int, default=3)
    parser.add_argument("--pred_dilate_radius", type=int, default=2)
    parser.add_argument("--pred_sigma", type=float, default=0.85)
    parser.add_argument("--pred_focus_threshold", type=float, default=0.18)
    parser.add_argument("--pred_min_strength", type=float, default=0.80)
    parser.add_argument("--pred_min_component_pixels", type=int, default=1800)
    parser.add_argument("--pred_alpha_min", type=float, default=0.48)
    parser.add_argument("--pred_alpha_max", type=float, default=0.98)
    parser.add_argument("--edge_margin_pixels", type=float, default=12.0)
    parser.add_argument("--unsharp_radius", type=float, default=1.6)
    parser.add_argument("--unsharp_percent", type=int, default=138)
    parser.add_argument("--unsharp_threshold", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def project_x_theta_seam_aware(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(point_cloud[:, 0], dtype=np.float32)
    y = np.asarray(point_cloud[:, 1], dtype=np.float64)
    z = np.asarray(point_cloud[:, 2], dtype=np.float64)
    center_y, center_z, _ = fit_circle_yz(point_cloud)
    theta = np.arctan2(z - center_z, y - center_y)
    if len(theta) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    sorted_theta = np.sort(theta)
    wrapped = np.concatenate([sorted_theta, sorted_theta[:1] + 2.0 * np.pi])
    gaps = np.diff(wrapped)
    seam_idx = int(np.argmax(gaps))
    seam_angle = float((wrapped[seam_idx] + wrapped[seam_idx + 1]) * 0.5)
    theta_shift = (theta - seam_angle + np.pi) % (2.0 * np.pi)
    theta_span = max(2.0 * np.pi - float(gaps[seam_idx]), 1e-6)

    h = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    v = np.clip(theta_shift / theta_span, 0.0, 1.0).astype(np.float32)
    return np.clip(h.astype(np.float32), 0.0, 1.0), v


def robust_normalize(values: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(np.percentile(values, low_q))
    high = float(np.percentile(values, high_q))
    if high <= low + 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    clipped = np.clip(values, low, high)
    return np.clip((clipped - low) / (high - low), 0.0, 1.0).astype(np.float32)


def gaussian_normalized(sum_map: np.ndarray, weight_map: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 1e-6:
        out = np.zeros_like(sum_map, dtype=np.float32)
        valid = weight_map > 1e-8
        out[valid] = sum_map[valid] / weight_map[valid]
        return out
    blur_sum = ndimage.gaussian_filter(sum_map.astype(np.float32), sigma=float(sigma), mode="nearest")
    blur_weight = ndimage.gaussian_filter(weight_map.astype(np.float32), sigma=float(sigma), mode="nearest")
    out = np.zeros_like(sum_map, dtype=np.float32)
    valid = blur_weight > 1e-6
    out[valid] = blur_sum[valid] / blur_weight[valid]
    return out


def build_view_rotation(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(float(elev_deg))
    azim = np.deg2rad(float(azim_deg))
    rz = np.asarray(
        [
            [np.cos(azim), -np.sin(azim), 0.0],
            [np.sin(azim), np.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev), -np.sin(elev)],
            [0.0, np.sin(elev), np.cos(elev)],
        ],
        dtype=np.float32,
    )
    return rx @ rz


def project_view_coords(
    point_cloud: np.ndarray,
    elev_deg: float,
    azim_deg: float,
    perspective: float,
    axis_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(point_cloud[:, :3], dtype=np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    rotation = build_view_rotation(elev_deg=elev_deg, azim_deg=azim_deg)
    view = centered @ rotation.T
    axis_lookup = {
        "xy_z": (0, 1, 2),
        "xz_y": (0, 2, 1),
        "yz_x": (1, 2, 0),
    }
    ui, vi, di = axis_lookup.get(str(axis_mode), (0, 2, 1))
    u = np.asarray(view[:, ui], dtype=np.float32)
    v = np.asarray(view[:, vi], dtype=np.float32)
    depth = np.asarray(view[:, di], dtype=np.float32)

    if abs(float(perspective)) > 1e-6:
        depth_norm = (depth - float(np.mean(depth))) / max(float(np.std(depth)), 1e-6)
        scale = 1.0 / (1.0 + float(perspective) * depth_norm)
        u = u * scale.astype(np.float32)
        v = v * scale.astype(np.float32)
    return u, v, depth


def build_overview_maps(
    point_cloud: np.ndarray,
    pred_prob: np.ndarray,
    width: int,
    height: int,
    prob_threshold: float,
    elev_deg: float,
    azim_deg: float,
    perspective: float,
    axis_mode: str,
) -> dict[str, np.ndarray]:
    u, v, depth = project_view_coords(
        point_cloud=point_cloud,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        perspective=perspective,
        axis_mode=axis_mode,
    )
    u_min, u_max = float(np.min(u)), float(np.max(u))
    v_min, v_max = float(np.min(v)), float(np.max(v))
    u_span = max(u_max - u_min, 1e-6)
    v_span = max(v_max - v_min, 1e-6)
    pad = 0.035
    px = ((u - u_min) / u_span) * (1.0 - 2.0 * pad) + pad
    py = ((v - v_min) / v_span) * (1.0 - 2.0 * pad) + pad
    px = px * max(int(width) - 1, 1)
    py = (1.0 - py) * max(int(height) - 1, 1)

    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (px - x0).astype(np.float32)
    wy = (py - y0).astype(np.float32)

    intensity = point_cloud[:, 3] if point_cloud.shape[1] > 3 else np.ones((len(point_cloud),), dtype=np.float32)
    intensity = robust_normalize(intensity)
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    pred_seed = (pred_prob >= float(prob_threshold)).astype(np.float32)
    depth_norm = (depth - float(np.min(depth))) / max(float(np.max(depth) - np.min(depth)), 1e-6)
    frontness = np.power(1.0 - depth_norm, 1.3).astype(np.float32)
    frontness = 0.22 + 0.78 * frontness

    intensity_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    pred_sum = np.zeros((height, width), dtype=np.float32)
    pred_weight = np.zeros((height, width), dtype=np.float32)
    pred_seed_sum = np.zeros((height, width), dtype=np.float32)

    for yy, xx, ww in (
        (y0, x0, (1.0 - wx) * (1.0 - wy)),
        (y0, x1, wx * (1.0 - wy)),
        (y1, x0, (1.0 - wx) * wy),
        (y1, x1, wx * wy),
    ):
        ww = ww.astype(np.float32, copy=False) * frontness
        np.add.at(intensity_sum, (yy, xx), intensity * ww)
        np.add.at(weight_sum, (yy, xx), ww)
        np.add.at(pred_sum, (yy, xx), pred_prob * ww)
        np.add.at(pred_weight, (yy, xx), ww)
        np.add.at(pred_seed_sum, (yy, xx), pred_seed * ww)

    return {
        "intensity_sum": intensity_sum,
        "weight_sum": weight_sum,
        "pred_sum": pred_sum,
        "pred_weight": pred_weight,
        "pred_seed_sum": pred_seed_sum,
    }


def render_dense_local_overview_surfel(point_cloud: np.ndarray, pred_prob: np.ndarray, args: argparse.Namespace) -> Image.Image:
    out_w = int(getattr(args, "overview_width", 1850))
    out_h = int(getattr(args, "overview_height", 2250))
    supersample = max(float(getattr(args, "overview_supersample", 1.35)), 1.0)
    width = max(32, int(round(out_w * supersample)))
    height = max(32, int(round(out_h * supersample)))

    maps = build_overview_maps(
        point_cloud=point_cloud,
        pred_prob=pred_prob,
        width=width,
        height=height,
        prob_threshold=float(getattr(args, "prob_threshold", 0.5)),
        elev_deg=float(getattr(args, "overview_elev", 18.0)),
        azim_deg=float(getattr(args, "overview_azim", -64.0)),
        perspective=float(getattr(args, "overview_perspective", 0.10)),
        axis_mode=str(getattr(args, "overview_axis_mode", "xz_y")),
    )

    weight_sum = maps["weight_sum"]
    valid = weight_sum > 1e-6
    crisp = np.zeros_like(weight_sum, dtype=np.float32)
    crisp[valid] = maps["intensity_sum"][valid] / weight_sum[valid]
    crisp = fill_invalid_pixels(crisp, valid, mode="nearest")

    smooth = gaussian_normalized(maps["intensity_sum"], weight_sum, sigma=float(getattr(args, "overview_smooth_sigma", 1.35)))
    smooth = fill_invalid_pixels(smooth, valid, mode="nearest")
    coarse = gaussian_normalized(maps["intensity_sum"], weight_sum, sigma=float(getattr(args, "overview_coarse_sigma", 3.8)))
    coarse = fill_invalid_pixels(coarse, valid, mode="nearest")

    texture = 0.60 * smooth + 0.40 * coarse
    detail_base = ndimage.gaussian_filter(
        crisp.astype(np.float32),
        sigma=float(getattr(args, "overview_detail_sigma", 6.2)),
        mode="nearest",
    )
    texture = np.clip(
        texture + float(getattr(args, "overview_detail_amount", 0.46)) * (crisp - detail_base),
        0.0,
        1.0,
    )

    support = ndimage.gaussian_filter(
        weight_sum.astype(np.float32),
        sigma=float(getattr(args, "overview_support_sigma", 2.2)),
        mode="nearest",
    )
    support = support / max(float(support.max()), 1e-6)
    surface_mask = support > 0.0005
    surface_mask = ndimage.binary_fill_holes(surface_mask)
    surface_mask = smooth_binary_mask(surface_mask, close_radius=2, dilate_radius=2)
    if np.any(surface_mask):
        low = float(np.percentile(texture[surface_mask], 0.8))
        high = float(np.percentile(texture[surface_mask], 99.2))
        texture = np.clip((np.clip(texture, low, high) - low) / max(high - low, 1e-6), 0.0, 1.0)

    gray_map = np.clip(
        float(getattr(args, "overview_gray_floor", 0.72))
        + float(getattr(args, "overview_gray_gain", 0.24)) * np.power(np.clip(texture, 0.0, 1.0), 0.90),
        0.0,
        1.0,
    )
    base = np.repeat((gray_map * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    base[~surface_mask] = 255

    pred_soft = gaussian_normalized(
        maps["pred_sum"],
        maps["pred_weight"],
        sigma=float(getattr(args, "overview_pred_sigma", 1.05)),
    )
    pred_soft = np.clip(
        (pred_soft - float(getattr(args, "prob_threshold", 0.5))) / max(1.0 - float(getattr(args, "prob_threshold", 0.5)), 1e-6),
        0.0,
        1.0,
    )
    pred_binary = gaussian_normalized(
        maps["pred_seed_sum"],
        maps["pred_weight"],
        sigma=float(getattr(args, "overview_pred_sigma", 1.05)),
    ) >= float(getattr(args, "overview_pred_focus_threshold", 0.17))
    pred_binary = smooth_binary_mask(
        pred_binary,
        close_radius=int(getattr(args, "overview_pred_close_radius", 2)),
        dilate_radius=int(getattr(args, "overview_pred_dilate_radius", 1)),
    )
    pred_strength = np.maximum(
        pred_soft,
        float(getattr(args, "overview_pred_min_strength", 0.82)) * pred_binary.astype(np.float32),
    )
    pred_strength *= pred_binary.astype(np.float32)
    pred_strength *= surface_mask.astype(np.float32)
    pred_strength = np.clip(pred_strength, 0.0, 1.0)

    image = apply_blue_overlay(
        base_image=base,
        mask_strength=pred_strength,
        alpha_min=0.50,
        alpha_max=0.98,
    )
    pil = Image.fromarray(image, mode="RGB")
    pil = pil.filter(
        ImageFilter.UnsharpMask(
            radius=float(getattr(args, "overview_unsharp_radius", 1.3)),
            percent=int(getattr(args, "overview_unsharp_percent", 142)),
            threshold=int(getattr(args, "overview_unsharp_threshold", 2)),
        )
    )
    pil = pil.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return pil


def filter_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=bool)
    if not np.any(binary):
        return binary
    if int(min_pixels) <= 1:
        return binary
    labels, count = ndimage.label(binary)
    if count <= 0:
        return binary
    kept = np.zeros_like(binary, dtype=bool)
    for comp_id in range(1, count + 1):
        comp = labels == comp_id
        if int(np.sum(comp)) >= int(min_pixels):
            kept |= comp
    return kept


def build_texture_maps(
    point_cloud: np.ndarray,
    pred_prob: np.ndarray,
    tile_width: int,
    tile_height: int,
    prob_threshold: float,
    edge_margin_pixels: float,
) -> dict[str, np.ndarray]:
    h_norm, v_norm = project_x_theta_seam_aware(point_cloud)
    px = h_norm * max(int(tile_width) - 1, 1)
    py = (1.0 - v_norm) * max(int(tile_height) - 1, 1)
    margin = max(float(edge_margin_pixels), 0.0)
    px = np.clip(px, margin, max(float(tile_width - 1) - margin, margin))
    py = np.clip(py, margin, max(float(tile_height - 1) - margin, margin))
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, tile_width - 1)
    y1 = np.clip(y0 + 1, 0, tile_height - 1)
    wx = (px - x0).astype(np.float32)
    wy = (py - y0).astype(np.float32)

    intensity = point_cloud[:, 3] if point_cloud.shape[1] > 3 else np.ones((len(point_cloud),), dtype=np.float32)
    intensity = robust_normalize(intensity)
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    pred_seed = (pred_prob >= float(prob_threshold)).astype(np.float32)

    intensity_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
    weight_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
    pred_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
    pred_weight = np.zeros((tile_height, tile_width), dtype=np.float32)
    pred_seed_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
    hit_count = np.zeros((tile_height, tile_width), dtype=np.float32)

    for yy, xx, ww in (
        (y0, x0, (1.0 - wx) * (1.0 - wy)),
        (y0, x1, wx * (1.0 - wy)),
        (y1, x0, (1.0 - wx) * wy),
        (y1, x1, wx * wy),
    ):
        ww = ww.astype(np.float32, copy=False)
        np.add.at(intensity_sum, (yy, xx), intensity * ww)
        np.add.at(weight_sum, (yy, xx), ww)
        np.add.at(pred_sum, (yy, xx), pred_prob * ww)
        np.add.at(pred_weight, (yy, xx), ww)
        np.add.at(pred_seed_sum, (yy, xx), pred_seed * ww)
        np.add.at(hit_count, (yy, xx), ww)

    return {
        "intensity_sum": intensity_sum,
        "weight_sum": weight_sum,
        "pred_sum": pred_sum,
        "pred_weight": pred_weight,
        "pred_seed_sum": pred_seed_sum,
        "hit_count": hit_count,
    }


def render_texture_unwrap(point_cloud: np.ndarray, pred_prob: np.ndarray, args: argparse.Namespace) -> Image.Image:
    maps = build_texture_maps(
        point_cloud=point_cloud,
        pred_prob=pred_prob,
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        prob_threshold=float(args.prob_threshold),
        edge_margin_pixels=float(args.edge_margin_pixels),
    )
    weight_sum = maps["weight_sum"]
    valid = weight_sum > 1e-6

    crisp = np.zeros_like(weight_sum, dtype=np.float32)
    crisp[valid] = maps["intensity_sum"][valid] / weight_sum[valid]
    crisp = fill_invalid_pixels(crisp, valid, mode="nearest")

    smooth = gaussian_normalized(maps["intensity_sum"], weight_sum, sigma=float(args.smooth_sigma))
    smooth = fill_invalid_pixels(smooth, valid, mode="nearest")
    coarse = gaussian_normalized(maps["intensity_sum"], weight_sum, sigma=float(args.coarse_sigma))
    coarse = fill_invalid_pixels(coarse, valid, mode="nearest")

    texture = 0.58 * smooth + 0.42 * coarse
    detail_base = ndimage.gaussian_filter(crisp.astype(np.float32), sigma=float(args.detail_sigma), mode="nearest")
    texture = np.clip(texture + float(args.detail_amount) * (crisp - detail_base), 0.0, 1.0)

    support = ndimage.gaussian_filter(weight_sum.astype(np.float32), sigma=float(args.support_sigma), mode="nearest")
    support = support / max(float(support.max()), 1e-6)
    surface_mask = support > float(args.support_threshold)
    surface_mask = ndimage.binary_fill_holes(surface_mask)
    surface_mask = smooth_binary_mask(
        surface_mask,
        close_radius=int(args.surface_close_radius),
        dilate_radius=int(args.surface_dilate_radius),
    )
    if not np.any(surface_mask):
        surface_mask = support > max(float(args.support_threshold) * 0.25, 1e-6)

    if np.any(surface_mask):
        low = float(np.percentile(texture[surface_mask], 0.8))
        high = float(np.percentile(texture[surface_mask], 99.2))
        texture = np.clip((np.clip(texture, low, high) - low) / max(high - low, 1e-6), 0.0, 1.0)

    texture = np.power(np.clip(texture, 0.0, 1.0), float(args.contrast_gamma))
    gray_map = np.clip(float(args.gray_floor) + float(args.gray_gain) * texture, 0.0, 1.0)
    gray_rgb = (gray_map * 255.0).astype(np.uint8)
    base = np.repeat(gray_rgb[..., None], 3, axis=2)
    base[~surface_mask] = 255

    pred_soft = gaussian_normalized(maps["pred_sum"], maps["pred_weight"], sigma=0.85)
    pred_soft = np.clip(
        (pred_soft - float(args.prob_threshold)) / max(1.0 - float(args.prob_threshold), 1e-6),
        0.0,
        1.0,
    )
    pred_binary = gaussian_normalized(maps["pred_seed_sum"], maps["pred_weight"], sigma=float(args.pred_sigma))
    pred_binary = pred_binary >= float(args.pred_focus_threshold)
    pred_binary = smooth_binary_mask(
        pred_binary,
        close_radius=int(args.pred_close_radius),
        dilate_radius=int(args.pred_dilate_radius),
    )
    pred_binary = filter_small_components(pred_binary, min_pixels=int(args.pred_min_component_pixels))
    pred_strength = np.maximum(pred_soft, float(args.pred_min_strength) * pred_binary.astype(np.float32))
    pred_strength *= pred_binary.astype(np.float32)
    pred_strength *= surface_mask.astype(np.float32)
    pred_strength = np.clip(pred_strength, 0.0, 1.0)

    image = apply_blue_overlay(
        base_image=base,
        mask_strength=pred_strength,
        alpha_min=float(args.pred_alpha_min),
        alpha_max=float(args.pred_alpha_max),
    )
    pil = Image.fromarray(image, mode="RGB")
    pil = pil.filter(
        ImageFilter.UnsharpMask(
            radius=float(args.unsharp_radius),
            percent=int(args.unsharp_percent),
            threshold=int(args.unsharp_threshold),
        )
    )
    return pil


def render_dense_local_overview(point_cloud: np.ndarray, pred_prob: np.ndarray, args: argparse.Namespace) -> Image.Image:
    if str(getattr(args, "overview_mode", "scatter")).lower() == "surfel":
        return render_dense_local_overview_surfel(point_cloud=point_cloud, pred_prob=pred_prob, args=args)
    pred_label = (pred_prob >= float(args.prob_threshold)).astype(np.uint8)
    rng = np.random.default_rng(20260330)
    return render_overview_image(
        point_cloud=point_cloud.astype(np.float32, copy=False),
        pred_label=pred_label,
        output_size=(int(args.overview_width), int(args.overview_height)),
        max_background_points=max(int(len(point_cloud)), 1),
        max_positive_points=max(int(np.sum(pred_label > 0)), 1),
        bg_point_size=float(args.overview_bg_point_size),
        fg_point_size=float(args.overview_fg_point_size),
        elev=float(args.overview_elev),
        azim=float(args.overview_azim),
        rng=rng,
    )


def add_border(image: Image.Image, width: int = 1) -> Image.Image:
    out = image.copy()
    px = out.load()
    w, h = out.size
    for offset in range(width):
        for x in range(offset, w - offset):
            px[x, offset] = BORDER
            px[x, h - 1 - offset] = BORDER
        for y in range(offset, h - offset):
            px[offset, y] = BORDER
            px[w - 1 - offset, y] = BORDER
    return out


def compose_pure_showcase(
    overview_image: Image.Image,
    unwrap_image: Image.Image,
    canvas_width: int,
    canvas_height: int,
    left_width: int,
    margin: int,
    panel_gap: int,
) -> Image.Image:
    canvas = Image.new("RGB", (canvas_width, canvas_height), PANEL_BG)
    inner_height = canvas_height - margin * 2
    right_width = canvas_width - margin * 2 - panel_gap - left_width

    overview_fit = fit_contain(crop_content(overview_image, threshold=247, padding=18), left_width, inner_height)
    unwrap_fit = fit_contain(crop_content(unwrap_image, threshold=250, padding=12), right_width, inner_height)

    overview_fit = add_border(overview_fit, width=1)
    unwrap_fit = add_border(unwrap_fit, width=1)

    x_left = margin
    x_right = margin + left_width + panel_gap
    y_left = margin + (inner_height - overview_fit.height) // 2
    y_right = margin + (inner_height - unwrap_fit.height) // 2
    canvas.paste(overview_fit, (x_left, y_left))
    canvas.paste(unwrap_fit, (x_right, y_right))
    return canvas


def main() -> None:
    args = parse_args()
    summary_path = args.dense_summary_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)
    threshold = float(summary.get("prob_threshold", 0.5)) if float(args.prob_threshold) < 0 else float(args.prob_threshold)
    args.prob_threshold = threshold

    samples = summary.get("samples", [])
    if not samples:
        raise RuntimeError("dense summary has no samples")
    if args.sample_indices is None:
        selected = list(range(min(3, len(samples))))
    else:
        selected = [int(v) for v in args.sample_indices]

    saved: list[dict[str, object]] = []
    for local_idx in selected:
        sample = samples[local_idx]
        npz_path = Path(sample["files"]["dense_npz"]).resolve()
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing dense npz: {npz_path}")
        npz = np.load(npz_path)
        point_cloud = np.asarray(npz["point_cloud"], dtype=np.float32)
        pred_prob = np.asarray(npz["leak_prob"], dtype=np.float32)

        overview_image = render_dense_local_overview(point_cloud=point_cloud, pred_prob=pred_prob, args=args)
        unwrap_image = render_texture_unwrap(point_cloud=point_cloud, pred_prob=pred_prob, args=args)
        showcase = compose_pure_showcase(
            overview_image=overview_image,
            unwrap_image=unwrap_image,
            canvas_width=int(args.canvas_width),
            canvas_height=int(args.canvas_height),
            left_width=int(args.left_width),
            margin=int(args.margin),
            panel_gap=int(args.panel_gap),
        )

        stem = f"showcase_rank_{int(sample['rank']):02d}_area_{int(sample['area_id']):02d}_gx_{int(sample['grid_x']):03d}_gy_{int(sample['grid_y']):02d}"
        out_path = output_dir / f"{stem}.png"
        showcase.save(out_path)
        unwrap_path = output_dir / f"{stem}_unwrap_texture.png"
        unwrap_image.save(unwrap_path)
        overview_path = output_dir / f"{stem}_overview_3d.png"
        overview_image.save(overview_path)

        saved.append(
            {
                "rank": int(sample["rank"]),
                "area_id": int(sample["area_id"]),
                "grid_x": int(sample["grid_x"]),
                "grid_y": int(sample["grid_y"]),
                "image": str(out_path),
                "unwrap_texture": str(unwrap_path),
                "overview_3d": str(overview_path),
            }
        )
        print(f"saved {out_path}", flush=True)

    summary_out = {
        "dense_summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "saved_count": len(saved),
        "figures": saved,
    }
    summary_path_out = output_dir / "ppt_texture_showcase_summary.json"
    summary_path_out.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {summary_path_out}", flush=True)


if __name__ == "__main__":
    main()
