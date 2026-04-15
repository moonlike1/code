from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from leakage_detection_torch.render_paper_figure import crop_content, fit_contain, load_font, render_overview_image
from leakage_detection_torch.visualize_dense_projections import (
    apply_blue_overlay,
    compose_gray_background,
    fit_circle_yz,
    smooth_binary_mask,
)


FIG_BG = (247, 245, 240)
PANEL_BG = (255, 255, 255)
TEXT = (34, 36, 40)
MUTED = (108, 114, 122)
BORDER = (222, 225, 230)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PPT-friendly 3D + unwrap showcase figures.")
    parser.add_argument("--dense_summary_path", type=Path, required=True)
    parser.add_argument("--overview_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--tile_width", type=int, default=2200)
    parser.add_argument("--tile_height", type=int, default=1200)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--prob_threshold", type=float, default=-1.0)
    parser.add_argument("--base_gray", type=float, default=0.58)
    parser.add_argument("--gray_gamma", type=float, default=0.78)
    parser.add_argument("--background_fill_mode", type=str, default="nearest")
    parser.add_argument("--background_sigma", type=float, default=1.8)
    parser.add_argument("--occupancy_sigma", type=float, default=1.6)
    parser.add_argument("--mask_close_radius", type=int, default=3)
    parser.add_argument("--mask_dilate_radius", type=int, default=2)
    parser.add_argument("--pred_sigma", type=float, default=1.0)
    parser.add_argument("--pred_min_strength", type=float, default=0.72)
    parser.add_argument("--occupancy_close_radius", type=int, default=8)
    parser.add_argument("--occupancy_dilate_radius", type=int, default=6)
    parser.add_argument("--canvas_width", type=int, default=2400)
    parser.add_argument("--canvas_height", type=int, default=1220)
    parser.add_argument("--left_width", type=int, default=760)
    parser.add_argument("--page_margin", type=int, default=42)
    parser.add_argument("--panel_gap", type=int, default=28)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None)
    parser.add_argument("--overview_width", type=int, default=1400)
    parser.add_argument("--overview_height", type=int, default=2000)
    parser.add_argument("--overview_bg_point_size", type=float, default=0.95)
    parser.add_argument("--overview_fg_point_size", type=float, default=2.7)
    parser.add_argument("--overview_elev", type=float, default=18.0)
    parser.add_argument("--overview_azim", type=float, default=-64.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_border(image: Image.Image, width: int = 1) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for offset in range(width):
        draw.rectangle([offset, offset, out.width - 1 - offset, out.height - 1 - offset], outline=BORDER, width=1)
    return out


def add_caption(image: Image.Image, caption: str, font: ImageFont.ImageFont) -> Image.Image:
    cap_h = 50
    canvas = Image.new("RGB", (image.width, image.height + cap_h), PANEL_BG)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, image.height + 10), caption, font=font, fill=MUTED)
    return add_border(canvas)


def render_unwrap_prediction(
    point_cloud: np.ndarray,
    pred_prob: np.ndarray,
    gt_label: np.ndarray,
    args: argparse.Namespace,
) -> Image.Image:
    gray_sum, gray_count, pred_map = build_projection_maps_seam_aware(
        point_cloud=point_cloud,
        pred_prob=pred_prob,
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        point_radius=int(args.point_radius),
    )
    base = compose_gray_background(
        gray_sum=gray_sum,
        gray_count=gray_count,
        base_gray=float(args.base_gray),
        gray_gamma=float(args.gray_gamma),
        background_fill_mode=str(args.background_fill_mode),
        background_sigma=float(args.background_sigma),
        occupancy_sigma=float(args.occupancy_sigma),
    )
    threshold = float(args.prob_threshold)
    pred_strength = np.clip((pred_map - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    pred_binary = pred_map >= threshold
    pred_binary = smooth_binary_mask(
        pred_binary,
        close_radius=int(args.mask_close_radius),
        dilate_radius=int(args.mask_dilate_radius),
    )
    occupancy_mask = smooth_binary_mask(
        gray_count > 0,
        close_radius=int(args.occupancy_close_radius),
        dilate_radius=int(args.occupancy_dilate_radius),
    )
    pred_strength = np.maximum(pred_strength, float(args.pred_min_strength) * pred_binary.astype(np.float32))
    if float(args.pred_sigma) > 1e-6:
        pred_strength = ndimage.gaussian_filter(pred_strength.astype(np.float32), sigma=float(args.pred_sigma), mode="nearest")
    pred_strength = np.clip(pred_strength, 0.0, 1.0)
    pred_strength = pred_strength * occupancy_mask.astype(np.float32)
    pred_image = apply_blue_overlay(base, pred_strength, alpha_min=0.42, alpha_max=0.96)
    pred_array = np.asarray(pred_image, dtype=np.uint8)
    pred_array[~occupancy_mask] = 255
    return Image.fromarray(pred_array, mode="RGB")


def build_projection_maps_seam_aware(
    point_cloud: np.ndarray,
    pred_prob: np.ndarray,
    tile_width: int,
    tile_height: int,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_norm, v_norm = project_x_theta_seam_aware(point_cloud)
    x_pix = np.rint(h_norm * max(int(tile_width) - 1, 1)).astype(np.int32)
    y_pix = np.rint((1.0 - v_norm) * max(int(tile_height) - 1, 1)).astype(np.int32)

    intensity = point_cloud[:, 3] if point_cloud.shape[1] > 3 else np.ones((len(point_cloud),), dtype=np.float32)
    intensity = intensity.astype(np.float32, copy=False)
    low = float(np.percentile(intensity, 1.0))
    high = float(np.percentile(intensity, 99.0))
    if high <= low + 1e-6:
        intensity_norm = np.full_like(intensity, 0.5, dtype=np.float32)
    else:
        intensity_norm = np.clip((np.clip(intensity, low, high) - low) / (high - low), 0.0, 1.0).astype(np.float32)

    gray_sum = np.zeros((tile_height, tile_width), dtype=np.float32)
    gray_count = np.zeros((tile_height, tile_width), dtype=np.float32)
    pred_map = np.zeros((tile_height, tile_width), dtype=np.float32)

    radius = max(int(point_radius), 0)
    pred_prob = pred_prob.astype(np.float32, copy=False)
    for dy in range(-radius, radius + 1):
        yy = np.clip(y_pix + dy, 0, tile_height - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(x_pix + dx, 0, tile_width - 1)
            np.add.at(gray_sum, (yy, xx), intensity_norm)
            np.add.at(gray_count, (yy, xx), 1.0)
            np.maximum.at(pred_map, (yy, xx), pred_prob)
    return gray_sum, gray_count, pred_map


def project_x_theta_seam_aware(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(point_cloud[:, 0], dtype=np.float32)
    y = np.asarray(point_cloud[:, 1], dtype=np.float64)
    z = np.asarray(point_cloud[:, 2], dtype=np.float64)
    center_y, center_z, _ = fit_circle_yz(point_cloud)
    theta = np.arctan2(z - center_z, y - center_y)
    sorted_theta = np.sort(theta)
    if len(sorted_theta) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    wrapped = np.concatenate([sorted_theta, sorted_theta[:1] + 2.0 * np.pi])
    gaps = np.diff(wrapped)
    seam_idx = int(np.argmax(gaps))
    seam_angle = float((wrapped[seam_idx] + wrapped[seam_idx + 1]) * 0.5)
    theta_shift = (theta - seam_angle + np.pi) % (2.0 * np.pi)
    theta_span = 2.0 * np.pi - float(gaps[seam_idx])
    theta_span = max(theta_span, 1e-6)

    h = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    v = np.clip(theta_shift / theta_span, 0.0, 1.0).astype(np.float32)
    return np.clip(h.astype(np.float32), 0.0, 1.0), v


def render_dense_local_overview(point_cloud: np.ndarray, pred_prob: np.ndarray, args: argparse.Namespace) -> Image.Image:
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


def compose_showcase(
    overview_image: Image.Image,
    unwrap_image: Image.Image,
    title: str,
    subtitle: str,
    canvas_width: int,
    canvas_height: int,
    left_width: int,
    margin: int,
    panel_gap: int,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    caption_font: ImageFont.ImageFont,
) -> Image.Image:
    canvas = Image.new("RGB", (canvas_width, canvas_height), FIG_BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 18), title, font=title_font, fill=TEXT)
    draw.text((margin, 64), subtitle, font=subtitle_font, fill=MUTED)
    draw.line([(margin, 102), (canvas_width - margin, 102)], fill=BORDER, width=2)

    body_top = 132
    body_h = canvas_height - body_top - margin
    right_width = canvas_width - margin * 2 - panel_gap - left_width

    left_inner_h = body_h - 54
    right_inner_h = body_h - 54

    overview_fit = fit_contain(crop_content(overview_image, threshold=247, padding=18), left_width, left_inner_h)
    unwrap_fit = fit_contain(crop_content(unwrap_image, threshold=247, padding=18), right_width, right_inner_h)

    left_panel = add_caption(overview_fit, "3D local patch | blue = predicted leakage", caption_font)
    right_panel = add_caption(unwrap_fit, "Surface unwrap (x-theta) | blue = predicted leakage", caption_font)

    x_left = margin
    x_right = margin + left_panel.width + panel_gap
    y_panel = body_top + (body_h - max(left_panel.height, right_panel.height)) // 2
    canvas.paste(left_panel, (x_left, y_panel))
    canvas.paste(right_panel, (x_right, y_panel))
    return canvas


def main() -> None:
    args = parse_args()
    dense_summary_path = args.dense_summary_path.resolve()
    overview_root = args.overview_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else dense_summary_path.parent / "ppt_showcase_pairs")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(dense_summary_path)
    threshold = float(summary.get("prob_threshold", 0.5)) if float(args.prob_threshold) < 0 else float(args.prob_threshold)
    args.prob_threshold = threshold

    samples = summary.get("samples", [])
    if not samples:
        raise RuntimeError("dense summary has no samples")
    if args.sample_indices is None:
        selected = list(range(min(3, len(samples))))
    else:
        selected = [int(v) for v in args.sample_indices]

    fonts = {
        "title": load_font(36, bold=True),
        "subtitle": load_font(21, bold=False),
        "caption": load_font(20, bold=False),
    }

    saved = []
    for local_idx in selected:
        sample = samples[local_idx]
        npz = np.load(sample["files"]["dense_npz"])
        point_cloud = np.asarray(npz["point_cloud"], dtype=np.float32)
        pred_prob = np.asarray(npz["leak_prob"], dtype=np.float32)
        gt_label = np.asarray(npz["gt_label"], dtype=np.uint8)

        overview_image = render_dense_local_overview(point_cloud=point_cloud, pred_prob=pred_prob, args=args)
        unwrap_image = render_unwrap_prediction(point_cloud=point_cloud, pred_prob=pred_prob, gt_label=gt_label, args=args)

        title = f"Leakage Showcase #{int(sample['rank'])}"
        subtitle = (
            f"Area {int(sample['area_id'])} | grid ({int(sample['grid_x'])}, {int(sample['grid_y'])}) | "
            f"pred {float(sample['dense_pred_ratio']) * 100.0:.2f}% | gt {float(sample['dense_gt_ratio']) * 100.0:.2f}%"
        )

        figure = compose_showcase(
            overview_image=overview_image,
            unwrap_image=unwrap_image,
            title=title,
            subtitle=subtitle,
            canvas_width=int(args.canvas_width),
            canvas_height=int(args.canvas_height),
            left_width=int(args.left_width),
            margin=int(args.page_margin),
            panel_gap=int(args.panel_gap),
            title_font=fonts["title"],
            subtitle_font=fonts["subtitle"],
            caption_font=fonts["caption"],
        )
        name = f"showcase_rank_{int(sample['rank']):02d}_area_{int(sample['area_id']):02d}_gx_{int(sample['grid_x']):03d}_gy_{int(sample['grid_y']):02d}.png"
        path = output_dir / name
        figure.save(path)
        saved.append(
            {
                "rank": int(sample["rank"]),
                "area_id": int(sample["area_id"]),
                "grid_x": int(sample["grid_x"]),
                "grid_y": int(sample["grid_y"]),
                "image": str(path),
                "overview_image": "generated_from_dense_point_cloud",
            }
        )
        print(f"saved {path}", flush=True)

    summary_out = {
        "dense_summary_path": str(dense_summary_path),
        "overview_root": str(overview_root),
        "output_dir": str(output_dir),
        "saved_count": len(saved),
        "figures": saved,
    }
    summary_path = output_dir / "ppt_showcase_summary.json"
    summary_path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
