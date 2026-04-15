from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FIGURE_BG = (248, 246, 242)
PANEL_BG = (255, 255, 255)
TEXT_MAIN = (32, 35, 39)
TEXT_MUTED = (104, 111, 118)
TEXT_ACCENT = (18, 58, 96)
BLUE = (18, 86, 255)
DIVIDER = (165, 168, 172)
BORDER = (220, 223, 228)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a paper-style figure from dense 3D and unwrapped 2D outputs.")
    parser.add_argument("--dense_summary_path", type=Path, required=True)
    parser.add_argument("--projection_summary_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--area_source_base_dir", type=Path, default=None)
    parser.add_argument("--projection", type=str, default="x_theta")
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None)
    parser.add_argument("--column_width", type=int, default=1260)
    parser.add_argument("--global_overview_height", type=int, default=320)
    parser.add_argument("--local_overview_height", type=int, default=320)
    parser.add_argument("--projection_height", type=int, default=250)
    parser.add_argument("--page_margin", type=int, default=44)
    parser.add_argument("--column_gap", type=int, default=28)
    parser.add_argument("--section_gap", type=int, default=22)
    parser.add_argument("--tile_gap", type=int, default=16)
    parser.add_argument("--overview_width", type=int, default=1200)
    parser.add_argument("--overview_render_height", type=int, default=720)
    parser.add_argument("--max_background_points", type=int, default=32000)
    parser.add_argument("--max_positive_points", type=int, default=32000)
    parser.add_argument("--bg_point_size", type=float, default=1.1)
    parser.add_argument("--fg_point_size", type=float, default=2.7)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-64.0)
    parser.add_argument("--global_background_points", type=int, default=100000)
    parser.add_argument("--global_leakage_points", type=int, default=12000)
    parser.add_argument("--global_positive_points", type=int, default=48000)
    parser.add_argument("--global_bg_point_size", type=float, default=0.85)
    parser.add_argument("--global_fg_point_size", type=float, default=2.8)
    parser.add_argument("--global_elev", type=float, default=18.0)
    parser.add_argument("--global_azim", type=float, default=-74.0)
    parser.add_argument("--seed", type=int, default=20260329)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return args.projection_summary_path.resolve().parent / "paper_style_figure"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_contain(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(target_width / image.width, target_height / image.height)
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), PANEL_BG)
    offset = ((target_width - resized.width) // 2, (target_height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def crop_content(image: Image.Image, threshold: int = 246, padding: int = 12) -> Image.Image:
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    content_mask = np.any(array < threshold, axis=2)
    if not np.any(content_mask):
        return rgb
    ys, xs = np.where(content_mask)
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, rgb.width)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, rgb.height)
    return rgb.crop((x0, y0, x1, y1))


def add_panel_border(image: Image.Image, border_width: int = 1) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for offset in range(border_width):
        draw.rectangle(
            [offset, offset, out.width - 1 - offset, out.height - 1 - offset],
            outline=BORDER,
            width=1,
        )
    return out


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return np.zeros_like(intensity, dtype=np.float32)
    finite = np.isfinite(intensity)
    if not np.any(finite):
        return np.zeros_like(intensity, dtype=np.float32)
    valid = intensity[finite].astype(np.float32)
    low = float(np.percentile(valid, 1.0))
    high = float(np.percentile(valid, 99.0))
    if high <= low + 1e-6:
        norm = np.zeros_like(intensity, dtype=np.float32)
        norm[finite] = 0.5
        return norm
    clipped = np.clip(intensity.astype(np.float32), low, high)
    return (clipped - low) / (high - low)


def sample_indices(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if limit <= 0 or len(idx) <= limit:
        return idx
    return np.sort(rng.choice(idx, size=limit, replace=False))


def reservoir_sample_txt_points(path: Path, sample_size: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    if sample_size <= 0:
        return np.empty((0, 4), dtype=np.float32), 0
    sample = np.empty((sample_size, 4), dtype=np.float32)
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            row = np.fromstring(stripped, sep=" ", dtype=np.float32)
            if row.size < 4:
                continue
            point = row[:4]
            if count < sample_size:
                sample[count] = point
            else:
                replace_idx = int(rng.integers(0, count + 1))
                if replace_idx < sample_size:
                    sample[replace_idx] = point
            count += 1
    if count == 0:
        return np.empty((0, 4), dtype=np.float32), 0
    return sample[: min(count, sample_size)].copy(), count


def make_bbox_edges(min_xyz: np.ndarray, max_xyz: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
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
    edge_ids = [
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
    return [(corners[i], corners[j]) for i, j in edge_ids]


def load_area_overview_points(
    source_base_dir: Path,
    area_id: int,
    background_points: int,
    leakage_points: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | int | str]:
    area_root = source_base_dir / f"Area_{area_id}_down_sampled" / f"Area_{area_id}_down_sampled"
    bg_path = area_root / "background.txt"
    leak_path = area_root / "leakage.txt"
    if (not bg_path.exists()) or (not leak_path.exists()):
        raise FileNotFoundError(f"Missing area source files: {bg_path} {leak_path}")

    bg_sample, bg_count = reservoir_sample_txt_points(bg_path, background_points, rng)
    leak_sample, leak_count = reservoir_sample_txt_points(leak_path, leakage_points, rng)
    if len(bg_sample) == 0 and len(leak_sample) == 0:
        raise RuntimeError(f"Area {area_id} produced no overview sample points.")
    if len(bg_sample) == 0:
        points = leak_sample.astype(np.float32, copy=False)
    elif len(leak_sample) == 0:
        points = bg_sample.astype(np.float32, copy=False)
    else:
        points = np.vstack([bg_sample, leak_sample]).astype(np.float32, copy=False)
    return {
        "points": points,
        "background_count_total": int(bg_count),
        "leakage_count_total": int(leak_count),
        "area_root": str(area_root),
    }


def render_overview_layers(
    base_point_cloud: np.ndarray,
    overlay_points: np.ndarray | None,
    output_size: tuple[int, int],
    max_background_points: int,
    max_overlay_points: int,
    bg_point_size: float,
    overlay_point_size: float,
    elev: float,
    azim: float,
    rng: np.random.Generator,
    outline_bbox: tuple[np.ndarray, np.ndarray] | None = None,
    center_marker: np.ndarray | None = None,
) -> Image.Image:
    base_point_cloud = np.asarray(base_point_cloud, dtype=np.float32)
    xyz = np.asarray(base_point_cloud[:, :3], dtype=np.float32)
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    intensity = base_point_cloud[:, 3].astype(np.float32) if base_point_cloud.shape[1] > 3 else np.zeros((len(base_point_cloud),), dtype=np.float32)
    intensity_norm = normalize_intensity(intensity)

    bg_mask = np.ones((len(base_point_cloud),), dtype=bool)
    bg_idx = sample_indices(bg_mask, max_background_points, rng)
    overlay_array = np.empty((0, 3), dtype=np.float32) if overlay_points is None else np.asarray(overlay_points[:, :3], dtype=np.float32)
    if len(overlay_array) > max_overlay_points > 0:
        overlay_pick = np.sort(rng.choice(len(overlay_array), size=max_overlay_points, replace=False))
        overlay_array = overlay_array[overlay_pick]
    overlay_centered = overlay_array - center if len(overlay_array) > 0 else overlay_array

    bg_colors = 0.34 + 0.48 * intensity_norm[bg_idx]
    bg_rgb = np.stack([bg_colors, bg_colors, bg_colors], axis=1)
    fg_rgb = np.tile(np.asarray(BLUE, dtype=np.float32)[None, :] / 255.0, (len(overlay_centered), 1))

    fig = plt.figure(figsize=(output_size[0] / 150.0, output_size[1] / 150.0), dpi=150, facecolor=np.asarray(PANEL_BG) / 255.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(np.asarray(PANEL_BG) / 255.0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(tuple(np.maximum(centered.max(axis=0) - centered.min(axis=0), 1e-3).tolist()))

    if len(bg_idx) > 0:
        bg_points = centered[bg_idx]
        ax.scatter(
            bg_points[:, 0],
            bg_points[:, 1],
            bg_points[:, 2],
            c=bg_rgb,
            s=bg_point_size,
            alpha=0.82,
            linewidths=0.0,
            depthshade=False,
        )
    if len(overlay_centered) > 0:
        ax.scatter(
            overlay_centered[:, 0],
            overlay_centered[:, 1],
            overlay_centered[:, 2],
            c=fg_rgb,
            s=overlay_point_size,
            alpha=0.96,
            linewidths=0.0,
            depthshade=False,
        )
    if outline_bbox is not None:
        min_xyz, max_xyz = outline_bbox
        for edge_start, edge_end in make_bbox_edges(np.asarray(min_xyz, dtype=np.float32), np.asarray(max_xyz, dtype=np.float32)):
            start = edge_start - center[0]
            end = edge_end - center[0]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=np.asarray(BLUE, dtype=np.float32) / 255.0,
                linewidth=2.3,
                alpha=0.98,
            )
    if center_marker is not None:
        marker = np.asarray(center_marker, dtype=np.float32).reshape(1, 3) - center
        ax.scatter(
            marker[:, 0],
            marker[:, 1],
            marker[:, 2],
            c=np.asarray(BLUE, dtype=np.float32)[None, :] / 255.0,
            s=54.0,
            alpha=0.98,
            linewidths=0.0,
            depthshade=False,
        )

    mins = centered.min(axis=0)
    maxs = centered.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return crop_content(Image.open(buffer).convert("RGB"), threshold=247, padding=24)


def render_overview_image(
    point_cloud: np.ndarray,
    pred_label: np.ndarray,
    output_size: tuple[int, int],
    max_background_points: int,
    max_positive_points: int,
    bg_point_size: float,
    fg_point_size: float,
    elev: float,
    azim: float,
    rng: np.random.Generator,
) -> Image.Image:
    positive_points = np.asarray(point_cloud[np.asarray(pred_label, dtype=np.uint8) > 0], dtype=np.float32)
    return render_overview_layers(
        base_point_cloud=point_cloud,
        overlay_points=positive_points,
        output_size=output_size,
        max_background_points=max_background_points,
        max_overlay_points=max_positive_points,
        bg_point_size=bg_point_size,
        overlay_point_size=fg_point_size,
        elev=elev,
        azim=azim,
        rng=rng,
    )


def draw_text(draw: ImageDraw.ImageDraw, position: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> tuple[int, int]:
    draw.text(position, text, font=font, fill=fill)
    bbox = draw.textbbox(position, text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def build_label_strip(width: int, title: str, subtitle: str, label: str, title_font: ImageFont.ImageFont, subtitle_font: ImageFont.ImageFont) -> Image.Image:
    height = 106
    canvas = Image.new("RGB", (width, height), PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (24, 16), title, title_font, TEXT_MAIN)
    draw_text(draw, (24, 58), subtitle, subtitle_font, TEXT_MUTED)

    label_bbox = draw.textbbox((0, 0), label, font=subtitle_font)
    pill_width = label_bbox[2] - label_bbox[0] + 28
    pill_height = label_bbox[3] - label_bbox[1] + 14
    pill_x = width - pill_width - 22
    pill_y = 22
    draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_width, pill_y + pill_height], radius=12, fill=(231, 239, 248))
    draw_text(draw, (pill_x + 14, pill_y + 7), label, subtitle_font, TEXT_ACCENT)
    return add_panel_border(canvas)


def build_section_header(width: int, text: str, font: ImageFont.ImageFont) -> Image.Image:
    height = 46
    canvas = Image.new("RGB", (width, height), FIGURE_BG)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (0, 6), text, font, TEXT_MAIN)
    return canvas


def stack_vertical(images: list[Image.Image], gap: int, background: tuple[int, int, int]) -> Image.Image:
    width = max(img.width for img in images)
    height = sum(img.height for img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    y = 0
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    return canvas


def stack_horizontal(images: list[Image.Image], gap: int, background: tuple[int, int, int]) -> Image.Image:
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for image in images:
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        x += image.width + gap
    return canvas


def add_caption(image: Image.Image, caption: str, caption_font: ImageFont.ImageFont) -> Image.Image:
    caption_height = 44
    canvas = Image.new("RGB", (image.width, image.height + caption_height), PANEL_BG)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (18, image.height + 8), caption, caption_font, TEXT_MUTED)
    return add_panel_border(canvas)


def compose_column(
    summary_sample: dict,
    projection_sample: dict,
    projection: str,
    area_overview_cache: dict[int, dict[str, np.ndarray | int | str]],
    args: argparse.Namespace,
    rng: np.random.Generator,
    fonts: dict[str, ImageFont.ImageFont],
    output_dir: Path,
) -> tuple[Image.Image, dict]:
    dense_npz = np.load(summary_sample["files"]["dense_npz"])
    point_cloud = dense_npz["point_cloud"]
    pred_label = dense_npz["pred_label"]

    local_overview_raw = render_overview_image(
        point_cloud=point_cloud,
        pred_label=pred_label,
        output_size=(int(args.overview_width), int(args.overview_render_height)),
        max_background_points=int(args.max_background_points),
        max_positive_points=int(args.max_positive_points),
        bg_point_size=float(args.bg_point_size),
        fg_point_size=float(args.fg_point_size),
        elev=float(args.elev),
        azim=float(args.azim),
        rng=rng,
    )

    block_tag = f"area_{int(summary_sample['area_id']):02d}_gx_{int(summary_sample['grid_x']):03d}_gy_{int(summary_sample['grid_y']):02d}"
    sample_dir = output_dir / block_tag
    sample_dir.mkdir(parents=True, exist_ok=True)
    local_overview_path = sample_dir / "overview_block_3d.png"
    local_overview_raw.save(local_overview_path)

    area_id = int(summary_sample["area_id"])
    area_cache = area_overview_cache[area_id]
    block_min_xyz = np.asarray(point_cloud[:, :3].min(axis=0), dtype=np.float32)
    block_max_xyz = np.asarray(point_cloud[:, :3].max(axis=0), dtype=np.float32)
    block_center_xyz = (block_min_xyz + block_max_xyz) * 0.5
    area_overview_raw = render_overview_layers(
        base_point_cloud=np.asarray(area_cache["points"], dtype=np.float32),
        overlay_points=np.asarray(point_cloud, dtype=np.float32),
        output_size=(int(args.overview_width), int(args.overview_render_height)),
        max_background_points=int(args.global_background_points) + int(args.global_leakage_points),
        max_overlay_points=int(args.global_positive_points),
        bg_point_size=float(args.global_bg_point_size),
        overlay_point_size=float(args.global_fg_point_size),
        elev=float(args.global_elev),
        azim=float(args.global_azim),
        rng=rng,
        outline_bbox=(block_min_xyz, block_max_xyz),
        center_marker=block_center_xyz,
    )
    area_overview_path = sample_dir / "overview_area_3d.png"
    area_overview_raw.save(area_overview_path)

    area_panel = fit_contain(crop_content(area_overview_raw, threshold=247, padding=20), int(args.column_width), int(args.global_overview_height))
    area_panel = add_caption(area_panel, f"Area {area_id} overview | blue = current block footprint", fonts["caption"])
    local_panel = fit_contain(crop_content(local_overview_raw, threshold=247, padding=20), int(args.column_width), int(args.local_overview_height))
    local_panel = add_caption(local_panel, "Block 3D patch | fixed camera template", fonts["caption"])
    context_panel = stack_vertical([area_panel, local_panel], gap=int(args.tile_gap), background=FIGURE_BG)

    projection_meta = projection_sample["projections"][projection]
    projection_tiles = [
        ("Prediction", Image.open(projection_meta["prediction_image"]).convert("RGB")),
        ("Ground Truth", Image.open(projection_meta["ground_truth_image"]).convert("RGB")),
        ("Compare", Image.open(projection_meta["compare_image"]).convert("RGB")),
    ]

    tile_panels: list[Image.Image] = []
    for caption, image in projection_tiles:
        cropped = crop_content(image, threshold=247, padding=18)
        fitted = fit_contain(cropped, int(args.column_width), int(args.projection_height))
        tile_panels.append(add_caption(fitted, f"{projection} | {caption}", fonts["caption"]))

    title = f"Rank {int(summary_sample['rank'])}"
    subtitle = (
        f"Area {int(summary_sample['area_id'])} | grid ({int(summary_sample['grid_x'])}, {int(summary_sample['grid_y'])}) | "
        f"pred {float(summary_sample['dense_pred_ratio']) * 100.0:.2f}% | gt {float(summary_sample['dense_gt_ratio']) * 100.0:.2f}%"
    )
    label = f"{int(summary_sample['raw_point_count'])} pts"

    pieces = [
        build_label_strip(int(args.column_width), title, subtitle, label, fonts["title"], fonts["subtitle"]),
        build_section_header(int(args.column_width), "Top View: Area Context + Standardized Local 3D", fonts["section"]),
        context_panel,
        build_section_header(int(args.column_width), f"Bottom View: Surface Unwrap ({projection})", fonts["section"]),
        *tile_panels,
    ]
    column = stack_vertical(pieces, gap=int(args.section_gap), background=FIGURE_BG)
    return column, {
        "rank": int(summary_sample["rank"]),
        "area_id": int(summary_sample["area_id"]),
        "grid_x": int(summary_sample["grid_x"]),
        "grid_y": int(summary_sample["grid_y"]),
        "area_overview_image": str(area_overview_path),
        "local_overview_image": str(local_overview_path),
        "projection_images": {
            "prediction": str(projection_meta["prediction_image"]),
            "ground_truth": str(projection_meta["ground_truth_image"]),
            "compare": str(projection_meta["compare_image"]),
        },
    }


def draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int, dash: int, gap: int, color: tuple[int, int, int], width: int) -> None:
    y = y0
    while y < y1:
        y_end = min(y + dash, y1)
        draw.line([(x, y), (x, y_end)], fill=color, width=width)
        y = y_end + gap


def compose_page(columns: list[Image.Image], args: argparse.Namespace, fonts: dict[str, ImageFont.ImageFont]) -> Image.Image:
    title_height = 136
    margin = int(args.page_margin)
    column_gap = int(args.column_gap)
    width = margin * 2 + sum(col.width for col in columns) + column_gap * max(len(columns) - 1, 0)
    height = title_height + margin + max(col.height for col in columns) + margin
    canvas = Image.new("RGB", (width, height), FIGURE_BG)
    draw = ImageDraw.Draw(canvas)

    draw_text(draw, (margin, 22), "Dense Leakage Segmentation Results", fonts["page_title"], TEXT_MAIN)
    draw_text(
        draw,
        (margin, 74),
        "Top rows show area-level context and a standardized local 3D patch. Bottom rows show x-theta surface unwrap with flat 2D boxes.",
        fonts["page_subtitle"],
        TEXT_MUTED,
    )
    draw_text(
        draw,
        (width - 480, 74),
        "Blue = leakage | Gray = surface | Dashed box = 2D component bbox",
        fonts["page_subtitle"],
        TEXT_ACCENT,
    )
    draw.line([(margin, title_height - 4), (width - margin, title_height - 4)], fill=BORDER, width=2)

    x = margin
    y = title_height + 18
    for idx, column in enumerate(columns):
        canvas.paste(column, (x, y))
        x += column.width
        if idx < len(columns) - 1:
            divider_x = x + column_gap // 2
            draw_dashed_vertical(draw, divider_x, y, y + max(col.height for col in columns), dash=14, gap=10, color=DIVIDER, width=2)
            x += column_gap
    return canvas


def select_samples(dense_samples: list[dict], projection_samples: list[dict], sample_indices: list[int] | None) -> list[tuple[dict, dict]]:
    projection_map = {
        (
            int(sample["rank"]),
            int(sample["index"]),
            int(sample["area_id"]),
            int(sample["grid_x"]),
            int(sample["grid_y"]),
        ): sample
        for sample in projection_samples
    }
    paired: list[tuple[dict, dict]] = []
    if sample_indices is None:
        target_dense_samples = dense_samples
    else:
        target_dense_samples = [dense_samples[int(i)] for i in sample_indices]
    for sample in target_dense_samples:
        key = (
            int(sample["rank"]),
            int(sample["index"]),
            int(sample["area_id"]),
            int(sample["grid_x"]),
            int(sample["grid_y"]),
        )
        projection_sample = projection_map.get(key)
        if projection_sample is None:
            raise KeyError(f"Projection summary missing sample for key={key}")
        paired.append((sample, projection_sample))
    return paired


def resolve_area_source_base_dir(args: argparse.Namespace, dense_summary: dict) -> Path:
    if args.area_source_base_dir is not None:
        return args.area_source_base_dir.resolve()
    summary_base = dense_summary.get("source_base_dir")
    if summary_base is not None:
        return Path(summary_base).resolve()
    dense_summary_path = Path(args.dense_summary_path).resolve()
    return dense_summary_path.parents[4]


def main() -> None:
    args = parse_args()
    dense_summary_path = args.dense_summary_path.resolve()
    projection_summary_path = args.projection_summary_path.resolve()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_summary = load_json(dense_summary_path)
    projection_summary = load_json(projection_summary_path)
    dense_samples = dense_summary.get("samples", [])
    projection_samples = projection_summary.get("samples", [])
    if not dense_samples:
        raise RuntimeError("Dense summary contains no samples.")
    if not projection_samples:
        raise RuntimeError("Projection summary contains no samples.")

    paired_samples = select_samples(dense_samples, projection_samples, args.sample_indices)
    if not paired_samples:
        raise RuntimeError("No paired samples were selected.")

    area_source_base_dir = resolve_area_source_base_dir(args, dense_summary)
    selected_area_ids = sorted({int(summary_sample["area_id"]) for summary_sample, _ in paired_samples})

    fonts = {
        "title": load_font(30, bold=True),
        "subtitle": load_font(18, bold=False),
        "section": load_font(22, bold=True),
        "caption": load_font(18, bold=False),
        "page_title": load_font(40, bold=True),
        "page_subtitle": load_font(20, bold=False),
    }

    rng = np.random.default_rng(int(args.seed))
    area_overview_cache: dict[int, dict[str, np.ndarray | int | str]] = {}
    for area_id in selected_area_ids:
        area_overview_cache[area_id] = load_area_overview_points(
            source_base_dir=area_source_base_dir,
            area_id=area_id,
            background_points=int(args.global_background_points),
            leakage_points=int(args.global_leakage_points),
            rng=rng,
        )
        print(
            f"loaded area overview sample | area={area_id} sampled_points={len(np.asarray(area_overview_cache[area_id]['points']))}",
            flush=True,
        )

    columns: list[Image.Image] = []
    metadata_records: list[dict] = []
    for summary_sample, projection_sample in paired_samples:
        column, metadata = compose_column(
            summary_sample=summary_sample,
            projection_sample=projection_sample,
            projection=args.projection,
            area_overview_cache=area_overview_cache,
            args=args,
            rng=rng,
            fonts=fonts,
            output_dir=output_dir,
        )
        columns.append(column)
        metadata_records.append(metadata)
        print(
            f"composed column | rank={metadata['rank']} area={metadata['area_id']} "
            f"grid=({metadata['grid_x']}, {metadata['grid_y']})",
            flush=True,
        )

    page = compose_page(columns, args=args, fonts=fonts)
    figure_path = output_dir / f"paper_style_figure_{args.projection}.png"
    page.save(figure_path)

    metadata = {
        "dense_summary_path": str(dense_summary_path),
        "projection_summary_path": str(projection_summary_path),
        "area_source_base_dir": str(area_source_base_dir),
        "projection": args.projection,
        "sample_count": len(metadata_records),
        "output_figure": str(figure_path),
        "columns": metadata_records,
    }
    metadata_path = output_dir / "paper_style_figure_summary.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved {figure_path}", flush=True)
    print(f"saved {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
