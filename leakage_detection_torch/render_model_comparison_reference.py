from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from leakage_detection_torch.dense_block_inference import (
    fill_unsampled_probabilities,
    load_dataset_attrs,
    load_area_source_blocks,
    extract_source_block,
)
from leakage_detection_torch.export_predicted_pointclouds import (
    choose_device,
    load_model as load_ours_model,
    load_run_args,
    normalize_point_cloud,
)
from leakage_detection_torch.render_paper_figure import crop_content, fit_contain, load_font
from leakage_detection_torch.render_ppt_texture_showcase import (
    add_border,
    render_dense_local_overview,
    render_texture_unwrap,
)


FIG_BG = (248, 246, 242)
PANEL_BG = (255, 255, 255)
TEXT = (32, 35, 40)
MUTED = (102, 108, 116)
BORDER = (222, 225, 230)


@dataclass
class ModelSpec:
    name: str
    kind: str
    checkpoint: Path | None = None
    args_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render reference-style comparison figures across segmentation models.")
    parser.add_argument(
        "--dense_summary_path",
        type=Path,
        default=Path("/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260327_223510/dense_pointcloud_visualizations_texture_attempt/dense_pointcloud_inference_summary.json"),
    )
    parser.add_argument(
        "--ours_checkpoint",
        type=Path,
        default=Path("/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260327_223510/best_model.pt"),
    )
    parser.add_argument(
        "--dgcnn_checkpoint",
        type=Path,
        default=Path("/ai/0309/cloud/compare/DGCNN/dgcnn.pytorch/outputs_leakage_v2/20260327_112315/best_model.pth"),
    )
    parser.add_argument(
        "--dgcnn_args_path",
        type=Path,
        default=Path("/ai/0309/cloud/compare/DGCNN/dgcnn.pytorch/outputs_leakage_v2/20260327_112315/args.json"),
    )
    parser.add_argument(
        "--randla_checkpoint",
        type=Path,
        default=Path("/ai/0309/cloud/compare/RandLA-Net/RandLA-Net-pytorch/outputs_leakage_v2/20260327_153227/best_model.pth"),
    )
    parser.add_argument(
        "--randla_args_path",
        type=Path,
        default=Path("/ai/0309/cloud/compare/RandLA-Net/RandLA-Net-pytorch/outputs_leakage_v2/20260327_153227/args.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260327_223510/model_comparison_reference"),
    )
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None, help="Indices within dense summary samples.")
    parser.add_argument("--num_votes", type=int, default=32)
    parser.add_argument("--vote_batch_size", type=int, default=8)
    parser.add_argument("--prob_threshold", type=float, default=-1.0)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--canvas_width", type=int, default=2440)
    parser.add_argument("--header_height", type=int, default=86)
    parser.add_argument("--cell_gap", type=int, default=18)
    parser.add_argument("--outer_margin", type=int, default=24)
    parser.add_argument("--sample_gap", type=int, default=28)
    parser.add_argument("--model_order", type=str, nargs="+", default=["DGCNN", "RandLA-Net", "Ours", "Ground Truth"])
    parser.add_argument("--render_tile_width", type=int, default=2400)
    parser.add_argument("--render_tile_height", type=int, default=1280)
    parser.add_argument("--pred_focus_threshold", type=float, default=0.18)
    parser.add_argument("--pred_min_component_pixels", type=int, default=1800)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dgcnn_model(args_path: Path, checkpoint: Path, device: torch.device):
    spec = load_json(args_path)
    model_path = (ROOT_DIR / "compare" / "DGCNN" / "dgcnn.pytorch" / "model.py").resolve()
    module_spec = importlib.util.spec_from_file_location("dgcnn_model_refcmp", str(model_path))
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Failed to load DGCNN model module: {model_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    DGCNN_semseg_leakage = module.DGCNN_semseg_leakage

    model = DGCNN_semseg_leakage(
        num_classes=int(spec.get("classes", 2)),
        in_channels=int(spec.get("in_channels", 4)),
        k=int(spec.get("k", 20)),
        emb_dims=int(spec.get("emb_dims", 1024)),
        dropout=float(spec.get("dropout", 0.5)),
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, spec


def load_randla_model(args_path: Path, checkpoint: Path, device: torch.device):
    spec = load_json(args_path)
    model_path = (ROOT_DIR / "compare" / "RandLA-Net" / "RandLA-Net-pytorch" / "model.py").resolve()
    module_spec = importlib.util.spec_from_file_location("randla_model_refcmp", str(model_path))
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Failed to load RandLA model module: {model_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    RandLANet = module.RandLANet

    model = RandLANet(
        d_in=int(spec.get("in_channels", 4)),
        num_classes=int(spec.get("classes", 2)),
        num_neighbors=int(spec.get("neighbors", 16)),
        decimation=int(spec.get("decimation", 4)),
        device=device,
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, spec


def predict_batch_ours(model, batch_np: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        batch_tensor = torch.from_numpy(batch_np.transpose(0, 2, 1)).to(device)
        seg_output, _ = model(batch_tensor, return_intermediate=False)
        return seg_output[:, 1, :].detach().cpu().numpy().astype(np.float32)


def predict_batch_dgcnn(model, batch_np: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        batch_tensor = torch.from_numpy(batch_np.transpose(0, 2, 1)).to(device)
        logits, _ = model(batch_tensor)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]
        return probs.detach().cpu().numpy().astype(np.float32)


def predict_batch_randla(model, batch_np: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        batch_tensor = torch.from_numpy(batch_np).to(device)
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1, :]
        return probs.detach().cpu().numpy().astype(np.float32)


def infer_dense_probabilities_generic(
    point_cloud: np.ndarray,
    model,
    predict_batch_fn,
    device: torch.device,
    in_channels: int,
    sample_points: int,
    num_votes: int,
    vote_batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dense_points = np.asarray(point_cloud[:, :in_channels], dtype=np.float32)
    num_dense_points = int(len(dense_points))
    prob_sum = np.zeros((num_dense_points,), dtype=np.float32)
    vote_count = np.zeros((num_dense_points,), dtype=np.int32)

    for vote_start in range(0, int(num_votes), int(vote_batch_size)):
        vote_end = min(vote_start + int(vote_batch_size), int(num_votes))
        sampled_batches = []
        sampled_indices = []
        for _ in range(vote_start, vote_end):
            replace = num_dense_points < int(sample_points)
            choice = rng.choice(num_dense_points, size=int(sample_points), replace=replace)
            sampled_indices.append(choice.astype(np.int64, copy=False))
            sampled = normalize_point_cloud(dense_points[choice].copy()).astype(np.float32, copy=False)
            sampled_batches.append(sampled)
        batch_np = np.stack(sampled_batches, axis=0)
        leak_prob = predict_batch_fn(model, batch_np, device)
        for block_indices, block_prob in zip(sampled_indices, leak_prob):
            np.add.at(prob_sum, block_indices, block_prob)
            np.add.at(vote_count, block_indices, 1)

    mean_prob = np.zeros((num_dense_points,), dtype=np.float32)
    known_mask = vote_count > 0
    if np.any(known_mask):
        mean_prob[known_mask] = prob_sum[known_mask] / vote_count[known_mask].astype(np.float32)
    return mean_prob, vote_count


def render_model_column(
    point_cloud: np.ndarray,
    pred_prob: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Image.Image, Image.Image]:
    render_args = argparse.Namespace(
        prob_threshold=float(args.prob_threshold),
        overview_width=1850,
        overview_height=2250,
        overview_bg_point_size=0.95,
        overview_fg_point_size=2.8,
        overview_elev=18.0,
        overview_azim=-64.0,
        tile_width=int(args.render_tile_width),
        tile_height=int(args.render_tile_height),
        smooth_sigma=1.25,
        coarse_sigma=3.4,
        detail_sigma=7.0,
        support_sigma=2.2,
        detail_amount=0.44,
        contrast_gamma=0.88,
        gray_floor=0.76,
        gray_gain=0.20,
        support_threshold=0.0005,
        surface_close_radius=2,
        surface_dilate_radius=2,
        pred_close_radius=3,
        pred_dilate_radius=2,
        pred_sigma=0.85,
        pred_focus_threshold=float(args.pred_focus_threshold),
        pred_min_strength=0.80,
        pred_min_component_pixels=int(args.pred_min_component_pixels),
        pred_alpha_min=0.48,
        pred_alpha_max=0.98,
        edge_margin_pixels=12.0,
        unsharp_radius=1.6,
        unsharp_percent=138,
        unsharp_threshold=2,
    )
    overview = render_dense_local_overview(point_cloud=point_cloud, pred_prob=pred_prob, args=render_args)
    unwrap = render_texture_unwrap(point_cloud=point_cloud, pred_prob=pred_prob, args=render_args)
    return overview, unwrap


def crop_unwrap_main_band(
    image: Image.Image,
    threshold: int = 250,
    padding: int = 12,
    min_row_pixels: int | None = None,
) -> Image.Image:
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    content_mask = np.any(array < threshold, axis=2)
    if not np.any(content_mask):
        return rgb

    row_counts = content_mask.sum(axis=1)
    if min_row_pixels is None:
        min_row_pixels = max(24, int(round(rgb.width * 0.02)))
    active_rows = row_counts >= int(min_row_pixels)
    if not np.any(active_rows):
        return crop_content(rgb, threshold=threshold, padding=padding)

    segments: list[tuple[int, int, int]] = []
    start = None
    for idx, flag in enumerate(active_rows):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx - 1
            score = int(row_counts[start : end + 1].sum())
            segments.append((start, end, score))
            start = None
    if start is not None:
        end = len(active_rows) - 1
        score = int(row_counts[start : end + 1].sum())
        segments.append((start, end, score))

    if not segments:
        return crop_content(rgb, threshold=threshold, padding=padding)

    best_start, best_end, _ = max(segments, key=lambda item: item[2])
    band_mask = np.zeros_like(content_mask, dtype=bool)
    band_mask[best_start : best_end + 1, :] = content_mask[best_start : best_end + 1, :]
    if not np.any(band_mask):
        return crop_content(rgb, threshold=threshold, padding=padding)

    ys, xs = np.where(band_mask)
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, rgb.width)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, rgb.height)
    return rgb.crop((x0, y0, x1, y1))


def fit_panel(image: Image.Image, width: int, height: int, threshold: int = 247) -> Image.Image:
    return add_border(fit_contain(crop_content(image, threshold=threshold, padding=12), width, height), width=1)


def fit_unwrap_panel(image: Image.Image, width: int, height: int, threshold: int = 250) -> Image.Image:
    cropped = crop_unwrap_main_band(image, threshold=threshold, padding=12)
    return add_border(fit_contain(cropped, width, height), width=1)


def compose_sample_figure(
    sample_title: str,
    columns: list[tuple[str, Image.Image, Image.Image]],
    args: argparse.Namespace,
    title_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> Image.Image:
    outer_margin = int(args.outer_margin)
    cell_gap = int(args.cell_gap)
    header_height = int(args.header_height)
    canvas_width = int(args.canvas_width)
    n_cols = len(columns)
    col_width = (canvas_width - outer_margin * 2 - cell_gap * (n_cols - 1)) // n_cols
    top_panel_h = int(round(col_width * 0.82))
    bottom_panel_h = int(round(col_width * 0.56))
    bottom_gap = 18
    body_height = top_panel_h + bottom_gap + bottom_panel_h + 40
    canvas_height = outer_margin + header_height + body_height + outer_margin

    canvas = Image.new("RGB", (canvas_width, canvas_height), FIG_BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((outer_margin, outer_margin - 2), sample_title, font=title_font, fill=TEXT)
    draw.line(
        (
            outer_margin,
            outer_margin + header_height - 8,
            canvas_width - outer_margin,
            outer_margin + header_height - 8,
        ),
        fill=BORDER,
        width=2,
    )

    y_top = outer_margin + header_height
    for col_idx, (label, overview_img, unwrap_img) in enumerate(columns):
        x = outer_margin + col_idx * (col_width + cell_gap)
        draw.text((x + 8, y_top - 32), label, font=label_font, fill=MUTED)
        top_panel = fit_panel(overview_img, col_width, top_panel_h, threshold=247)
        bottom_panel = fit_unwrap_panel(unwrap_img, col_width, bottom_panel_h, threshold=250)
        canvas.paste(top_panel, (x, y_top))
        canvas.paste(bottom_panel, (x, y_top + top_panel_h + bottom_gap))
    return canvas


def concat_vertical(images: list[Image.Image], gap: int, background: tuple[int, int, int]) -> Image.Image:
    width = max(img.width for img in images)
    height = sum(img.height for img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    y = 0
    for img in images:
        x = (width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + gap
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.resolve())
    cache_dir = ensure_dir(output_dir / "cache")
    sample_fig_dir = ensure_dir(output_dir / "sample_figures")
    summary = load_json(args.dense_summary_path.resolve())

    prob_threshold = float(summary.get("prob_threshold", 0.5)) if float(args.prob_threshold) < 0 else float(args.prob_threshold)
    args.prob_threshold = prob_threshold
    samples = summary.get("samples", [])
    if not samples:
        raise RuntimeError("No samples found in dense summary.")
    selected = list(range(min(3, len(samples)))) if args.sample_indices is None else [int(v) for v in args.sample_indices]

    dataset_path = Path(summary["dataset_path"]).resolve()
    dataset_attrs = load_dataset_attrs(dataset_path)
    sample_points = int(dataset_attrs.get("num_points", 4096))
    cell_size_x = float(dataset_attrs.get("cell_size_x", 2.0))
    cell_size_y = float(dataset_attrs.get("cell_size_y", 2.0))
    source_base_dir = Path(summary["source_base_dir"]).resolve()

    device = choose_device(args.device)
    rng_seed = int(args.seed)

    ours_run_args = load_run_args(args.ours_checkpoint.with_name("args.json"))
    ours_model = load_ours_model(ours_run_args, args.ours_checkpoint.resolve(), device)
    ours_in_channels = int(ours_run_args.get("in_channels", 4))

    dgcnn_model, dgcnn_spec = load_dgcnn_model(args.dgcnn_args_path.resolve(), args.dgcnn_checkpoint.resolve(), device)
    randla_model, randla_spec = load_randla_model(args.randla_args_path.resolve(), args.randla_checkpoint.resolve(), device)

    area_cache: dict[int, dict[str, np.ndarray | float]] = {}
    title_font = load_font(28, bold=True)
    label_font = load_font(24, bold=True)

    sample_figures: list[Image.Image] = []
    saved_samples = []
    model_lookup = {
        "Ours": ModelSpec("Ours", "ours"),
        "DGCNN": ModelSpec("DGCNN", "dgcnn"),
        "RandLA-Net": ModelSpec("RandLA-Net", "randla"),
        "Ground Truth": ModelSpec("Ground Truth", "gt"),
    }

    for local_idx in selected:
        sample = samples[local_idx]
        area_id = int(sample["area_id"])
        grid_x = int(sample["grid_x"])
        grid_y = int(sample["grid_y"])
        block_index = int(sample["index"])

        if area_id not in area_cache:
            area_cache[area_id] = load_area_source_blocks(
                source_base_dir=source_base_dir,
                area_id=area_id,
                cell_size_x=cell_size_x,
                cell_size_y=cell_size_y,
            )
        dense_points = extract_source_block(area_cache[area_id], grid_x=grid_x, grid_y=grid_y)
        point_cloud = np.asarray(dense_points[:, :4], dtype=np.float32)
        gt_label = np.asarray(dense_points[:, 4], dtype=np.uint8)

        ours_npz = Path(sample["files"]["dense_npz"]).resolve()
        ours_data = np.load(ours_npz)
        ours_prob = np.asarray(ours_data["leak_prob"], dtype=np.float32)

        comparison_columns: list[tuple[str, Image.Image, Image.Image]] = []
        per_model_stats = []

        for model_name in args.model_order:
            spec = model_lookup[model_name]
            if spec.kind == "gt":
                pred_prob = gt_label.astype(np.float32)
            elif spec.kind == "ours":
                pred_prob = ours_prob
            else:
                model_cache_dir = ensure_dir(cache_dir / spec.kind)
                stem = f"area_{area_id:02d}_gx_{grid_x:03d}_gy_{grid_y:02d}"
                cache_path = model_cache_dir / f"{stem}.npz"
                if cache_path.exists():
                    cache_npz = np.load(cache_path)
                    pred_prob = np.asarray(cache_npz["leak_prob"], dtype=np.float32)
                else:
                    if spec.kind == "dgcnn":
                        pred_prob, vote_count = infer_dense_probabilities_generic(
                            point_cloud=point_cloud,
                            model=dgcnn_model,
                            predict_batch_fn=predict_batch_dgcnn,
                            device=device,
                            in_channels=int(dgcnn_spec.get("in_channels", 4)),
                            sample_points=sample_points,
                            num_votes=int(args.num_votes),
                            vote_batch_size=int(args.vote_batch_size),
                            rng=np.random.default_rng(rng_seed + block_index + 101),
                        )
                    elif spec.kind == "randla":
                        pred_prob, vote_count = infer_dense_probabilities_generic(
                            point_cloud=point_cloud,
                            model=randla_model,
                            predict_batch_fn=predict_batch_randla,
                            device=device,
                            in_channels=int(randla_spec.get("in_channels", 4)),
                            sample_points=sample_points,
                            num_votes=int(args.num_votes),
                            vote_batch_size=int(args.vote_batch_size),
                            rng=np.random.default_rng(rng_seed + block_index + 202),
                        )
                    else:
                        raise ValueError(f"Unsupported model kind: {spec.kind}")
                    pred_prob, unknown_count, fill_method = fill_unsampled_probabilities(
                        dense_points_xyz=point_cloud[:, :3],
                        leak_prob=pred_prob,
                        vote_count=vote_count,
                        knn_k=int(args.knn_k),
                    )
                    np.savez_compressed(
                        cache_path,
                        leak_prob=pred_prob.astype(np.float32),
                        vote_count=vote_count.astype(np.int32),
                        fill_unknown_count=np.array([unknown_count], dtype=np.int32),
                    )
                pred_prob = pred_prob.astype(np.float32, copy=False)

            overview, unwrap = render_model_column(point_cloud=point_cloud, pred_prob=pred_prob, args=args)
            comparison_columns.append((model_name, overview, unwrap))
            per_model_stats.append(
                {
                    "model": model_name,
                    "positive_ratio": float((pred_prob >= float(args.prob_threshold)).mean()),
                }
            )

        sample_title = (
            f"Sample #{int(sample['rank'])}  |  Area {area_id}  |  Grid ({grid_x}, {grid_y})  |  "
            f"GT {float(gt_label.mean()) * 100.0:.2f}%"
        )
        sample_figure = compose_sample_figure(
            sample_title=sample_title,
            columns=comparison_columns,
            args=args,
            title_font=title_font,
            label_font=label_font,
        )
        sample_path = sample_fig_dir / f"compare_rank_{int(sample['rank']):02d}_area_{area_id:02d}_gx_{grid_x:03d}_gy_{grid_y:02d}.png"
        sample_figure.save(sample_path)
        sample_figures.append(sample_figure)
        saved_samples.append(
            {
                "rank": int(sample["rank"]),
                "area_id": area_id,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "figure": str(sample_path),
                "models": per_model_stats,
            }
        )
        print(f"saved {sample_path}", flush=True)

    gallery = concat_vertical(sample_figures, gap=int(args.sample_gap), background=FIG_BG)
    gallery_path = output_dir / "comparison_gallery_reference.png"
    gallery.save(gallery_path)
    print(f"saved {gallery_path}", flush=True)

    summary_out = {
        "dense_summary_path": str(args.dense_summary_path.resolve()),
        "output_dir": str(output_dir),
        "gallery": str(gallery_path),
        "models": args.model_order,
        "saved_samples": saved_samples,
        "num_votes": int(args.num_votes),
        "prob_threshold": float(args.prob_threshold),
    }
    summary_path = output_dir / "comparison_gallery_summary.json"
    save_json(summary_path, summary_out)
    print(f"saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
