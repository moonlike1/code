from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from leakage_detection_torch.render_area_intro_overviews import ensure_dir, save_json
from leakage_detection_torch.render_paper_figure import (
    crop_content,
    fit_contain,
    reservoir_sample_txt_points,
    render_overview_layers,
)
from leakage_detection_torch.render_ppt_texture_showcase import render_dense_local_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerender selected local tunnel segments at higher density and split them into two halves."
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=Path("/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260327_223510/area_intro_segments_tight/area_intro_segments_summary.json"),
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["5:1"],
        help="Target segments in the form area_id:segment_rank, e.g. 5:1 4:2",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260327_223510/area_intro_segments_dense_split"),
    )
    parser.add_argument("--render_mode", type=str, default="surfel", choices=["scatter", "surfel"])
    parser.add_argument("--background_points", type=int, default=420000)
    parser.add_argument("--leakage_points", type=int, default=72000)
    parser.add_argument("--output_width", type=int, default=2400)
    parser.add_argument("--output_height", type=int, default=1200)
    parser.add_argument("--render_width", type=int, default=3200)
    parser.add_argument("--render_height", type=int, default=1800)
    parser.add_argument("--bg_point_size", type=float, default=1.10)
    parser.add_argument("--fg_point_size", type=float, default=2.70)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-64.0)
    parser.add_argument("--overview_axis_mode", type=str, default="xz_y", choices=["xy_z", "xz_y", "yz_x"])
    parser.add_argument("--seed", type=int, default=20260330)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_targets(values: list[str]) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for value in values:
        area_str, rank_str = value.split(":", 1)
        parsed.append((int(area_str), int(rank_str)))
    return parsed


def load_leakage_sample(area_root: Path, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    leak_sample, _ = reservoir_sample_txt_points(area_root / "leakage.txt", sample_size, rng)
    return np.asarray(leak_sample, dtype=np.float32)


def load_background_sample(area_root: Path, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    bg_sample, _ = reservoir_sample_txt_points(area_root / "background.txt", sample_size, rng)
    return np.asarray(bg_sample, dtype=np.float32)


def render_segment_image(
    points: np.ndarray,
    leak_points: np.ndarray | None,
    leak_prob: np.ndarray | None,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> "Image.Image":
    if str(args.render_mode).lower() == "surfel":
        render_args = SimpleNamespace(
            prob_threshold=0.5,
            overview_mode="surfel",
            overview_width=int(args.render_width),
            overview_height=int(args.render_height),
            overview_axis_mode=str(args.overview_axis_mode),
            overview_elev=float(args.elev),
            overview_azim=float(args.azim),
            overview_bg_point_size=float(args.bg_point_size),
            overview_fg_point_size=float(args.fg_point_size),
            overview_supersample=1.35,
            overview_perspective=0.10,
            overview_smooth_sigma=1.35,
            overview_coarse_sigma=3.8,
            overview_detail_sigma=6.2,
            overview_support_sigma=2.2,
            overview_detail_amount=0.46,
            overview_gray_floor=0.72,
            overview_gray_gain=0.24,
            overview_pred_sigma=1.05,
            overview_pred_focus_threshold=0.17,
            overview_pred_min_strength=0.82,
            overview_pred_close_radius=2,
            overview_pred_dilate_radius=1,
            overview_unsharp_radius=1.3,
            overview_unsharp_percent=142,
            overview_unsharp_threshold=2,
        )
        raw = render_dense_local_overview(
            point_cloud=points,
            pred_prob=np.asarray(leak_prob, dtype=np.float32),
            args=render_args,
        )
    else:
        raw = render_overview_layers(
            base_point_cloud=points,
            overlay_points=leak_points,
            output_size=(int(args.render_width), int(args.render_height)),
            max_background_points=max(int(len(points)), 1),
            max_overlay_points=max(int(len(leak_points)) if leak_points is not None else 0, 0),
            bg_point_size=float(args.bg_point_size),
            overlay_point_size=float(args.fg_point_size),
            elev=float(args.elev),
            azim=float(args.azim),
            rng=rng,
        )
    return fit_contain(
        crop_content(raw, threshold=247, padding=20),
        int(args.output_width),
        int(args.output_height),
    )


def select_window(points: np.ndarray, x0: float, x1: float) -> np.ndarray:
    mask = (points[:, 0] >= float(x0)) & (points[:, 0] <= float(x1))
    return np.asarray(points[mask], dtype=np.float32)


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary_path.resolve())
    output_dir = ensure_dir(args.output_dir.resolve())
    rng_master = np.random.default_rng(int(args.seed))
    targets = parse_targets(args.targets)

    summary_map: dict[tuple[int, int], dict] = {}
    for area in summary["areas"]:
        area_id = int(area["area_id"])
        for segment in area["segments"]:
            summary_map[(area_id, int(segment["rank"]))] = {
                "area_root": Path(area["area_root"]),
                "x_range": [float(segment["x_range"][0]), float(segment["x_range"][1])],
            }

    saved_items: list[dict] = []
    for area_id, rank in targets:
        if (area_id, rank) not in summary_map:
            raise KeyError(f"Target area {area_id} rank {rank} not found in {args.summary_path}")

        info = summary_map[(area_id, rank)]
        area_root = Path(info["area_root"])
        source_base_dir = area_root.parent.parent
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        bg_points = load_background_sample(area_root, int(args.background_points), rng)
        leak_points = load_leakage_sample(area_root, int(args.leakage_points), rng)
        if len(bg_points) == 0 and len(leak_points) == 0:
            raise RuntimeError(f"No points loaded for area {area_id}")
        if len(bg_points) == 0:
            points = leak_points.copy()
            leak_prob = np.ones((len(leak_points),), dtype=np.float32)
        elif len(leak_points) == 0:
            points = bg_points.copy()
            leak_prob = np.zeros((len(bg_points),), dtype=np.float32)
        else:
            points = np.vstack([bg_points, leak_points]).astype(np.float32, copy=False)
            leak_prob = np.concatenate(
                [
                    np.zeros((len(bg_points),), dtype=np.float32),
                    np.ones((len(leak_points),), dtype=np.float32),
                ],
                axis=0,
            )

        x0, x1 = info["x_range"]
        x_mid = 0.5 * (x0 + x1)

        windows = [
            ("full", x0, x1),
            ("left", x0, x_mid),
            ("right", x_mid, x1),
        ]
        item_record = {
            "area_id": int(area_id),
            "segment_rank": int(rank),
            "x_range": [float(x0), float(x1)],
            "files": {},
        }

        for label, wx0, wx1 in windows:
            seg_mask = (points[:, 0] >= float(wx0)) & (points[:, 0] <= float(wx1))
            seg_points = np.asarray(points[seg_mask], dtype=np.float32)
            seg_prob = np.asarray(leak_prob[seg_mask], dtype=np.float32)
            seg_leak = np.asarray(seg_points[seg_prob > 0.5], dtype=np.float32)
            if len(seg_leak) == 0:
                seg_leak = None
            image = render_segment_image(seg_points, seg_leak, seg_prob, args=args, rng=rng)
            out_path = output_dir / f"area_{int(area_id):02d}_segment_{int(rank):02d}_{label}.png"
            image.save(out_path)
            print(f"saved {out_path}", flush=True)
            item_record["files"][label] = str(out_path)
            item_record[f"{label}_point_count"] = int(len(seg_points))
            item_record[f"{label}_leakage_point_count"] = int(len(seg_leak)) if seg_leak is not None else 0

        saved_items.append(item_record)

    save_json(
        output_dir / "dense_split_summary.json",
        {
            "summary_path": str(args.summary_path.resolve()),
            "output_dir": str(output_dir),
            "targets": [{"area_id": a, "segment_rank": r} for a, r in targets],
            "background_points": int(args.background_points),
            "leakage_points": int(args.leakage_points),
            "saved": saved_items,
        },
    )
    print(f"saved {output_dir / 'dense_split_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
