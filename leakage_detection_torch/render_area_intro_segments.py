from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from leakage_detection_torch.render_area_intro_overviews import ensure_dir, save_json
from leakage_detection_torch.render_paper_figure import (
    crop_content,
    fit_contain,
    load_area_overview_points,
    reservoir_sample_txt_points,
    render_overview_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render local multi-ring tunnel segment overviews for PPT intro slides."
    )
    parser.add_argument("--source_base_dir", type=Path, default=Path("/ai/0309/cloud"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260327_223510/area_intro_segments"),
    )
    parser.add_argument("--area_ids", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--segments_per_area", type=int, default=2)
    parser.add_argument("--background_points", type=int, default=180000)
    parser.add_argument("--leakage_points", type=int, default=26000)
    parser.add_argument("--segment_fraction", type=float, default=0.22)
    parser.add_argument("--candidate_steps", type=int, default=36)
    parser.add_argument("--output_width", type=int, default=2200)
    parser.add_argument("--output_height", type=int, default=1100)
    parser.add_argument("--render_width", type=int, default=2600)
    parser.add_argument("--render_height", type=int, default=1500)
    parser.add_argument("--bg_point_size", type=float, default=0.82)
    parser.add_argument("--fg_point_size", type=float, default=2.4)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-72.0)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--include_leakage", action="store_true", default=True)
    parser.add_argument("--pure_gray", action="store_true", default=False)
    return parser.parse_args()


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 1e-8:
        return 0.0
    return float(inter / union)


def choose_segment_windows(
    base_points: np.ndarray,
    leakage_points: np.ndarray | None,
    segment_fraction: float,
    candidate_steps: int,
    segments_per_area: int,
) -> list[tuple[float, float, float]]:
    x = np.asarray(base_points[:, 0], dtype=np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    span = max(x_max - x_min, 1e-6)
    window = max(float(segment_fraction), 1e-3) * span
    if window >= span * 0.98:
        return [(x_min, x_max, 0.0)]

    leakage_x = None if leakage_points is None or len(leakage_points) == 0 else np.asarray(leakage_points[:, 0], dtype=np.float32)
    starts = np.linspace(x_min, max(x_max - window, x_min), max(int(candidate_steps), 2))
    candidates: list[tuple[float, float, float]] = []
    for start in starts:
        end = float(start + window)
        base_mask = (x >= float(start)) & (x <= end)
        base_count = int(np.sum(base_mask))
        if base_count < 4000:
            continue
        base_density = base_count / max(len(base_points), 1)
        leak_score = 0.0
        leak_count = 0
        if leakage_x is not None:
            leak_mask = (leakage_x >= float(start)) & (leakage_x <= end)
            leak_count = int(np.sum(leak_mask))
            leak_score = leak_count / max(len(leakage_x), 1)
        score = 0.78 * leak_score + 0.22 * base_density
        candidates.append((float(start), float(end), float(score + leak_count * 1e-7)))

    if not candidates:
        return [(x_min + 0.39 * (span - window), x_min + 0.39 * (span - window) + window, 0.0)]

    candidates.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[float, float, float]] = []
    for cand in candidates:
        if all(interval_iou(cand[0], cand[1], old[0], old[1]) <= 0.35 for old in selected):
            selected.append(cand)
        if len(selected) >= int(segments_per_area):
            break
    if not selected:
        selected = [candidates[0]]
    return selected


def load_leakage_sample(area_root: Path, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    leak_path = area_root / "leakage.txt"
    leak_sample, _ = reservoir_sample_txt_points(leak_path, sample_size, rng)
    return np.asarray(leak_sample, dtype=np.float32)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.resolve())
    rng_master = np.random.default_rng(int(args.seed))

    summary = {
        "source_base_dir": str(args.source_base_dir.resolve()),
        "output_dir": str(output_dir),
        "area_ids": [int(v) for v in args.area_ids],
        "segments_per_area": int(args.segments_per_area),
        "segment_fraction": float(args.segment_fraction),
        "candidate_steps": int(args.candidate_steps),
        "background_points": int(args.background_points),
        "leakage_points": int(args.leakage_points),
        "areas": [],
    }

    for area_id in args.area_ids:
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        overview = load_area_overview_points(
            source_base_dir=args.source_base_dir.resolve(),
            area_id=int(area_id),
            background_points=int(args.background_points),
            leakage_points=int(args.leakage_points),
            rng=rng,
        )
        points = np.asarray(overview["points"], dtype=np.float32)
        area_root = Path(overview["area_root"])
        leakage_points = None
        if bool(args.include_leakage) and not bool(args.pure_gray):
            leakage_points = load_leakage_sample(area_root, int(args.leakage_points), rng)

        windows = choose_segment_windows(
            base_points=points,
            leakage_points=leakage_points,
            segment_fraction=float(args.segment_fraction),
            candidate_steps=int(args.candidate_steps),
            segments_per_area=int(args.segments_per_area),
        )

        area_records = []
        for rank, (x0, x1, score) in enumerate(windows, start=1):
            base_mask = (points[:, 0] >= float(x0)) & (points[:, 0] <= float(x1))
            seg_points = np.asarray(points[base_mask], dtype=np.float32)
            seg_leak = None
            if leakage_points is not None and len(leakage_points) > 0:
                leak_mask = (leakage_points[:, 0] >= float(x0)) & (leakage_points[:, 0] <= float(x1))
                seg_leak = np.asarray(leakage_points[leak_mask], dtype=np.float32)
                if len(seg_leak) == 0:
                    seg_leak = None

            raw = render_overview_layers(
                base_point_cloud=seg_points,
                overlay_points=seg_leak,
                output_size=(int(args.render_width), int(args.render_height)),
                max_background_points=max(int(len(seg_points)), 1),
                max_overlay_points=max(int(len(seg_leak)) if seg_leak is not None else 0, 0),
                bg_point_size=float(args.bg_point_size),
                overlay_point_size=float(args.fg_point_size),
                elev=float(args.elev),
                azim=float(args.azim),
                rng=rng,
            )
            final = fit_contain(
                crop_content(raw, threshold=247, padding=18),
                int(args.output_width),
                int(args.output_height),
            )
            file_name = f"area_{int(area_id):02d}_segment_{rank:02d}.png"
            final_path = output_dir / file_name
            final.save(final_path)
            print(f"saved {final_path}", flush=True)

            area_records.append(
                {
                    "rank": int(rank),
                    "file": str(final_path),
                    "x_range": [float(x0), float(x1)],
                    "score": float(score),
                    "base_points": int(len(seg_points)),
                    "leakage_points": int(len(seg_leak)) if seg_leak is not None else 0,
                }
            )

        summary["areas"].append(
            {
                "area_id": int(area_id),
                "area_root": str(area_root),
                "segments": area_records,
            }
        )

    save_json(output_dir / "area_intro_segments_summary.json", summary)
    print(f"saved {output_dir / 'area_intro_segments_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
