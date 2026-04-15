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

from leakage_detection_torch.render_paper_figure import (
    crop_content,
    fit_contain,
    load_area_overview_points,
    render_overview_layers,
)


PANEL_BG = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Area-level tunnel intro overviews for PPT opening slides."
    )
    parser.add_argument(
        "--source_base_dir",
        type=Path,
        default=Path("/ai/0309/cloud"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260327_223510/area_intro_overviews"),
    )
    parser.add_argument("--area_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--background_points", type=int, default=140000)
    parser.add_argument("--leakage_points", type=int, default=18000)
    parser.add_argument("--output_width", type=int, default=2200)
    parser.add_argument("--output_height", type=int, default=1100)
    parser.add_argument("--render_width", type=int, default=2800)
    parser.add_argument("--render_height", type=int, default=1500)
    parser.add_argument("--bg_point_size", type=float, default=0.72)
    parser.add_argument("--fg_point_size", type=float, default=2.0)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-74.0)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--include_leakage", action="store_true", default=True)
    parser.add_argument("--pure_gray", action="store_true", default=False)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.resolve())
    rng_master = np.random.default_rng(int(args.seed))

    summary = {
        "source_base_dir": str(args.source_base_dir.resolve()),
        "output_dir": str(output_dir),
        "area_ids": [int(v) for v in args.area_ids],
        "background_points": int(args.background_points),
        "leakage_points": int(args.leakage_points),
        "output_width": int(args.output_width),
        "output_height": int(args.output_height),
        "render_width": int(args.render_width),
        "render_height": int(args.render_height),
        "elev": float(args.elev),
        "azim": float(args.azim),
        "include_leakage": bool(args.include_leakage),
        "pure_gray": bool(args.pure_gray),
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
        if points.shape[1] < 4:
            raise RuntimeError(f"Area {area_id} overview points have invalid shape: {points.shape}")

        leakage_points = None
        if bool(args.include_leakage) and not bool(args.pure_gray):
            area_root = Path(overview["area_root"])
            leak_path = area_root / "leakage.txt"
            if leak_path.exists():
                leak_data = np.fromfile(leak_path, sep=" ", dtype=np.float32)
                if leak_data.size >= 4:
                    leak_data = leak_data.reshape(-1, 4)
                    if len(leak_data) > int(args.leakage_points):
                        pick = np.sort(rng.choice(len(leak_data), size=int(args.leakage_points), replace=False))
                        leak_data = leak_data[pick]
                    leakage_points = leak_data.astype(np.float32, copy=False)

        raw_image = render_overview_layers(
            base_point_cloud=points,
            overlay_points=leakage_points,
            output_size=(int(args.render_width), int(args.render_height)),
            max_background_points=int(args.background_points),
            max_overlay_points=int(args.leakage_points) if leakage_points is not None else 0,
            bg_point_size=float(args.bg_point_size),
            overlay_point_size=float(args.fg_point_size),
            elev=float(args.elev),
            azim=float(args.azim),
            rng=rng,
        )
        final_image = fit_contain(
            crop_content(raw_image, threshold=247, padding=18),
            int(args.output_width),
            int(args.output_height),
        )
        file_name = f"area_{int(area_id):02d}_intro_overview.png"
        final_path = output_dir / file_name
        final_image.save(final_path)
        print(f"saved {final_path}", flush=True)

        summary["areas"].append(
            {
                "area_id": int(area_id),
                "file": str(final_path),
                "background_count_total": int(overview["background_count_total"]),
                "leakage_count_total": int(overview["leakage_count_total"]),
                "area_root": str(overview["area_root"]),
            }
        )

    save_json(output_dir / "area_intro_overview_summary.json", summary)
    print(f"saved {output_dir / 'area_intro_overview_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
