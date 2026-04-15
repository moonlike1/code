from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pure-image PPT showcases by blending sharp and refined outputs.")
    parser.add_argument("--sharp_dir", type=Path, required=True)
    parser.add_argument("--refined_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--blend_alpha", type=float, default=0.42, help="Weight of sharp image in right-panel blend.")
    parser.add_argument("--canvas_width", type=int, default=3000)
    parser.add_argument("--canvas_height", type=int, default=1180)
    parser.add_argument("--margin", type=int, default=28)
    parser.add_argument("--panel_gap", type=int, default=26)
    parser.add_argument("--left_width", type=int, default=900)
    parser.add_argument("--sample_glob", type=str, default="showcase_rank_*.png")
    return parser.parse_args()


def crop_sharp_panels(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    # Source layout from render_ppt_showcase.py sharp preset:
    # canvas=3000x1420, margin=42, body_top=132, left_width=900, panel_gap=28
    left = image.crop((42, 134, 42 + 900, 134 + 1192))
    right = image.crop((970, 134, 970 + 1988, 134 + 1192))
    return left, right


def crop_refined_panels(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    # Source layout from render_ppt_showcase.py refined_v2 preset:
    # canvas=2400x1220, margin=42, body_top=132, left_width=760, panel_gap=28
    left = image.crop((42, 134, 42 + 760, 134 + 992))
    right = image.crop((830, 134, 830 + 1528, 134 + 992))
    return left, right


def fit_contain(image: Image.Image, width: int, height: int, background: tuple[int, int, int]) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), background)
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def add_border(image: Image.Image, color: tuple[int, int, int] = (225, 228, 232)) -> Image.Image:
    out = image.copy()
    px = out.load()
    w, h = out.size
    for x in range(w):
        px[x, 0] = color
        px[x, h - 1] = color
    for y in range(h):
        px[0, y] = color
        px[w - 1, y] = color
    return out


def compose_pure_canvas(
    left: Image.Image,
    right: Image.Image,
    canvas_width: int,
    canvas_height: int,
    margin: int,
    panel_gap: int,
    left_width: int,
) -> Image.Image:
    bg = (255, 255, 255)
    canvas = Image.new("RGB", (canvas_width, canvas_height), bg)
    right_width = canvas_width - margin * 2 - panel_gap - left_width
    inner_height = canvas_height - margin * 2

    left_fit = fit_contain(left, left_width, inner_height, bg)
    right_fit = fit_contain(right, right_width, inner_height, bg)
    left_fit = add_border(left_fit)
    right_fit = add_border(right_fit)

    x_left = margin
    x_right = margin + left_width + panel_gap
    y_left = margin + (inner_height - left_fit.height) // 2
    y_right = margin + (inner_height - right_fit.height) // 2
    canvas.paste(left_fit, (x_left, y_left))
    canvas.paste(right_fit, (x_right, y_right))
    return canvas


def main() -> None:
    args = parse_args()
    sharp_dir = args.sharp_dir.resolve()
    refined_dir = args.refined_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sharp_files = sorted(sharp_dir.glob(args.sample_glob))
    if not sharp_files:
        raise RuntimeError(f"No files matched in sharp_dir: {sharp_dir}")

    for sharp_path in sharp_files:
        refined_path = refined_dir / sharp_path.name
        if not refined_path.exists():
            raise FileNotFoundError(f"Missing refined pair: {refined_path}")

        sharp_image = Image.open(sharp_path).convert("RGB")
        refined_image = Image.open(refined_path).convert("RGB")

        sharp_left, sharp_right = crop_sharp_panels(sharp_image)
        _, refined_right = crop_refined_panels(refined_image)
        refined_right = refined_right.resize(sharp_right.size, Image.Resampling.LANCZOS)

        blended_right = Image.blend(refined_right, sharp_right, alpha=float(args.blend_alpha))
        blended_right = blended_right.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2))

        pure = compose_pure_canvas(
            left=sharp_left,
            right=blended_right,
            canvas_width=int(args.canvas_width),
            canvas_height=int(args.canvas_height),
            margin=int(args.margin),
            panel_gap=int(args.panel_gap),
            left_width=int(args.left_width),
        )
        out_path = output_dir / sharp_path.name
        pure.save(out_path)
        print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
