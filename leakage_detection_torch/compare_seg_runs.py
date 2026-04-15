from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent

PRESET_GROUPS: dict[str, list[tuple[str, Path]]] = {
    "next3": [
        ("Exp-A", ROOT_DIR / "output" / "seg_exp_a"),
        ("Exp-B", ROOT_DIR / "output" / "seg_exp_b"),
        ("Exp-C", ROOT_DIR / "output" / "seg_exp_c"),
    ],
    "legacy3": [
        ("Current", ROOT_DIR / "output" / "v2_training_torch_dataset_v2_segonly"),
        ("Balanced", ROOT_DIR / "output" / "seg_balanced"),
        ("Recall", ROOT_DIR / "output" / "seg_recall"),
    ],
}

METRIC_FIELD_MAP = {
    "loss": "val_loss",
    "iou": "val_iou",
    "leak_iou": "val_leak_iou",
    "precision": "val_precision",
    "recall": "val_recall",
    "f1": "val_f1",
    "global_leak_iou": "val_global_leak_iou",
    "global_bg_iou": "val_global_bg_iou",
    "global_miou": "val_global_miou",
    "accuracy": "val_accuracy",
    "cls_loss": "val_cls_loss",
    "cls_acc": "val_cls_acc",
}

SUMMARY_METRICS = ["global_leak_iou", "f1", "recall", "precision", "iou", "leak_iou"]


@dataclass
class GroupSelection:
    label: str
    input_path: str
    latest_run_dir: str | None
    status: str
    best_metric: str | None = None
    best_score: float | None = None
    best_epoch: int | None = None
    val_iou: float | None = None
    val_leak_iou: float | None = None
    val_precision: float | None = None
    val_recall: float | None = None
    val_f1: float | None = None
    val_global_leak_iou: float | None = None
    best_checkpoint: str | None = None
    best_checkpoint_source: str | None = None
    history_path: str | None = None
    args_path: str | None = None
    topk_path: str | None = None
    note: str | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="自动对比分割实验最新 run 的 global leak IoU / F1 / recall / best checkpoint"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_GROUPS.keys()),
        default="next3",
        help="使用内置的 3 组实验目录预设；传 --group 时会覆盖这个预设",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="手动指定待对比的组，既可传 output root，也可直接传某个 run 目录",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="可选：把完整结果另存为 JSON",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="",
        help="可选：把摘要结果另存为 Markdown",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_group_specs(raw_groups: list[str]) -> list[tuple[str, Path]]:
    groups: list[tuple[str, Path]] = []
    for item in raw_groups:
        if "=" not in item:
            raise ValueError(f"无效的 --group 参数：{item}，需要形如 LABEL=PATH")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"无效的 --group 参数：{item}，LABEL 和 PATH 都不能为空")
        groups.append((label, Path(raw_path).expanduser().resolve()))
    return groups


def extract_timestamp_key(path: Path) -> tuple[int, float]:
    name = path.name
    try:
        timestamp = datetime.strptime(name, "%Y%m%d_%H%M%S")
        return (1, timestamp.timestamp())
    except ValueError:
        history_path = path / "history.json"
        ref_path = history_path if history_path.exists() else path
        return (0, ref_path.stat().st_mtime)


def discover_latest_run(path: Path) -> Path | None:
    if (path / "history.json").exists():
        return path

    if not path.exists():
        return None

    run_dirs = sorted({history_path.parent.resolve() for history_path in path.rglob("history.json")})
    if not run_dirs:
        return None

    return max(run_dirs, key=extract_timestamp_key)


def best_metric_key(best_metric: str | None) -> str | None:
    if best_metric is None:
        return None
    return best_metric[4:] if best_metric.startswith("val_") else best_metric


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_metrics_from_history(history: dict[str, Any], epoch: int | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if epoch is None:
        return metrics

    val_epochs = history.get("val_epochs")
    if not isinstance(val_epochs, list) or epoch not in val_epochs:
        return metrics

    index = val_epochs.index(epoch)
    for metric_name, history_key in METRIC_FIELD_MAP.items():
        values = history.get(history_key)
        if isinstance(values, list) and index < len(values):
            metric_value = coerce_float(values[index])
            if metric_value is not None:
                metrics[metric_name] = metric_value
    return metrics


def resolve_best_checkpoint(run_dir: Path, history: dict[str, Any], args_data: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    best_checkpoint_path = run_dir / "best_checkpoint.json"
    payload = load_json(best_checkpoint_path)
    if payload is not None:
        return payload, "best_checkpoint.json"

    metrics = history.get("best_val_metrics")
    best_metric = history.get("best_metric")
    if best_metric is None and args_data is not None:
        best_metric = args_data.get("best_metric")
    best_epoch = coerce_int(history.get("best_epoch"))

    if not isinstance(metrics, dict):
        metrics = collect_metrics_from_history(history, best_epoch)

    best_score = coerce_float(history.get("best_score"))
    key = best_metric_key(best_metric)
    if best_score is None and key and isinstance(metrics, dict):
        best_score = coerce_float(metrics.get(key))
    if best_score is None:
        best_score = coerce_float(history.get("best_iou"))

    return {
        "epoch": best_epoch,
        "score": best_score,
        "metric": best_metric or "val_iou",
        "path": "best_model.pt" if (run_dir / "best_model.pt").exists() else None,
        "val_metrics": metrics if isinstance(metrics, dict) else {},
    }, "history_inferred"


def build_group_summary(label: str, input_path: Path) -> GroupSelection:
    latest_run = discover_latest_run(input_path)
    if latest_run is None:
        return GroupSelection(
            label=label,
            input_path=str(input_path),
            latest_run_dir=None,
            status="missing",
            note="未找到包含 history.json 的 completed run",
        )

    history_path = latest_run / "history.json"
    args_path = latest_run / "args.json"
    history = load_json(history_path)
    args_data = load_json(args_path)

    if history is None:
        return GroupSelection(
            label=label,
            input_path=str(input_path),
            latest_run_dir=str(latest_run),
            status="invalid",
            history_path=str(history_path),
            note="history.json 不存在或无法读取",
        )

    best_payload, checkpoint_source = resolve_best_checkpoint(latest_run, history, args_data)
    val_metrics = best_payload.get("val_metrics")
    if not isinstance(val_metrics, dict):
        val_metrics = {}

    checkpoint_value = best_payload.get("path")
    checkpoint_path = None
    if isinstance(checkpoint_value, str) and checkpoint_value:
        candidate = Path(checkpoint_value)
        checkpoint_path = candidate if candidate.is_absolute() else (latest_run / candidate)

    topk_path = latest_run / "topk_checkpoints.json"

    return GroupSelection(
        label=label,
        input_path=str(input_path),
        latest_run_dir=str(latest_run),
        status="ok",
        best_metric=str(best_payload.get("metric") or history.get("best_metric") or "val_iou"),
        best_score=coerce_float(best_payload.get("score")),
        best_epoch=coerce_int(best_payload.get("epoch")),
        val_iou=coerce_float(val_metrics.get("iou")),
        val_leak_iou=coerce_float(val_metrics.get("leak_iou")),
        val_precision=coerce_float(val_metrics.get("precision")),
        val_recall=coerce_float(val_metrics.get("recall")),
        val_f1=coerce_float(val_metrics.get("f1")),
        val_global_leak_iou=coerce_float(val_metrics.get("global_leak_iou")),
        best_checkpoint=str(checkpoint_path) if checkpoint_path is not None else None,
        best_checkpoint_source=checkpoint_source,
        history_path=str(history_path),
        args_path=str(args_path) if args_path.exists() else None,
        topk_path=str(topk_path) if topk_path.exists() else None,
    )


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def score_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def delta_text(value: float | None, ref: float | None) -> str:
    if value is None or ref is None:
        return "-"
    delta = (value - ref) * 100.0
    return f"{delta:+.2f} pp"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    divider = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), divider]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def make_markdown(summary: dict[str, Any]) -> str:
    ok_groups = [item for item in summary["groups"] if item["status"] == "ok"]
    missing_groups = [item for item in summary["groups"] if item["status"] != "ok"]

    lines = ["# Segmentation Run Comparison", ""]
    lines.append(f"- generated_at: {summary['generated_at']}")
    lines.append(f"- preset: {summary['preset']}")
    lines.append("")

    if ok_groups:
        lines.append("## Latest Runs")
        lines.append("")
        lines.append("| Group | Run | Best Metric | Best Epoch | gLeak IoU (%) | F1 (%) | Recall (%) | Best Checkpoint |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
        for item in ok_groups:
            checkpoint = item["best_checkpoint"] or "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        item["label"],
                        Path(item["latest_run_dir"]).name if item["latest_run_dir"] else "-",
                        item["best_metric"] or "-",
                        str(item["best_epoch"] or "-"),
                        pct(item["val_global_leak_iou"]),
                        pct(item["val_f1"]),
                        pct(item["val_recall"]),
                        checkpoint,
                    ]
                )
                + " |"
            )
        lines.append("")

    if summary["rankings"]:
        lines.append("## Rankings")
        lines.append("")
        for metric_name, ranking in summary["rankings"].items():
            winner = ranking[0]
            lines.append(
                f"- {metric_name}: {winner['label']} ({pct(winner['value'])}%)"
            )
        lines.append("")

    if summary["deltas_vs_first"]:
        lines.append("## Deltas Vs First Group")
        lines.append("")
        lines.append("| Group | Δ gLeak IoU | Δ F1 | Δ Recall |")
        lines.append("| --- | ---: | ---: | ---: |")
        for item in summary["deltas_vs_first"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        item["label"],
                        item["delta_global_leak_iou"],
                        item["delta_f1"],
                        item["delta_recall"],
                    ]
                )
                + " |"
            )
        lines.append("")

    if missing_groups:
        lines.append("## Missing Or Invalid")
        lines.append("")
        for item in missing_groups:
            lines.append(f"- {item['label']}: {item['status']} ({item['note'] or 'unknown'})")
        lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()

    groups = parse_group_specs(args.group) if args.group else PRESET_GROUPS[args.preset]
    summaries = [build_group_summary(label, path) for label, path in groups]
    ok_summaries = [item for item in summaries if item.status == "ok"]

    if not ok_summaries:
        print("Error: 没有找到任何可对比的 completed runs。", file=sys.stderr)
        sys.exit(1)

    summary_rows = []
    for item in ok_summaries:
        summary_rows.append(
            [
                item.label,
                Path(item.latest_run_dir).name if item.latest_run_dir else "-",
                item.best_metric or "-",
                str(item.best_epoch) if item.best_epoch is not None else "-",
                pct(item.val_global_leak_iou),
                pct(item.val_f1),
                pct(item.val_recall),
                score_text(item.best_score),
            ]
        )

    print("Latest completed runs", flush=True)
    print(
        render_table(
            ["Group", "Run", "Best Metric", "Epoch", "gLeak IoU (%)", "F1 (%)", "Recall (%)", "Best Score"],
            summary_rows,
        ),
        flush=True,
    )
    print("", flush=True)

    checkpoint_rows = []
    for item in ok_summaries:
        checkpoint_rows.append(
            [
                item.label,
                item.best_checkpoint or "-",
                item.best_checkpoint_source or "-",
            ]
        )
    print("Best checkpoints", flush=True)
    print(render_table(["Group", "Checkpoint", "Source"], checkpoint_rows), flush=True)
    print("", flush=True)

    rankings: dict[str, list[dict[str, Any]]] = {}
    for metric_name in SUMMARY_METRICS:
        ranked = []
        for item in ok_summaries:
            value = getattr(item, f"val_{metric_name}" if metric_name != "iou" else "val_iou", None)
            if value is None and metric_name == "global_leak_iou":
                value = item.val_global_leak_iou
            elif value is None and metric_name == "f1":
                value = item.val_f1
            elif value is None and metric_name == "recall":
                value = item.val_recall
            elif value is None and metric_name == "precision":
                value = item.val_precision
            elif value is None and metric_name == "leak_iou":
                value = item.val_leak_iou
            if value is not None:
                ranked.append({"label": item.label, "value": value, "run_dir": item.latest_run_dir})
        if ranked:
            rankings[metric_name] = sorted(ranked, key=lambda x: (-float(x["value"]), x["label"]))

    if rankings:
        print("Metric rankings", flush=True)
        for metric_name, ranked in rankings.items():
            leader = ranked[0]
            print(f"- {metric_name}: {leader['label']} ({pct(leader['value'])}%)", flush=True)
        print("", flush=True)

    deltas_vs_first: list[dict[str, str]] = []
    reference = ok_summaries[0]
    for item in ok_summaries:
        deltas_vs_first.append(
            {
                "label": item.label,
                "delta_global_leak_iou": delta_text(item.val_global_leak_iou, reference.val_global_leak_iou),
                "delta_f1": delta_text(item.val_f1, reference.val_f1),
                "delta_recall": delta_text(item.val_recall, reference.val_recall),
            }
        )

    print(f"Deltas vs {reference.label}", flush=True)
    print(
        render_table(
            ["Group", "Δ gLeak IoU", "Δ F1", "Δ Recall"],
            [[row["label"], row["delta_global_leak_iou"], row["delta_f1"], row["delta_recall"]] for row in deltas_vs_first],
        ),
        flush=True,
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "groups": [asdict(item) for item in summaries],
        "rankings": rankings,
        "deltas_vs_first": deltas_vs_first,
    }

    if args.output_json:
        save_json(Path(args.output_json).expanduser().resolve(), payload)
    if args.output_md:
        save_text(Path(args.output_md).expanduser().resolve(), make_markdown(payload))


if __name__ == "__main__":
    main()
