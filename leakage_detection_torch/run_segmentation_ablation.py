import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_v2_torch.py"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "output" / "ablation_exec"


@dataclass(frozen=True)
class SegmentationAblationSpec:
    experiment_id: str
    experiment_name: str
    description: str
    overrides: dict[str, object]


BASE_SEGMENTATION_CONFIG = {
    "train_path": str(ROOT_DIR / "full_area_train_v2.h5"),
    "test_path": str(ROOT_DIR / "full_area_val_v2.h5"),
    "num_points": 4096,
    "batch_size": 8,
    "epochs": 200,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "val_interval": 5,
    "augment": True,
    "augment_mode": "strong",
    "device": "cuda",
    "seg_only": True,
    "train_sampler": "small_positive",
    "seg_loss": "dice_ce",
    "seg_use_class_weights": True,
    "seg_class_weight_mode": "sqrt_inv",
    "boundary_input_mode": "features_fine_probs",
    "k_scales": [10, 20, 40],
    "use_noise_guidance": True,
    "use_progressive": True,
    "use_noise_leak_corr": True,
    "use_multi_scale": True,
    "best_metric": "val_global_leak_iou",
}


SEGMENTATION_ABLATIONS = {
    "A0": SegmentationAblationSpec(
        experiment_id="A0",
        experiment_name="full_seg",
        description="完整分割模型",
        overrides={},
    ),
    "A1": SegmentationAblationSpec(
        experiment_id="A1",
        experiment_name="wo_ng",
        description="去掉噪声引导编码",
        overrides={"use_noise_guidance": False},
    ),
    "A2": SegmentationAblationSpec(
        experiment_id="A2",
        experiment_name="wo_ms",
        description="多尺度结构退化为单尺度 [20]",
        overrides={"use_multi_scale": False, "k_scales": [20]},
    ),
    "A3": SegmentationAblationSpec(
        experiment_id="A3",
        experiment_name="wo_prog",
        description="去掉渐进式分割结构",
        overrides={"use_progressive": False},
    ),
    "A4": SegmentationAblationSpec(
        experiment_id="A4",
        experiment_name="wo_corr",
        description="去掉噪声-渗漏相关性校正",
        overrides={"use_noise_leak_corr": False},
    ),
    "A5": SegmentationAblationSpec(
        experiment_id="A5",
        experiment_name="wo_prob_boundary",
        description="边界分支不使用 fine 概率输入",
        overrides={"boundary_input_mode": "features"},
    ),
}


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_ids", nargs="+", choices=sorted(SEGMENTATION_ABLATIONS.keys()), default=["A0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--train_path", type=str, default=BASE_SEGMENTATION_CONFIG["train_path"])
    parser.add_argument("--test_path", type=str, default=BASE_SEGMENTATION_CONFIG["test_path"])
    parser.add_argument("--num_points", type=int, default=BASE_SEGMENTATION_CONFIG["num_points"])
    parser.add_argument("--batch_size", type=int, default=BASE_SEGMENTATION_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=BASE_SEGMENTATION_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=BASE_SEGMENTATION_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=BASE_SEGMENTATION_CONFIG["weight_decay"])
    parser.add_argument("--val_interval", type=int, default=BASE_SEGMENTATION_CONFIG["val_interval"])
    parser.add_argument("--augment", dest="augment", action="store_true", default=BASE_SEGMENTATION_CONFIG["augment"])
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--augment_mode", type=str, choices=["basic", "strong"], default=BASE_SEGMENTATION_CONFIG["augment_mode"])
    parser.add_argument("--device", type=str, default=BASE_SEGMENTATION_CONFIG["device"])
    parser.add_argument(
        "--train_sampler",
        type=str,
        choices=["shuffle", "balanced_presence", "small_positive"],
        default=BASE_SEGMENTATION_CONFIG["train_sampler"],
    )
    parser.add_argument("--seg_loss", type=str, choices=["ce", "dice_ce", "focal_ce"], default=BASE_SEGMENTATION_CONFIG["seg_loss"])
    parser.add_argument(
        "--seg_class_weight_mode",
        type=str,
        choices=["sqrt_inv", "inverse", "effective_num"],
        default=BASE_SEGMENTATION_CONFIG["seg_class_weight_mode"],
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        choices=[
            "val_iou",
            "val_leak_iou",
            "val_global_leak_iou",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_accuracy",
        ],
        default=BASE_SEGMENTATION_CONFIG["best_metric"],
    )
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--list", action="store_true", default=False)
    return parser


def get_experiment_config(cli_args, spec: SegmentationAblationSpec) -> dict[str, object]:
    config = dict(BASE_SEGMENTATION_CONFIG)
    config.update(
        {
            "train_path": cli_args.train_path,
            "test_path": cli_args.test_path,
            "num_points": cli_args.num_points,
            "batch_size": cli_args.batch_size,
            "epochs": cli_args.epochs,
            "lr": cli_args.lr,
            "weight_decay": cli_args.weight_decay,
            "val_interval": cli_args.val_interval,
            "augment": cli_args.augment,
            "augment_mode": cli_args.augment_mode,
            "device": cli_args.device,
            "train_sampler": cli_args.train_sampler,
            "seg_loss": cli_args.seg_loss,
            "seg_class_weight_mode": cli_args.seg_class_weight_mode,
            "best_metric": cli_args.best_metric,
        }
    )
    config.update(spec.overrides)
    return config


def build_train_command(python_bin: str, train_script: Path, output_dir: Path, seed: int, config: dict[str, object]) -> list[str]:
    cmd = [
        python_bin,
        str(train_script),
        "--train_path",
        str(config["train_path"]),
        "--test_path",
        str(config["test_path"]),
        "--num_points",
        str(config["num_points"]),
        "--batch_size",
        str(config["batch_size"]),
        "--epochs",
        str(config["epochs"]),
        "--lr",
        str(config["lr"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--train_sampler",
        str(config["train_sampler"]),
        "--seg_loss",
        str(config["seg_loss"]),
        "--boundary_input_mode",
        str(config["boundary_input_mode"]),
        "--output_dir",
        str(output_dir),
        "--device",
        str(config["device"]),
        "--seed",
        str(seed),
        "--val_interval",
        str(config["val_interval"]),
        "--best_metric",
        str(config["best_metric"]),
        "--k_scales",
        *[str(v) for v in config["k_scales"]],
    ]

    if config["augment"]:
        cmd.append("--augment")
    if config["seg_only"]:
        cmd.append("--seg_only")
    if config["seg_use_class_weights"]:
        cmd.extend(
            [
                "--seg_use_class_weights",
                "--seg_class_weight_mode",
                str(config["seg_class_weight_mode"]),
            ]
        )

    cmd.append("--use_noise_guidance" if config["use_noise_guidance"] else "--no_noise_guidance")
    cmd.append("--use_progressive" if config["use_progressive"] else "--no_progressive")
    cmd.append("--use_noise_leak_corr" if config["use_noise_leak_corr"] else "--no_noise_leak_corr")
    cmd.append("--use_multi_scale" if config["use_multi_scale"] else "--no_multi_scale")
    return cmd


def has_completed_run(experiment_dir: Path) -> bool:
    if not experiment_dir.exists():
        return False
    return any(path.name == "history.json" for path in experiment_dir.rglob("history.json"))


def print_registry():
    for experiment_id in sorted(SEGMENTATION_ABLATIONS.keys()):
        spec = SEGMENTATION_ABLATIONS[experiment_id]
        print(f"{spec.experiment_id} {spec.experiment_name}: {spec.description}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list:
        print_registry()
        return

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for ablation_id in args.ablation_ids:
        spec = SEGMENTATION_ABLATIONS[ablation_id]
        config = get_experiment_config(args, spec)
        for seed in args.seeds:
            experiment_dir = output_root / f"{spec.experiment_id}_{spec.experiment_name}_seed{seed}"
            if args.skip_existing and has_completed_run(experiment_dir):
                print(f"[skip] {experiment_dir} 已存在完成结果", flush=True)
                continue

            manifest = {
                "prepared_at": datetime.now().isoformat(timespec="seconds"),
                "experiment_id": spec.experiment_id,
                "experiment_name": spec.experiment_name,
                "description": spec.description,
                "seed": int(seed),
                "output_dir": str(experiment_dir),
                "train_script": str(TRAIN_SCRIPT),
                "config": config,
            }
            save_json(experiment_dir / "ablation_spec.json", manifest)

            cmd = build_train_command(args.python_bin, TRAIN_SCRIPT, experiment_dir, seed, config)
            print(f"[prepare] {spec.experiment_id} seed={seed} -> {experiment_dir}", flush=True)
            print(shlex.join(cmd), flush=True)

            if args.dry_run:
                continue

            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            subprocess.run(cmd, check=True, env=env, cwd=str(ROOT_DIR))


if __name__ == "__main__":
    main()
