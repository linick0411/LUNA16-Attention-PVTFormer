"""Run the four model experiments in a consistent order and environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


MODELS = {
    "baseline": ("train_baseline.py", "eval_baseline.py"),
    "concat_25d": ("train_concat_25d.py", "eval_concat_25d.py"),
    "attention_gate": ("train_attention_gate.py", "eval_attention_gate.py"),
    "voxel_attention": ("train_voxel_attention.py", "eval_voxel_attention.py"),
    "coordinate_attention": ("train_coordinate_attention.py", "eval_coordinate_attention.py"),
}

CHECKPOINTS = {
    "baseline": "checkpoint_baseline.pth",
    "concat_25d": "checkpoint_concat_25d.pth",
    "attention_gate": "checkpoint_attention_gate.pth",
    "voxel_attention": "checkpoint_voxel_attention.pth",
    "coordinate_attention": "checkpoint_coordinate_attention.pth",
}


def parse_models(value):
    if value == "all":
        return list(MODELS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [name for name in names if name not in MODELS]
    if not names or invalid:
        raise argparse.ArgumentTypeError(
            f"models must be 'all' or a comma-separated subset of: {', '.join(MODELS)}"
        )
    return names


def build_commands(stage, models):
    commands = []
    for model in models:
        train_script, eval_script = MODELS[model]
        if stage in ("train", "all"):
            commands.append((model, "train", [sys.executable, train_script]))
        if stage in ("evaluate", "all"):
            commands.append((model, "evaluate", [sys.executable, eval_script]))
    return commands


def should_skip_completed(model, stage, checkpoints_dir, results_dir):
    if stage == "train":
        return (checkpoints_dir / CHECKPOINTS[model]).is_file() and ((checkpoints_dir / CHECKPOINTS[model]).with_suffix(".complete.json").is_file() or (results_dir / model / "metrics.json").is_file())
    if stage == "evaluate":
        return (results_dir / model / "metrics.json").is_file()
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--models", type=parse_models, default=list(MODELS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip training or evaluation stages whose final artifact already exists.",
    )
    args = parser.parse_args()

    commands = build_commands(args.stage, args.models)
    checkpoints_dir = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    for model, stage, command in commands:
        if args.resume and should_skip_completed(model, stage, checkpoints_dir, results_dir):
            print(f"[{model}] {stage}: skipped (completed artifact exists)", flush=True)
            continue
        print(f"[{model}] {stage}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
