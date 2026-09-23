"""Create report-ready loss and F1 curves from a training history CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path):
    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        raise ValueError(f"History CSV has no epochs: {path}")
    return rows


def plot_history(history_path, output_path=None):
    history_path = Path(history_path)
    rows = read_history(history_path)
    epochs = [int(row["epoch"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="Train")
    axes[0].plot(epochs, [float(row["valid_loss"]) for row in rows], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Dice + BCE loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [float(row["train_f1"]) for row in rows], label="Train")
    axes[1].plot(epochs, [float(row["valid_f1"]) for row in rows], label="Validation")
    axes[1].set(title="F1 / Dice", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.suptitle(history_path.stem)
    figure.tight_layout()
    output_path = Path(output_path) if output_path else history_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_history(args.history_csv, args.output)
