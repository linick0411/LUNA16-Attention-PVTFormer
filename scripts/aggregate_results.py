"""Combine per-model metrics.json files into CSV and Markdown comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_ORDER = ("baseline", "concat_25d", "attention_gate", "voxel_attention", "coordinate_attention")
METRICS = (
    "jaccard",
    "f1",
    "recall",
    "precision",
    "accuracy",
    "f2",
    "hausdorff_pixels",
    "auc",
    "negative_slice_false_positive_rate",
    "mean_fps",
)


def aggregate_results(results_root):
    results_root = Path(results_root)
    rows = []
    for model in MODEL_ORDER:
        path = results_root / model / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"model": model, **{metric: payload.get(metric) for metric in METRICS}})

    csv_path = results_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=("model", *METRICS))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = results_root / "summary.md"
    headers = ("Model", "Jaccard", "F1", "Recall", "Precision", "F2", "HD px", "Neg FP rate", "FPS")
    selected = ("model", "jaccard", "f1", "recall", "precision", "f2", "hausdorff_pixels", "negative_slice_false_positive_rate", "mean_fps")

    def render(value):
        return "N/A" if value is None else (str(value) if isinstance(value, str) else f"{float(value):.4f}")

    lines = [
        "# Corrected Reproduction Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(render(row[key]) for key in selected) + " |")
    lines.extend(
        [
            "",
            "All overlap metrics are dataset-level micro scores from the corrected evaluator. HD excludes empty/empty slices.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/reproduction-sphere-v1")
    args = parser.parse_args()
    csv_path, markdown_path = aggregate_results(args.results_dir)
    print(f"Saved: {csv_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()
