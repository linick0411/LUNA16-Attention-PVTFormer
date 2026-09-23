"""Publish compact experiment evidence without committing datasets or weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


MODELS = ("baseline", "concat_25d", "attention_gate", "voxel_attention", "coordinate_attention")
HISTORY_PATTERNS = {
    "baseline": "history_pvtformer_baseline_2d_*.csv",
    "concat_25d": "history_concat_25d_control_*.csv",
    "attention_gate": "history_attention_gate_*.csv",
    "voxel_attention": "history_voxel_attention_*.csv",
    "coordinate_attention": "history_coordinate_attention_*.csv",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_history(path):
    with Path(path).open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        return list(reader), reader.fieldnames


def publish_results(results_root, logs_root, checkpoints_root, output_root, validation_path=None):
    results_root = Path(results_root)
    logs_root = Path(logs_root)
    checkpoints_root = Path(checkpoints_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows, summary_fields = _read_history(results_root / "summary.csv")
    _write_csv(output_root / "summary.csv", summary_rows, summary_fields)
    _write_json(output_root / "aggregation-audit.json", _read_json(results_root / "aggregation-audit.json"))

    validation_path = Path(validation_path) if validation_path else results_root / "data_validation.json"
    validation = _read_json(validation_path)
    validation["root"] = "data/Task03_lung_sphere-v1"
    _write_json(output_root / "data-validation.json", validation)

    metrics_root = output_root / "metrics"
    for model in MODELS:
        metrics = _read_json(results_root / model / "metrics.json")
        checkpoint = Path(metrics.pop("checkpoint", f"checkpoint_{model}.pth")).name
        metrics["checkpoint_file"] = f"checkpoint_{model}.pth" if model != "coordinate_attention" else checkpoint
        _write_json(metrics_root / f"{model}.json", metrics)

    training_root = output_root / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    run_root = training_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    training_summary = []
    for model, pattern in HISTORY_PATTERNS.items():
        history_paths = sorted(logs_root.glob(pattern))
        if not history_paths:
            raise RuntimeError(f"No training history found for {model}: {logs_root / pattern}")
        candidates = []
        for history_path in history_paths:
            rows, fieldnames = _read_history(history_path)
            if not rows:
                raise RuntimeError(f"Training history is empty: {history_path}")
            _write_csv(run_root / history_path.name, rows, fieldnames)
            best = max(rows, key=lambda row: float(row["valid_f1"]))
            candidates.append((float(best["valid_f1"]), history_path, rows, fieldnames, best))
        _, selected_path, rows, fieldnames, best = max(candidates, key=lambda item: item[0])
        _write_csv(training_root / f"{model}.csv", rows, fieldnames)
        final = rows[-1]
        training_summary.append(
            {
                "model": model,
                "selected_history": selected_path.name,
                "available_runs": len(history_paths),
                "epochs_recorded": len(rows),
                "last_epoch": int(final["epoch"]),
                "best_valid_f1": float(best["valid_f1"]),
                "best_valid_f1_epoch": int(best["epoch"]),
                "final_train_loss": float(final["train_loss"]),
                "final_valid_loss": float(final["valid_loss"]),
            }
        )
    _write_csv(output_root / "training-summary.csv", training_summary, training_summary[0].keys())

    checkpoint_manifest = {
        "storage": "Weights are intentionally excluded from Git because each file is larger than GitHub's 100 MB per-file limit.",
        "files": [],
    }
    for model in MODELS:
        checkpoint_path = checkpoints_root / f"checkpoint_{model}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_manifest["files"].append(
            {
                "model": model,
                "file": checkpoint_path.name,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            }
        )
    _write_json(output_root / "checkpoint-manifest.json", checkpoint_manifest)

    (output_root / "README.md").write_text(
        """# Published sphere-v1 evidence

This directory contains the compact evidence needed to review the corrected
five-model reproduction without committing LUNA16 data, prediction masks, raw
logs, or large model weights.

- `summary.csv`: fixed-contract test comparison for all five models.
- `metrics/*.json`: per-model test metrics and runtime metadata.
- `training/*.csv`: the run containing each model's highest recorded validation F1.
- `training/runs/*.csv`: every retained run, including the Coordinate Attention warm restart.
- `training-summary.csv`: best validation F1 and final losses.
- `data-validation.json`: pairing, binary-mask, split, and z-order provenance audit.
- `aggregation-audit.json`: micro versus per-slice F1 diagnostic.
- `checkpoint-manifest.json`: exact checkpoint filenames, sizes, and SHA-256 hashes.

The targets are spherical pseudo-masks generated from LUNA16 center coordinates
and diameter annotations. They are not expert-drawn clinical contours. The
checkpoint files exceed GitHub's normal 100 MB file limit and are therefore
identified by hash rather than stored in this repository.
""",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/reproduction-sphere-v1")
    parser.add_argument("--logs-root", default="logs/reproduction-sphere-v1")
    parser.add_argument("--checkpoints-root", default="checkpoints/reproduction-sphere-v1")
    parser.add_argument("--output-root", default="results/published/sphere-v1")
    parser.add_argument("--validation", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    publish_results(
        args.results_root,
        args.logs_root,
        args.checkpoints_root,
        args.output_root,
        validation_path=args.validation,
    )
