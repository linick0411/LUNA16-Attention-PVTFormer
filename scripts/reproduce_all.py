"""Prepare the corrected LUNA16 dataset, train every model, and save evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.preprocess_luna16.LUNA_mask_extraction import create_masks
from scripts.preprocess_luna16.convert_process_to_task03 import convert_process_to_task03
from scripts.preprocess_luna16.luna2D_mask import process_original_train_data
from scripts.aggregate_results import aggregate_results
from scripts.plot_training_curves import plot_history
from scripts.validate_task03 import validate_task03


EXPECTED_SCAN_COUNT = 888
PIPELINE_VERSION = "sphere-v1"


def verify_luna_ready(luna_root, expected_scans=EXPECTED_SCAN_COUNT):
    luna_root = Path(luna_root)
    annotations = luna_root / "annotations.csv"
    if not annotations.exists():
        raise RuntimeError(f"Missing official annotations file: {annotations}")

    missing_subsets = [index for index in range(10) if not (luna_root / f"subset{index}").is_dir()]
    if missing_subsets:
        raise RuntimeError(f"Missing extracted LUNA16 subsets: {missing_subsets}")

    mhd_files = sorted(luna_root.glob("subset*/*.mhd"))
    raw_files = sorted(luna_root.glob("subset*/*.raw"))
    if len(mhd_files) != expected_scans or len(raw_files) != expected_scans:
        raise RuntimeError(
            f"Expected {expected_scans} CT scans, found {len(mhd_files)} MHD and {len(raw_files)} RAW files"
        )

    annotated_scans = pd.read_csv(annotations, usecols=["seriesuid"])["seriesuid"].astype(str).nunique()
    return {
        "scan_count": len(mhd_files),
        "raw_count": len(raw_files),
        "annotated_scan_count": int(annotated_scans),
    }


def _write_status(path, stage, **details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "stage": stage,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **details,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _assert_new_staging(staging, final):
    if staging.exists():
        raise RuntimeError(f"Incomplete staging directory already exists and was preserved: {staging}")
    if final.exists():
        return False
    return True


def prepare_dataset(repo_root, luna_root, results_root, status_path):
    repo_root = Path(repo_root)
    luna_root = Path(luna_root)
    results_root = Path(results_root)
    readiness = verify_luna_ready(luna_root)
    _write_status(status_path, "official_data_verified", **readiness)

    mask_final = luna_root / f"mask_{PIPELINE_VERSION}"
    mask_staging = luna_root / f".mask_{PIPELINE_VERSION}.preparing"
    if _assert_new_staging(mask_staging, mask_final):
        _write_status(status_path, "creating_spherical_pseudo_masks", **readiness)
        create_masks(luna_root, output_root=mask_staging, mask_shape="sphere")
        if len(list(mask_staging.glob("subset*/*_segmentation.mhd"))) != readiness["scan_count"]:
            raise RuntimeError("Generated mask count does not match the official scan count")
        mask_staging.rename(mask_final)

    process_final = luna_root / f"process_{PIPELINE_VERSION}"
    process_staging = luna_root / f".process_{PIPELINE_VERSION}.preparing"
    if _assert_new_staging(process_staging, process_final):
        _write_status(status_path, "extracting_lossless_slices", **readiness)
        process_original_train_data(
            luna_root,
            output_root=process_staging,
            mask_root=mask_final,
            image_format="png",
        )
        image_folders = [path for path in (process_staging / "image").iterdir() if path.is_dir()]
        mask_folders = [path for path in (process_staging / "mask").iterdir() if path.is_dir()]
        if len(image_folders) != readiness["annotated_scan_count"] or len(mask_folders) != len(image_folders):
            raise RuntimeError(
                f"Expected {readiness['annotated_scan_count']} annotated scan folders, "
                f"found images={len(image_folders)}, masks={len(mask_folders)}"
            )
        process_staging.rename(process_final)

    task_final = repo_root / "data" / f"Task03_lung_{PIPELINE_VERSION}"
    task_staging = repo_root / "data" / f".Task03_lung_{PIPELINE_VERSION}.preparing"
    if _assert_new_staging(task_staging, task_final):
        _write_status(status_path, "building_task03_dataset", **readiness)
        convert_process_to_task03(process_final, task_staging)
        task_staging.rename(task_final)

    _write_status(status_path, "validating_task03_dataset", **readiness)
    validation = validate_task03(task_final, check_pixels=True)
    validation_path = results_root / "data_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    _write_status(status_path, "dataset_ready", validation_report=str(validation_path), **validation)
    return task_final, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luna-root", default="data/luna16")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed checkpoints and metrics, and continue from the first missing artifact.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    luna_root = (repo_root / args.luna_root).resolve() if not Path(args.luna_root).is_absolute() else Path(args.luna_root)
    results_root = repo_root / "results" / f"reproduction-{PIPELINE_VERSION}"
    logs_root = repo_root / "logs" / f"reproduction-{PIPELINE_VERSION}"
    checkpoints_root = repo_root / "checkpoints" / f"reproduction-{PIPELINE_VERSION}"
    status_path = logs_root / "pipeline-status.json"

    try:
        task_root, validation = prepare_dataset(repo_root, luna_root, results_root, status_path)
        if args.prepare_only:
            return 0

        _write_status(status_path, "training_and_evaluating", **validation)
        environment = os.environ.copy()
        environment.update(
            {
                "LUNA16_TASK_DIR": str(task_root),
                "CHECKPOINT_DIR": str(checkpoints_root),
                "LOG_DIR": str(logs_root),
                "RESULTS_DIR": str(results_root),
                "DATASET_VERSION": PIPELINE_VERSION,
            }
        )
        experiment_command = [
            sys.executable,
            "scripts/run_experiments.py",
            "--stage",
            "all",
            "--models",
            "all",
        ]
        if args.resume:
            experiment_command.append("--resume")
        subprocess.run(
            experiment_command,
            cwd=repo_root,
            env=environment,
            check=True,
        )
        for history_path in sorted(logs_root.glob("history_*.csv")):
            plot_history(history_path)
        summary_csv, summary_markdown = aggregate_results(results_root)
        _write_status(
            status_path,
            "complete",
            checkpoint_dir=str(checkpoints_root),
            results_dir=str(results_root),
            summary_csv=str(summary_csv),
            summary_markdown=str(summary_markdown),
            **validation,
        )
        return 0
    except Exception as exc:
        _write_status(status_path, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
