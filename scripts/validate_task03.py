"""Validate Task03_lung pairing, split coverage, and optional pixel contents."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import random
from pathlib import Path

import cv2
import numpy as np

from luna16_data import _media_by_stem, _nodule_index, _slice_index


def _numeric_source_stems(media):
    try:
        return sorted(media, key=lambda stem: int(stem))
    except ValueError as exc:
        raise RuntimeError("Process slice names must be numeric (0.png, 1.png, ...)") from exc


def validate_task03(root, check_pixels=False, process_root=None, provenance_samples=3):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Task03_lung directory not found: {root}")

    folders = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("nodule_")],
        key=_nodule_index,
    )
    if not folders:
        raise RuntimeError(f"No nodule_* folders found under {root}")

    manifest_path = root / "case_manifest.csv"
    manifest_rows = None
    manifest_by_case = {}
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            manifest_rows = list(csv.DictReader(manifest_file))
        manifest_names = [row["case_name"] for row in manifest_rows]
        folder_names = [folder.name for folder in folders]
        if manifest_names != folder_names:
            raise RuntimeError("case_manifest.csv does not exactly match the Task03 folder order")
        source_uids = [row["seriesuid"] for row in manifest_rows if row["seriesuid"]]
        if source_uids and len(set(source_uids)) != len(source_uids):
            raise RuntimeError("case_manifest.csv contains duplicate source scan UIDs")
        if any(row.get("split") not in {"train", "validation", "test"} for row in manifest_rows):
            raise RuntimeError("case_manifest.csv contains a missing or invalid split")
        manifest_by_case = {row["case_name"]: row for row in manifest_rows}

    process_root = Path(process_root) if process_root else None
    process_manifest_by_index = {}
    sampled_cases = set()
    if process_root is not None:
        if manifest_rows is None:
            raise RuntimeError("Provenance validation requires case_manifest.csv")
        process_manifest_path = process_root / "series_manifest.csv"
        if not process_manifest_path.exists():
            raise RuntimeError(f"Missing process manifest: {process_manifest_path}")
        with process_manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            process_rows = list(csv.DictReader(manifest_file))
        process_manifest_by_index = {row["series_index"]: row for row in process_rows}
        if len(process_manifest_by_index) != len(process_rows):
            raise RuntimeError("series_manifest.csv contains duplicate series_index values")
        sample_count = min(max(int(provenance_samples), 0), len(manifest_rows))
        sampled_cases = set(random.Random(42).sample([row["case_name"] for row in manifest_rows], sample_count))

    report = {
        "root": str(root.resolve()),
        "folder_count": len(folders),
        "image_count": 0,
        "mask_count": 0,
        "empty_mask_count": 0,
        "non_binary_mask_count": 0,
        "splits": {"train": 0, "validation": 0, "test": 0},
        "check_pixels": bool(check_pixels),
    }
    provenance_case_count = 0
    provenance_slice_count = 0
    content_sample_slice_count = 0

    for folder in folders:
        image_dir = folder / "images"
        mask_dir = folder / "masks" / "nodule"
        if not image_dir.exists() or not mask_dir.exists():
            raise RuntimeError(f"Missing images or masks/nodule directory: {folder}")

        images = _media_by_stem(image_dir)
        masks = _media_by_stem(mask_dir)
        if not images:
            raise RuntimeError(f"No image slices found: {image_dir}")
        if images.keys() != masks.keys():
            missing_masks = sorted(images.keys() - masks.keys())[:5]
            missing_images = sorted(masks.keys() - images.keys())[:5]
            raise RuntimeError(
                f"Image/mask mismatch under {folder}; missing masks={missing_masks}, "
                f"missing images={missing_images}"
            )

        ordered_stems = sorted(images, key=_slice_index)
        case_index = _nodule_index(folder)[1]
        if not isinstance(case_index, int):
            raise RuntimeError(f"Cannot determine numeric case index: {folder.name}")
        expected_stems = [f"{case_index}_{index:04d}" for index in range(len(ordered_stems))]
        if ordered_stems != expected_stems:
            raise RuntimeError(
                f"Non-contiguous Task03 slice order under {folder}; "
                f"expected {expected_stems[:3]}..., got {ordered_stems[:3]}..."
            )

        report["image_count"] += len(images)
        report["mask_count"] += len(masks)
        if manifest_by_case:
            split = manifest_by_case[folder.name]["split"]
        else:
            index = _nodule_index(folder)[1]
            if isinstance(index, int) and index < 30:
                split = "test"
            elif isinstance(index, int) and index < 60:
                split = "validation"
            else:
                split = "train"
        report["splits"][split] += len(images)

        if process_root is not None:
            case_row = manifest_by_case[folder.name]
            source_index = case_row["source_series_index"]
            source_row = process_manifest_by_index.get(source_index)
            if source_row is None:
                raise RuntimeError(f"Missing source series {source_index} for {folder.name}")
            if source_row.get("seriesuid") != case_row.get("seriesuid"):
                raise RuntimeError(f"Series UID mismatch for {folder.name}")
            source_count = int(source_row["slice_count"])
            first_slice = int(source_row["first_kept_slice"])
            last_slice = int(source_row["last_kept_slice_exclusive"])
            if last_slice - first_slice != source_count:
                raise RuntimeError(f"Non-contiguous original z range for {folder.name}")
            if source_count != len(ordered_stems) or source_count != int(case_row["slice_count"]):
                raise RuntimeError(f"Slice count mismatch for {folder.name}")

            source_images = _media_by_stem(process_root / "image" / source_index)
            source_masks = _media_by_stem(process_root / "mask" / source_index)
            if source_images.keys() != source_masks.keys():
                raise RuntimeError(f"Source image/mask mismatch for series {source_index}")
            source_stems = _numeric_source_stems(source_images)
            expected_source_stems = [str(index) for index in range(source_count)]
            if source_stems != expected_source_stems:
                raise RuntimeError(f"Non-contiguous source slice order for series {source_index}")

            if folder.name in sampled_cases:
                for index, source_stem in enumerate(source_stems):
                    target_stem = expected_stems[index]
                    if not filecmp.cmp(source_images[source_stem], images[target_stem], shallow=False):
                        raise RuntimeError(f"Source/task image content mismatch for {folder.name}/{target_stem}")
                    if not filecmp.cmp(source_masks[source_stem], masks[target_stem], shallow=False):
                        raise RuntimeError(f"Source/task mask content mismatch for {folder.name}/{target_stem}")
                    content_sample_slice_count += 1
            provenance_case_count += 1
            provenance_slice_count += source_count

        if check_pixels:
            for stem, path in masks.items():
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Cannot read mask: {path}")
                unique = np.unique(mask)
                if not np.any(mask > 127):
                    report["empty_mask_count"] += 1
                if not np.all(np.isin(unique, (0, 255))):
                    report["non_binary_mask_count"] += 1

                image = cv2.imread(images[stem], cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise RuntimeError(f"Cannot read image: {images[stem]}")
                if image.shape != mask.shape:
                    raise RuntimeError(
                        f"Image/mask shape mismatch for {folder.name}/{stem}: "
                        f"{image.shape} vs {mask.shape}"
                    )

    if report["splits"]["test"] == 0 or report["splits"]["validation"] == 0 or report["splits"]["train"] == 0:
        raise RuntimeError(f"One or more dataset splits are empty: {report['splits']}")

    if manifest_rows is not None:
        report["manifest_cases"] = len(manifest_rows)
        report["manifest_source_uids"] = len(source_uids)
    else:
        report["manifest_cases"] = None
        report["manifest_source_uids"] = None
    if process_root is not None:
        report["slice_order_validation"] = {
            "status": "passed",
            "cases_checked": provenance_case_count,
            "slices_checked": provenance_slice_count,
            "content_sample_cases": sorted(sampled_cases, key=_nodule_index),
            "content_sample_slices": content_sample_slice_count,
            "original_z_index_rule": "first_kept_slice + exported_slice_index",
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/Task03_lung")
    parser.add_argument("--check-pixels", action="store_true")
    parser.add_argument("--process-root", default=None, help="Optional process directory used to verify source-to-Task03 slice order.")
    parser.add_argument("--provenance-samples", type=int, default=3, help="Number of deterministic cases checked byte-for-byte.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = validate_task03(
        args.data_dir,
        check_pixels=args.check_pixels,
        process_root=args.process_root,
        provenance_samples=args.provenance_samples,
    )
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
