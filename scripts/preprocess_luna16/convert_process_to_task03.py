from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _assign_scan_splits(rows, seed=42, validation_fraction=0.1, test_fraction=0.1):
    """Assign deterministic scan-level splits, stratified by official subset."""
    groups = {}
    for row in rows:
        groups.setdefault(row.get("subset") or "unknown", []).append(row)

    rng = random.Random(seed)
    for group_name in sorted(groups, key=str):
        group = sorted(groups[group_name], key=lambda row: int(row["source_series_index"]))
        rng.shuffle(group)
        if len(group) < 3:
            test_count = 0
            validation_count = 0
        else:
            test_count = max(1, round(len(group) * test_fraction))
            validation_count = max(1, round(len(group) * validation_fraction))
            if test_count + validation_count >= len(group):
                validation_count = max(0, len(group) - test_count - 1)

        for index, row in enumerate(group):
            if index < test_count:
                row["split"] = "test"
            elif index < test_count + validation_count:
                row["split"] = "validation"
            else:
                row["split"] = "train"
    return rows


def _slice_number(path):
    """Sort unpadded slice names numerically (0.jpg, 1.jpg, 2.jpg, ...)."""
    path = Path(path)
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def _media_by_stem(directory):
    result = {}
    for path in Path(directory).iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate slice stem under {directory}: {path.stem}")
        result[path.stem] = path
    return result


def convert_process_to_task03(process_root, output_root):
    process_root = Path(process_root)
    output_root = Path(output_root)
    process_image_path = process_root / "image"
    process_mask_path = process_root / "mask"

    if not process_image_path.exists() or not process_mask_path.exists():
        raise FileNotFoundError(f"Missing process/image or process/mask under {process_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    series_dirs = sorted([path for path in process_image_path.iterdir() if path.is_dir()], key=lambda p: int(p.name))
    source_manifest_path = process_root / "series_manifest.csv"
    source_manifest = {}
    if source_manifest_path.exists():
        with source_manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            source_manifest = {row["series_index"]: row for row in csv.DictReader(manifest_file)}

    nodule_counter = 0
    output_manifest = []
    for series_dir in series_dirs:
        src_image_dir = process_image_path / series_dir.name
        src_mask_dir = process_mask_path / series_dir.name
        if not src_mask_dir.exists():
            print(f"Skipping {series_dir.name}: missing mask directory")
            continue

        image_map = _media_by_stem(src_image_dir)
        mask_map = _media_by_stem(src_mask_dir)
        if not image_map or not mask_map:
            print(f"Skipping {series_dir.name}: no image or mask files")
            continue

        if image_map.keys() != mask_map.keys():
            raise ValueError(
                f"Image/mask filename mismatch in series {series_dir.name}; "
                "refusing to create shifted training pairs."
            )
        stems = sorted(image_map, key=lambda stem: _slice_number(stem))

        nodule_dir = output_root / f"nodule_{nodule_counter}"
        target_image_dir = nodule_dir / "images"
        target_mask_dir = nodule_dir / "masks" / "nodule"
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)

        for idx, stem in enumerate(stems):
            img_file = image_map[stem]
            mask_file = mask_map[stem]
            new_stem = f"{nodule_counter}_{idx:04d}"
            shutil.copy2(img_file, target_image_dir / f"{new_stem}{img_file.suffix.lower()}")
            shutil.copy2(mask_file, target_mask_dir / f"{new_stem}{mask_file.suffix.lower()}")

        source = source_manifest.get(series_dir.name, {})
        output_manifest.append(
            {
                "case_name": f"nodule_{nodule_counter}",
                "source_series_index": series_dir.name,
                "seriesuid": source.get("seriesuid", ""),
                "subset": source.get("subset", ""),
                "slice_count": len(stems),
            }
        )
        nodule_counter += 1
        if nodule_counter % 10 == 0:
            print(f"Converted {nodule_counter} nodule folders")

    _assign_scan_splits(output_manifest)
    manifest_path = output_root / "case_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("case_name", "source_series_index", "seriesuid", "subset", "slice_count", "split"),
        )
        writer.writeheader()
        writer.writerows(output_manifest)

    print(f"Converted {nodule_counter} folders into {output_root}")
    print(f"Saved case provenance: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert luna16/process folders into Task03_lung format.")
    parser.add_argument("--process-root", default="data/luna16/process", help="Directory containing image/ and mask/.")
    parser.add_argument("--output-root", default="data/Task03_lung", help="Output Task03_lung directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_process_to_task03(args.process_root, args.output_root)
