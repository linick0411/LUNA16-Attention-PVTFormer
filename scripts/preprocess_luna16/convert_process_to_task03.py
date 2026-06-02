from __future__ import annotations

import argparse
import shutil
from glob import glob
from pathlib import Path


def convert_process_to_task03(process_root, output_root):
    process_root = Path(process_root)
    output_root = Path(output_root)
    process_image_path = process_root / "image"
    process_mask_path = process_root / "mask"

    if not process_image_path.exists() or not process_mask_path.exists():
        raise FileNotFoundError(f"Missing process/image or process/mask under {process_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    series_dirs = sorted([path for path in process_image_path.iterdir() if path.is_dir()], key=lambda p: int(p.name))

    nodule_counter = 0
    for series_dir in series_dirs:
        src_image_dir = process_image_path / series_dir.name
        src_mask_dir = process_mask_path / series_dir.name
        if not src_mask_dir.exists():
            print(f"Skipping {series_dir.name}: missing mask directory")
            continue

        image_files = sorted(glob(str(src_image_dir / "*.jpg")))
        mask_files = sorted(glob(str(src_mask_dir / "*.jpg")))
        if not image_files or not mask_files:
            print(f"Skipping {series_dir.name}: no jpg files")
            continue

        min_count = min(len(image_files), len(mask_files))
        if len(image_files) != len(mask_files):
            print(f"Warning: {series_dir.name} image/mask count mismatch; using {min_count} pairs")

        nodule_dir = output_root / f"nodule_{nodule_counter}"
        target_image_dir = nodule_dir / "images"
        target_mask_dir = nodule_dir / "masks" / "nodule"
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_file, mask_file) in enumerate(zip(image_files[:min_count], mask_files[:min_count])):
            new_name = f"{nodule_counter}_{idx:04d}.jpg"
            shutil.copy2(img_file, target_image_dir / new_name)
            shutil.copy2(mask_file, target_mask_dir / new_name)

        nodule_counter += 1
        if nodule_counter % 10 == 0:
            print(f"Converted {nodule_counter} nodule folders")

    print(f"Converted {nodule_counter} folders into {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert luna16/process folders into Task03_lung format.")
    parser.add_argument("--process-root", default="data/luna16/process", help="Directory containing image/ and mask/.")
    parser.add_argument("--output-root", default="data/Task03_lung", help="Output Task03_lung directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_process_to_task03(args.process_root, args.output_root)
