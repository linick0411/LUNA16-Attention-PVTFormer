from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def normalize_ct_slice(slice_array, window_center=-600, window_width=1500):
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    slice_array = np.clip(slice_array, min_val, max_val)
    return ((slice_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)


def extract_slices_from_volume(ct_path, mask_path, output_dir, nodule_id, min_mask_area=50):
    ct_img = sitk.ReadImage(str(ct_path))
    mask_img = sitk.ReadImage(str(mask_path))
    ct_array = sitk.GetArrayFromImage(ct_img)
    mask_array = sitk.GetArrayFromImage(mask_img)

    if ct_array.shape != mask_array.shape:
        raise ValueError(f"CT/mask shape mismatch: {ct_array.shape} vs {mask_array.shape}")

    nodule_dir = Path(output_dir) / f"nodule_{nodule_id}"
    image_dir = nodule_dir / "images"
    mask_dir = nodule_dir / "masks" / "nodule"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    slice_count = 0
    for z_idx in range(ct_array.shape[0]):
        mask_slice = mask_array[z_idx]
        if np.sum(mask_slice > 0) < min_mask_area:
            continue

        ct_normalized = normalize_ct_slice(ct_array[z_idx])
        mask_uint8 = ((mask_slice > 0).astype(np.uint8) * 255)
        filename = f"{nodule_id}_{slice_count:04d}.jpg"
        cv2.imwrite(str(image_dir / filename), ct_normalized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(mask_dir / filename), mask_uint8)
        slice_count += 1

    return slice_count


def convert_luna_to_jpg(luna_root, output_root, min_mask_area=50):
    luna_root = Path(luna_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    nodule_counter = 0
    for subset_idx in range(10):
        ct_subset_path = luna_root / f"subset{subset_idx}"
        mask_subset_path = luna_root / "mask" / f"subset{subset_idx}"
        if not ct_subset_path.exists() or not mask_subset_path.exists():
            print(f"Skipping subset{subset_idx}: missing CT or mask directory")
            continue

        ct_files = sorted(glob(str(ct_subset_path / "*.mhd")))
        for ct_file in tqdm(ct_files, desc=f"subset{subset_idx}"):
            ct_file = Path(ct_file)
            mask_file = mask_subset_path / f"{ct_file.stem}_segmentation.mhd"
            if not mask_file.exists():
                print(f"Skipping {ct_file.name}: missing mask {mask_file}")
                continue

            slice_count = extract_slices_from_volume(ct_file, mask_file, output_root, nodule_counter, min_mask_area)
            if slice_count > 0:
                print(f"nodule_{nodule_counter}: {slice_count} slices")
                nodule_counter += 1

    print(f"Converted {nodule_counter} nodules into {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Directly convert LUNA16 CT volumes and generated masks into Task03_lung JPG folders.")
    parser.add_argument("--luna-root", default="data/luna16", help="Directory containing subset0..subset9 and mask/subset0..subset9.")
    parser.add_argument("--output-root", default="data/Task03_lung", help="Output Task03_lung directory.")
    parser.add_argument("--min-mask-area", type=int, default=50, help="Minimum foreground pixels required to keep a slice.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_luna_to_jpg(args.luna_root, args.output_root, args.min_mask_area)
