from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def _match_series_file(file_list, seriesuid):
    for path in file_list:
        if seriesuid in Path(path).stem:
            return path
    return None


def _add_box_mask(mask, center_xyz, diameter_mm, origin_xyz, spacing_xyz):
    """Create a cuboid approximation from LUNA16 center and diameter annotations."""
    center_xyz = np.asarray(center_xyz, dtype=np.float64)
    origin_xyz = np.asarray(origin_xyz, dtype=np.float64)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float64)

    center_voxel_xyz = np.rint((center_xyz - origin_xyz) / spacing_xyz).astype(int)
    center_zyx = np.array([center_voxel_xyz[2], center_voxel_xyz[1], center_voxel_xyz[0]])
    radius_zyx = np.ceil((diameter_mm / np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])) / 2).astype(int)

    z0, y0, x0 = np.maximum(center_zyx - radius_zyx, 0)
    z1, y1, x1 = np.minimum(center_zyx + radius_zyx + 1, mask.shape)
    if z0 < z1 and y0 < y1 and x0 < x1:
        mask[z0:z1, y0:y1, x0:x1] = 1.0


def create_masks(luna_root, output_root=None, annotations_csv=None):
    luna_root = Path(luna_root)
    output_root = Path(output_root) if output_root else luna_root / "mask"
    annotations_csv = Path(annotations_csv) if annotations_csv else luna_root / "annotations.csv"

    if not annotations_csv.exists():
        raise FileNotFoundError(f"annotations.csv not found: {annotations_csv}")

    annotations = pd.read_csv(annotations_csv)
    output_root.mkdir(parents=True, exist_ok=True)

    for subset_index in range(10):
        subset_dir = luna_root / f"subset{subset_index}"
        subset_output = output_root / f"subset{subset_index}"
        subset_output.mkdir(parents=True, exist_ok=True)

        file_list = sorted(glob(str(subset_dir / "*.mhd")))
        if not file_list:
            print(f"Skipping subset{subset_index}: no .mhd files found")
            continue

        series_paths = [str(Path(path).with_suffix("")) for path in file_list]
        subset_annotations = annotations.copy()
        subset_annotations["file"] = subset_annotations["seriesuid"].map(
            lambda seriesuid: _match_series_file(series_paths, seriesuid)
        )
        subset_annotations = subset_annotations.dropna()

        for img_file_no_ext in tqdm(series_paths, desc=f"subset{subset_index}"):
            mini_df = subset_annotations[subset_annotations["file"] == img_file_no_ext]
            img_file = Path(img_file_no_ext).with_suffix(".mhd")
            itk_img = sitk.ReadImage(str(img_file))
            img_array = sitk.GetArrayFromImage(itk_img)
            mask_array = np.zeros(img_array.shape, dtype=np.float32)

            origin = np.array(itk_img.GetOrigin())
            spacing = np.array(itk_img.GetSpacing())

            for _, row in mini_df.iterrows():
                center_xyz = [row["coordX"], row["coordY"], row["coordZ"]]
                _add_box_mask(mask_array, center_xyz, row["diameter_mm"], origin, spacing)

            mask_uint8 = np.clip(mask_array * 255, 0, 255).astype(np.uint8)
            mask_itk = sitk.GetImageFromArray(mask_uint8)
            mask_itk.SetSpacing(itk_img.GetSpacing())
            mask_itk.SetOrigin(itk_img.GetOrigin())
            sitk.WriteImage(str(mask_itk), str(subset_output / f"{img_file.stem}_segmentation.mhd"))


def parse_args():
    parser = argparse.ArgumentParser(description="Create coarse LUNA16 nodule masks from annotations.csv.")
    parser.add_argument("--luna-root", default="data/luna16", help="Directory containing subset0..subset9 and annotations.csv.")
    parser.add_argument("--output-root", default=None, help="Output mask directory. Default: <luna-root>/mask.")
    parser.add_argument("--annotations-csv", default=None, help="Optional annotations.csv path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_masks(args.luna_root, args.output_root, args.annotations_csv)
