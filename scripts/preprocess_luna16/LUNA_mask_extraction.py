from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def _match_series_file(file_list, seriesuid):
    seriesuid = str(seriesuid)
    for path in file_list:
        # ``path`` already has its .mhd suffix removed.  LUNA series UIDs
        # contain many dots, so Path(path).stem would incorrectly drop the
        # final UID component as if it were a file extension.
        if Path(path).name == seriesuid:
            return path
    return None


def _physical_to_continuous_index(center_xyz, origin_xyz, spacing_xyz, direction_xyz=None):
    center_xyz = np.asarray(center_xyz, dtype=np.float64)
    origin_xyz = np.asarray(origin_xyz, dtype=np.float64)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float64)
    direction = np.eye(3, dtype=np.float64) if direction_xyz is None else np.asarray(direction_xyz, dtype=np.float64).reshape(3, 3)
    return np.linalg.solve(direction, center_xyz - origin_xyz) / spacing_xyz


def _add_box_mask(mask, center_xyz, diameter_mm, origin_xyz, spacing_xyz, direction_xyz=None):
    """Create a cuboid approximation from LUNA16 center and diameter annotations."""
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float64)
    center_voxel_xyz = np.rint(
        _physical_to_continuous_index(center_xyz, origin_xyz, spacing_xyz, direction_xyz)
    ).astype(int)
    center_zyx = np.array([center_voxel_xyz[2], center_voxel_xyz[1], center_voxel_xyz[0]])
    radius_zyx = np.ceil((diameter_mm / np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])) / 2).astype(int)

    z0, y0, x0 = np.maximum(center_zyx - radius_zyx, 0)
    z1, y1, x1 = np.minimum(center_zyx + radius_zyx + 1, mask.shape)
    if z0 < z1 and y0 < y1 and x0 < x1:
        mask[z0:z1, y0:y1, x0:x1] = 1.0


def _add_sphere_mask(mask, center_xyz, diameter_mm, origin_xyz, spacing_xyz, direction_xyz=None):
    """Create a spherical pseudo-mask using physical millimetre distances."""
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float64)
    center_voxel_xyz = _physical_to_continuous_index(
        center_xyz, origin_xyz, spacing_xyz, direction_xyz
    )
    center_zyx = center_voxel_xyz[::-1]
    spacing_zyx = spacing_xyz[::-1]
    radius_mm = float(diameter_mm) / 2.0
    radius_voxels = np.ceil(radius_mm / spacing_zyx).astype(int)

    lower = np.maximum(np.floor(center_zyx).astype(int) - radius_voxels, 0)
    upper = np.minimum(np.ceil(center_zyx).astype(int) + radius_voxels + 1, mask.shape)
    z0, y0, x0 = lower
    z1, y1, x1 = upper
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return

    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    distance_sq = (
        ((zz - center_zyx[0]) * spacing_zyx[0]) ** 2
        + ((yy - center_zyx[1]) * spacing_zyx[1]) ** 2
        + ((xx - center_zyx[2]) * spacing_zyx[2]) ** 2
    )
    region = mask[z0:z1, y0:y1, x0:x1]
    region[distance_sq <= radius_mm**2] = 1.0


def create_masks(luna_root, output_root=None, annotations_csv=None, mask_shape="sphere"):
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
            # The UID itself contains dots. ``Path.with_suffix`` would replace
            # the last UID component, so append the real file extension.
            img_file = Path(f"{img_file_no_ext}.mhd")
            itk_img = sitk.ReadImage(str(img_file))
            img_array = sitk.GetArrayFromImage(itk_img)
            mask_array = np.zeros(img_array.shape, dtype=np.float32)

            origin = np.array(itk_img.GetOrigin())
            spacing = np.array(itk_img.GetSpacing())
            direction = np.array(itk_img.GetDirection())

            for _, row in mini_df.iterrows():
                center_xyz = [row["coordX"], row["coordY"], row["coordZ"]]
                add_mask = _add_sphere_mask if mask_shape == "sphere" else _add_box_mask
                add_mask(mask_array, center_xyz, row["diameter_mm"], origin, spacing, direction)

            mask_uint8 = np.clip(mask_array * 255, 0, 255).astype(np.uint8)
            mask_itk = sitk.GetImageFromArray(mask_uint8)
            mask_itk.SetSpacing(itk_img.GetSpacing())
            mask_itk.SetOrigin(itk_img.GetOrigin())
            mask_itk.SetDirection(itk_img.GetDirection())
            sitk.WriteImage(mask_itk, str(subset_output / f"{img_file.stem}_segmentation.mhd"))


def parse_args():
    parser = argparse.ArgumentParser(description="Create coarse LUNA16 nodule masks from annotations.csv.")
    parser.add_argument("--luna-root", default="data/luna16", help="Directory containing subset0..subset9 and annotations.csv.")
    parser.add_argument("--output-root", default=None, help="Output mask directory. Default: <luna-root>/mask.")
    parser.add_argument("--annotations-csv", default=None, help="Optional annotations.csv path.")
    parser.add_argument(
        "--mask-shape",
        choices=("sphere", "box"),
        default="sphere",
        help="Pseudo-mask geometry. Sphere is recommended; box reproduces the legacy project method.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_masks(args.luna_root, args.output_root, args.annotations_csv, args.mask_shape)
