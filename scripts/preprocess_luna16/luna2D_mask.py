from __future__ import annotations

import argparse
import csv
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


def get_mask_depth_range(mask_volume):
    foreground_slices = np.flatnonzero(np.any(mask_volume > 0, axis=(1, 2)))
    if foreground_slices.size == 0:
        return None, None

    # Return a half-open range so a one-slice nodule remains valid and the
    # final foreground slice is included when slicing the volume.
    return int(foreground_slices[0]), int(foreground_slices[-1] + 1)


def resize_image_itk(itk_image, new_spacing, resample_method=sitk.sitkNearestNeighbor):
    new_spacing = np.array(new_spacing, dtype=np.float64)
    origin_spacing = np.array(itk_image.GetSpacing(), dtype=np.float64)
    origin_size = np.array(itk_image.GetSize(), dtype=np.int64)
    new_size = np.maximum(np.rint(origin_size * (origin_spacing / new_spacing)).astype(int), 1)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(itk_image)
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize([int(item) for item in new_size])
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)
    return resampler.Execute(itk_image)


def load_ct_with_truncation(filename, upper=600, lower=-1000):
    if upper <= lower:
        raise ValueError(f"upper must be greater than lower, got lower={lower}, upper={upper}")
    src = sitk.Cast(sitk.ReadImage(str(filename)), sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(src)
    array = np.clip(array, lower, upper)
    array = (array - lower) / float(upper - lower) * 255.0

    truncated = sitk.GetImageFromArray(array)
    truncated.SetSpacing(src.GetSpacing())
    truncated.SetOrigin(src.GetOrigin())
    truncated.SetDirection(src.GetDirection())
    return sitk.Cast(truncated, sitk.sitkFloat32)


def process_original_train_data(
    luna_root,
    output_root=None,
    mask_root=None,
    expand_slices=13,
    upper=600,
    lower=-1000,
    image_format="png",
    jpg_quality=95,
):
    luna_root = Path(luna_root)
    output_root = Path(output_root) if output_root else luna_root / "process"
    mask_root = Path(mask_root) if mask_root else luna_root / "mask"
    train_image_root = output_root / "image"
    train_mask_root = output_root / "mask"
    train_image_root.mkdir(parents=True, exist_ok=True)
    train_mask_root.mkdir(parents=True, exist_ok=True)

    series_index = 0
    manifest_rows = []
    for subset_index in range(10):
        subset_dir = luna_root / f"subset{subset_index}"
        subset_mask_dir = mask_root / f"subset{subset_index}"
        file_list = sorted(glob(str(subset_dir / "*.mhd")))

        for ct_file in file_list:
            ct_file = Path(ct_file)
            mask_file = subset_mask_dir / f"{ct_file.stem}_segmentation.mhd"
            if not mask_file.exists():
                print(f"Skipping {ct_file.name}: missing mask {mask_file}")
                continue

            src = load_ct_with_truncation(ct_file, upper=upper, lower=lower)
            seg = sitk.ReadImage(str(mask_file), sitk.sitkUInt8)

            if seg.GetSpacing()[-1] > 1.0:
                target_spacing = (seg.GetSpacing()[0], seg.GetSpacing()[1], 1.0)
                seg = resize_image_itk(seg, target_spacing, sitk.sitkNearestNeighbor)
                src = resize_image_itk(src, target_spacing, sitk.sitkLinear)

            seg_array = sitk.GetArrayFromImage(seg)
            src_array = sitk.GetArrayFromImage(src)
            mask_volume = (seg_array > 0).astype(np.uint8) * 255

            start, end = get_mask_depth_range(mask_volume)
            if start is None:
                continue

            start = max(start - expand_slices, 0)
            end = min(end + expand_slices, mask_volume.shape[0])
            src_array = np.clip(src_array[start:end], 0, 255).astype(np.uint8)
            mask_volume = mask_volume[start:end]

            image_dir = train_image_root / str(series_index)
            mask_dir = train_mask_root / str(series_index)
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            for z in range(mask_volume.shape[0]):
                image_path = image_dir / f"{z}.{image_format}"
                image_options = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality] if image_format == "jpg" else []
                cv2.imwrite(str(image_path), src_array[z], image_options)
                cv2.imwrite(str(mask_dir / f"{z}.png"), mask_volume[z])

            manifest_rows.append(
                {
                    "series_index": series_index,
                    "seriesuid": ct_file.stem,
                    "subset": subset_index,
                    "source_mhd": str(ct_file),
                    "first_kept_slice": start,
                    "last_kept_slice_exclusive": end,
                    "slice_count": int(mask_volume.shape[0]),
                    "positive_slice_count": int(np.count_nonzero(np.any(mask_volume > 0, axis=(1, 2)))),
                }
            )
            series_index += 1

    manifest_path = output_root / "series_manifest.csv"
    fieldnames = [
        "series_index",
        "seriesuid",
        "subset",
        "source_mhd",
        "first_kept_slice",
        "last_kept_slice_exclusive",
        "slice_count",
        "positive_slice_count",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Processed {series_index} series into {output_root}")
    print(f"Saved series provenance: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LUNA16 3D volumes and masks into 2D process/image and process/mask folders.")
    parser.add_argument("--luna-root", default="data/luna16", help="Directory containing subset0..subset9 and mask/subset0..subset9.")
    parser.add_argument("--output-root", default=None, help="Output process directory. Default: <luna-root>/process.")
    parser.add_argument("--mask-root", default=None, help="Generated 3D mask root. Default: <luna-root>/mask.")
    parser.add_argument("--expand-slices", type=int, default=13, help="Extra slices kept before and after mask depth range.")
    parser.add_argument("--upper", type=int, default=600, help="HU upper clipping value.")
    parser.add_argument("--lower", type=int, default=-1000, help="HU lower clipping value.")
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png", help="CT slice format. PNG avoids JPEG artifacts.")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality when --image-format jpg is selected.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_original_train_data(
        args.luna_root,
        args.output_root,
        args.mask_root,
        args.expand_slices,
        args.upper,
        args.lower,
        args.image_format,
        args.jpg_quality,
    )
