from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


def get_mask_depth_range(mask_volume):
    start = 0
    end = 0
    found = False
    for z in range(mask_volume.shape[0]):
        if np.max(mask_volume[z]) > 0:
            if not found:
                start = z
                found = True
            end = z
    return start, end


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
    src = sitk.Cast(sitk.ReadImage(str(filename)), sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(src)
    array = np.clip(array, lower, upper)

    truncated = sitk.GetImageFromArray(array)
    truncated.SetSpacing(src.GetSpacing())
    truncated.SetOrigin(src.GetOrigin())

    rescale = sitk.RescaleIntensityImageFilter()
    rescale.SetOutputMaximum(255)
    rescale.SetOutputMinimum(0)
    return rescale.Execute(sitk.Cast(truncated, sitk.sitkFloat32))


def process_original_train_data(luna_root, output_root=None, expand_slices=13, upper=600, lower=-1000):
    luna_root = Path(luna_root)
    output_root = Path(output_root) if output_root else luna_root / "process"
    train_image_root = output_root / "image"
    train_mask_root = output_root / "mask"
    train_image_root.mkdir(parents=True, exist_ok=True)
    train_mask_root.mkdir(parents=True, exist_ok=True)

    series_index = 0
    for subset_index in range(10):
        subset_dir = luna_root / f"subset{subset_index}"
        subset_mask_dir = luna_root / "mask" / f"subset{subset_index}"
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
            if start == end:
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
                cv2.imwrite(str(image_dir / f"{z}.jpg"), src_array[z], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cv2.imwrite(str(mask_dir / f"{z}.jpg"), mask_volume[z])

            series_index += 1

    print(f"Processed {series_index} series into {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LUNA16 3D volumes and masks into 2D process/image and process/mask folders.")
    parser.add_argument("--luna-root", default="data/luna16", help="Directory containing subset0..subset9 and mask/subset0..subset9.")
    parser.add_argument("--output-root", default=None, help="Output process directory. Default: <luna-root>/process.")
    parser.add_argument("--expand-slices", type=int, default=13, help="Extra slices kept before and after mask depth range.")
    parser.add_argument("--upper", type=int, default=600, help="HU upper clipping value.")
    parser.add_argument("--lower", type=int, default=-1000, help="HU lower clipping value.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_original_train_data(args.luna_root, args.output_root, args.expand_slices, args.upper, args.lower)
