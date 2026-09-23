from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from luna16_data import _media_by_stem
from scripts.preprocess_luna16.LUNA_mask_extraction import create_masks
from scripts.preprocess_luna16.convert_process_to_task03 import convert_process_to_task03
from scripts.preprocess_luna16.luna2D_mask import process_original_train_data


def test_synthetic_luna_volume_reaches_paired_png_task03(tmp_path):
    luna_root = tmp_path / "luna16"
    subset = luna_root / "subset0"
    subset.mkdir(parents=True)

    uid = "1.2.3.4.5"
    ct_array = np.full((7, 8, 8), -500.0, dtype=np.float32)
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(ct_image, str(subset / f"{uid}.mhd"))
    pd.DataFrame(
        [
            {
                "seriesuid": uid,
                "coordX": 4.0,
                "coordY": 4.0,
                "coordZ": 3.0,
                "diameter_mm": 4.0,
            }
        ]
    ).to_csv(luna_root / "annotations.csv", index=False)

    create_masks(luna_root)
    process_original_train_data(luna_root, expand_slices=0)
    task_root = tmp_path / "Task03_lung"
    convert_process_to_task03(luna_root / "process", task_root)

    image_dir = task_root / "nodule_0" / "images"
    mask_dir = task_root / "nodule_0" / "masks" / "nodule"
    images = _media_by_stem(image_dir)
    masks = _media_by_stem(mask_dir)

    assert images
    assert images.keys() == masks.keys()
    assert all(Path(path).suffix == ".png" for path in images.values())
    assert all(Path(path).suffix == ".png" for path in masks.values())
    assert any(np.any(sitk.GetArrayFromImage(sitk.ReadImage(path)) > 0) for path in masks.values())
    process_manifest = pd.read_csv(luna_root / "process" / "series_manifest.csv")
    task_manifest = pd.read_csv(task_root / "case_manifest.csv")
    assert process_manifest.loc[0, "seriesuid"] == uid
    assert task_manifest.loc[0, "seriesuid"] == uid
    assert task_manifest.loc[0, "case_name"] == "nodule_0"
    assert task_manifest.loc[0, "split"] == "train"
