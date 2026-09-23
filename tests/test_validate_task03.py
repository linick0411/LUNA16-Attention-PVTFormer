import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.validate_task03 import validate_task03


def _write_case(root, index, mask_value=255):
    image_dir = Path(root) / f"nodule_{index}" / "images"
    mask_dir = Path(root) / f"nodule_{index}" / "masks" / "nodule"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = np.full((4, 4), 100, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = mask_value
    cv2.imwrite(str(image_dir / f"{index}_0000.png"), image)
    cv2.imwrite(str(mask_dir / f"{index}_0000.png"), mask)


def _write_provenance_fixture(task_root, process_root):
    cases = [(0, "test"), (30, "validation"), (60, "train")]
    case_rows = []
    series_rows = []
    for source_index, (case_index, split) in enumerate(cases):
        _write_case(task_root, case_index)
        source_image_dir = process_root / "image" / str(source_index)
        source_mask_dir = process_root / "mask" / str(source_index)
        source_image_dir.mkdir(parents=True)
        source_mask_dir.mkdir(parents=True)
        shutil.copy2(task_root / f"nodule_{case_index}" / "images" / f"{case_index}_0000.png", source_image_dir / "0.png")
        shutil.copy2(task_root / f"nodule_{case_index}" / "masks" / "nodule" / f"{case_index}_0000.png", source_mask_dir / "0.png")
        uid = f"series-{source_index}"
        case_rows.append(
            {
                "case_name": f"nodule_{case_index}",
                "source_series_index": source_index,
                "seriesuid": uid,
                "subset": source_index,
                "slice_count": 1,
                "split": split,
            }
        )
        series_rows.append(
            {
                "series_index": source_index,
                "seriesuid": uid,
                "subset": source_index,
                "source_mhd": f"subset{source_index}/{uid}.mhd",
                "first_kept_slice": 10,
                "last_kept_slice_exclusive": 11,
                "slice_count": 1,
                "positive_slice_count": 1,
            }
        )

    with (task_root / "case_manifest.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=case_rows[0].keys())
        writer.writeheader()
        writer.writerows(case_rows)
    with (process_root / "series_manifest.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=series_rows[0].keys())
        writer.writeheader()
        writer.writerows(series_rows)


def test_validator_reports_all_three_splits(tmp_path):
    _write_case(tmp_path, 0)
    _write_case(tmp_path, 30)
    _write_case(tmp_path, 60)

    report = validate_task03(tmp_path, check_pixels=True)

    assert report["folder_count"] == 3
    assert report["image_count"] == report["mask_count"] == 3
    assert report["splits"] == {"train": 1, "validation": 1, "test": 1}
    assert report["empty_mask_count"] == 0
    assert report["non_binary_mask_count"] == 0


def test_validator_rejects_shifted_pairs(tmp_path):
    _write_case(tmp_path, 0)
    _write_case(tmp_path, 30)
    _write_case(tmp_path, 60)
    mask = tmp_path / "nodule_60" / "masks" / "nodule" / "60_0000.png"
    mask.rename(mask.with_name("60_0001.png"))

    with pytest.raises(RuntimeError, match="Image/mask mismatch"):
        validate_task03(tmp_path)


def test_validator_traces_contiguous_source_slices_to_task03(tmp_path):
    task_root = tmp_path / "task"
    process_root = tmp_path / "process"
    _write_provenance_fixture(task_root, process_root)

    report = validate_task03(task_root, process_root=process_root, provenance_samples=3)

    assert report["slice_order_validation"] == {
        "status": "passed",
        "cases_checked": 3,
        "slices_checked": 3,
        "content_sample_cases": ["nodule_0", "nodule_30", "nodule_60"],
        "content_sample_slices": 3,
        "original_z_index_rule": "first_kept_slice + exported_slice_index",
    }


def test_validator_rejects_non_contiguous_source_order(tmp_path):
    task_root = tmp_path / "task"
    process_root = tmp_path / "process"
    _write_provenance_fixture(task_root, process_root)
    source = process_root / "image" / "0" / "0.png"
    source.rename(source.with_name("1.png"))

    with pytest.raises(RuntimeError, match="Source image/mask mismatch|Non-contiguous source slice order"):
        validate_task03(task_root, process_root=process_root)
