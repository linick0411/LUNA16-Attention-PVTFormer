from pathlib import Path

import cv2
import numpy as np

import demo_infer
from demo_infer import binary_metrics, collect_case_slices, resolve_checkpoint


def _write_gray(path: Path, value: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_auto_positive_selects_largest_mask(tmp_path):
    case_dir = tmp_path / "nodule_1"
    for index in range(3):
        _write_gray(case_dir / "images" / f"1_{index:04d}.png", value=30 + index)
        mask = np.zeros((8, 8), dtype=np.uint8)
        if index == 1:
            mask[2:6, 2:6] = 255
        elif index == 2:
            mask[3:5, 3:5] = 255
        path = case_dir / "masks" / "nodule" / f"1_{index:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), mask)

    previous, center, following, mask = collect_case_slices(case_dir, auto_positive=True)

    assert previous.name == "1_0000.png"
    assert center.name == "1_0001.png"
    assert following.name == "1_0002.png"
    assert mask.name == "1_0001.png"


def test_legacy_attention_gate_checkpoint_is_original_weight():
    checkpoint = resolve_checkpoint("attention_gate", checkpoint=None, weight_set="legacy")
    expected = Path(demo_infer.__file__).resolve().parent.parent / "PVTFormer-main/files/checkpoint_model3.pth"
    assert checkpoint == expected


def test_binary_metrics_report_overlap_and_errors():
    prediction = np.zeros((2, 2), dtype=np.uint8)
    prediction[0, :2] = 255
    target = np.zeros((2, 2, 3), dtype=np.uint8)
    target[0, 0] = 255
    target[1, 0] = 255

    metrics = binary_metrics(prediction, target)

    assert metrics["dice_f1"] == 0.5
    assert metrics["iou_jaccard"] == 1 / 3
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5
