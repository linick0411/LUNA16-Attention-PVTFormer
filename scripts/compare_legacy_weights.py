"""Evaluate recovered legacy checkpoints under the current sphere-v1 contract.

The script reports both the historical per-slice aggregation and the corrected
dataset-level micro aggregation.  It never overwrites either legacy or current
checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from luna16_data import load_data
from metrics import confusion_counts, metrics_from_confusion
from model2 import PVTFormer_Voxel
from model3 import PVTFormer as AttentionGate
from model4 import PVTFormer as CoordinateAttention
from model_baseline import PVTFormerBaseline
from utils import seeding


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("LUNA16_TASK_DIR", REPO / "data/Task03_lung_sphere-v1"))
LEGACY_ROOT = Path(os.environ.get("LEGACY_PVTFORMER_ROOT", REPO.parent / "PVTFormer-main"))
OUTPUT_ROOT = Path(os.environ.get("LEGACY_COMPARISON_DIR", REPO / "results/legacy-comparison-sphere-v1"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "192"))
THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))


MODELS = {
    # files/checkpoint.pth was overwritten by a later, incompatible experiment.
    # files0 is the recovered baseline candidate whose behavior matches the
    # LUNA16 experiment family; its exact report-era provenance is unavailable.
    "baseline": (PVTFormerBaseline, "files0/checkpoint.pth", False),
    "voxel_attention": (PVTFormer_Voxel, "files/checkpoint_model2.pth", True),
    "attention_gate": (AttentionGate, "files/checkpoint_model3.pth", True),
    "coordinate_attention": (CoordinateAttention, "files/checkpoint_model4.pth", True),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remap_baseline(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map legacy module names to the semantically identical current names."""
    remapped = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("d1", "d2", "d3", "u1", "u2", "u3"):
            if new_key.startswith(f"{prefix}.r1."):
                new_key = new_key.replace(f"{prefix}.r1.", f"{prefix}.residual.", 1)
        if new_key.startswith("r1."):
            new_key = new_key.replace("r1.", "fuse.", 1)
        if new_key.startswith("y."):
            new_key = new_key.replace("y.", "output.", 1)
        remapped[new_key] = value
    return remapped


def _read_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _neighbors(index: int, paths: list[str]) -> tuple[int, int, int]:
    center_dir = os.path.dirname(paths[index])
    previous = index - 1 if index > 0 and os.path.dirname(paths[index - 1]) == center_dir else index
    following = index + 1 if index + 1 < len(paths) and os.path.dirname(paths[index + 1]) == center_dir else index
    return previous, index, following


def _input_tensor(index: int, paths: list[str], use_25d: bool, device: torch.device) -> torch.Tensor:
    size = (IMAGE_SIZE, IMAGE_SIZE)
    if use_25d:
        slices = [cv2.resize(_read_gray(paths[i]), size, interpolation=cv2.INTER_LINEAR) for i in _neighbors(index, paths)]
        array = np.stack(slices, axis=0)[:, None, :, :]
    else:
        image = cv2.imread(paths[index], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {paths[index]}")
        array = np.transpose(cv2.resize(image, size, interpolation=cv2.INTER_LINEAR), (2, 0, 1))
    return torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(0).to(device)


def _target_and_valid(mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    size = (IMAGE_SIZE, IMAGE_SIZE)
    target = cv2.resize(_read_gray(mask_path), size, interpolation=cv2.INTER_NEAREST) > 127
    ignore_path = Path(mask_path).parent.parent / "ignore" / Path(mask_path).name
    if ignore_path.exists():
        ignore = cv2.resize(_read_gray(str(ignore_path)), size, interpolation=cv2.INTER_NEAREST) > 127
        valid = ~ignore
    else:
        valid = np.ones_like(target, dtype=bool)
    return target, valid


def evaluate_one(name: str, factory, checkpoint: Path, use_25d: bool, test_x: list[str], test_y: list[str]) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if name == "baseline":
        state = _remap_baseline(state)
    model = factory().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    total_counts = np.zeros(4, dtype=np.float64)
    per_slice = []
    positive_slice = []
    positive_predicted_empty = 0

    for index, mask_path in tqdm(list(enumerate(test_y)), desc=f"legacy {name}", unit="slice"):
        image = _input_tensor(index, test_x, use_25d, device)
        target, valid = _target_and_valid(mask_path)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probability = torch.sigmoid(model(image))[0, 0].float().cpu().numpy()
        prediction = probability > THRESHOLD
        counts = confusion_counts(target, prediction, valid_mask=valid)
        total_counts += counts
        slice_metrics = metrics_from_confusion(counts)
        per_slice.append(slice_metrics)
        if np.any(target & valid):
            positive_slice.append(slice_metrics)
            if not np.any(prediction & valid):
                positive_predicted_empty += 1

    metric_names = ("jaccard", "f1", "recall", "precision", "accuracy", "f2")
    result = {
        "model": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "test_samples": len(test_y),
        "positive_slices": len(positive_slice),
        "positive_slices_predicted_empty": positive_predicted_empty,
        "threshold": THRESHOLD,
        "micro": metrics_from_confusion(total_counts),
        "legacy_per_slice": {key: float(np.mean([item[key] for item in per_slice])) for key in metric_names},
        "positive_slice_mean": {key: float(np.mean([item[key] for item in positive_slice])) for key in metric_names},
        "counts_tp_fp_tn_fn": total_counts.tolist(),
    }
    return result


def main() -> None:
    seeding(42)
    (_, _), (_, _), (test_x, test_y) = load_data(DATA_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, (factory, checkpoint_name, use_25d) in MODELS.items():
        checkpoint = LEGACY_ROOT / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = evaluate_one(name, factory, checkpoint, use_25d, test_x, test_y)
        results.append(result)
        (OUTPUT_ROOT / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (OUTPUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "micro_f1", "legacy_per_slice_f1", "positive_slice_f1", "micro_jaccard", "micro_recall", "micro_precision", "positive_slices_predicted_empty"])
        for result in results:
            writer.writerow([
                result["model"], result["micro"]["f1"], result["legacy_per_slice"]["f1"],
                result["positive_slice_mean"]["f1"], result["micro"]["jaccard"], result["micro"]["recall"],
                result["micro"]["precision"], result["positive_slices_predicted_empty"],
            ])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
