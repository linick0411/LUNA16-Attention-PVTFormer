import os
import time
from dataclasses import dataclass
from operator import add
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from luna16_data import DEFAULT_DATA_DIR, load_data
from utils import calculate_metrics, create_dir, seeding


@dataclass
class EvaluationConfig:
    model_name: str
    checkpoint_name: str
    output_dir: Path
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoints_dir: Path = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    image_size: int = int(os.environ.get("IMAGE_SIZE", "192"))
    use_25d: bool = os.environ.get("USE_25D", "1") != "0"
    seed: int = int(os.environ.get("SEED", "42"))


def _read_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _neighbor_indices(index, paths):
    center_dir = os.path.dirname(paths[index])
    prev_idx = index
    next_idx = index

    if index - 1 >= 0 and os.path.dirname(paths[index - 1]) == center_dir:
        prev_idx = index - 1

    if index + 1 < len(paths) and os.path.dirname(paths[index + 1]) == center_dir:
        next_idx = index + 1

    return prev_idx, index, next_idx


def _process_prediction(y_pred):
    y_pred = y_pred[0].detach().cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = (y_pred > 0.5).astype(np.uint8) * 255
    y_pred = np.expand_dims(y_pred, axis=-1)
    return np.concatenate([y_pred, y_pred, y_pred], axis=2)


def _print_score(metrics_score, num_samples):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples
    acc = metrics_score[4] / num_samples
    f2 = metrics_score[5] / num_samples
    hd = metrics_score[6] / num_samples
    auc_count = metrics_score[8]
    auc = metrics_score[7] / auc_count if auc_count > 0 else float("nan")

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - "
        f"Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - "
        f"HD: {hd:1.4f} - AUC: {auc:1.4f}"
    )


def _load_ignore_mask(mask_path, size):
    path = Path(mask_path)
    ignore_path = path.parent.parent / "ignore" / path.name
    if not ignore_path.exists():
        return None

    ignore = _read_gray(str(ignore_path))
    ignore = cv2.resize(ignore, size, interpolation=cv2.INTER_NEAREST)
    ignore = (ignore > 127).astype(np.float32)
    return torch.from_numpy(ignore).unsqueeze(0).unsqueeze(0)


def run_evaluation(model_factory, config):
    seeding(config.seed)
    size = (config.image_size, config.image_size)
    checkpoint_path = config.checkpoints_dir / config.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run the matching train script first.")

    for item in ("mask", "joint"):
        create_dir(config.output_dir / item)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    (_, _), (_, _), (test_x, test_y) = load_data(config.data_dir)
    if len(test_x) == 0:
        raise RuntimeError(f"Test split is empty. Check data directory: {config.data_dir}")

    metrics_score = [0.0] * 9
    time_taken = []

    for index, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        if config.use_25d:
            prev_idx, center_idx, next_idx = _neighbor_indices(index, test_x)
            slices = []
            for slice_idx in (prev_idx, center_idx, next_idx):
                img = _read_gray(test_x[slice_idx])
                img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
                slices.append(img)

            image_np = np.stack(slices, axis=0)
            image_np = np.expand_dims(image_np, axis=1).astype(np.float32) / 255.0
            save_img = cv2.cvtColor(slices[1], cv2.COLOR_GRAY2BGR)
        else:
            img = _read_gray(x_path)
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
            image_np = np.expand_dims(img, axis=(0, 1)).astype(np.float32) / 255.0
            save_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        image = torch.from_numpy(image_np).unsqueeze(0).to(device)

        mask_np = _read_gray(y_path)
        mask_np = cv2.resize(mask_np, size, interpolation=cv2.INTER_NEAREST)
        mask_np = (mask_np > 127).astype(np.float32)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        save_mask = (mask_np * 255).astype(np.uint8)
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)

        ignore = _load_ignore_mask(y_path, size)
        if ignore is not None:
            ignore = ignore.to(device)

        with torch.no_grad():
            start_time = time.time()
            y_pred = torch.sigmoid(model(image))
            time_taken.append(time.time() - start_time)

            if ignore is not None:
                valid_mask = (ignore < 0.5).float()
                score = calculate_metrics(mask * valid_mask, y_pred * valid_mask, include_auc=True, auc_mask=valid_mask)
            else:
                score = calculate_metrics(mask, y_pred, include_auc=True)
            metrics_score = [add(a, b) for a, b in zip(metrics_score, score)]
            y_pred_vis = _process_prediction(y_pred)

        name = f"{Path(x_path).parent.parent.name}_{Path(x_path).name}"
        line = np.ones((size[0], 10, 3), dtype=np.uint8) * 255
        joint = np.concatenate([save_img, line, save_mask, line, y_pred_vis], axis=1)
        cv2.imwrite(str(config.output_dir / "joint" / name), joint)
        cv2.imwrite(str(config.output_dir / "mask" / name), y_pred_vis)

    _print_score(metrics_score, len(test_x))
    mean_time_taken = float(np.mean(time_taken))
    print(f"Mean FPS: {1 / mean_time_taken if mean_time_taken > 0 else 0:.4f}")
