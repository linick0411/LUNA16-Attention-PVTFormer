import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


DEFAULT_DATA_DIR = Path(os.environ.get("LUNA16_TASK_DIR", "data/Task03_lung"))


def _nodule_index(path):
    name = Path(path).name
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return name


def load_data(path=DEFAULT_DATA_DIR):
    """Load Task03_lung folders and keep the original split policy."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(
            f"LUNA16 Task03_lung directory not found: {root}. "
            "Set LUNA16_TASK_DIR or place data under data/Task03_lung."
        )

    def get_data(name):
        nodule_dir = root / name
        images = sorted(glob(str(nodule_dir / "images" / "*.jpg")))
        masks = sorted(glob(str(nodule_dir / "masks" / "nodule" / "*.jpg")))
        return images, masks

    dirs = sorted(
        [item.name for item in root.iterdir() if item.is_dir() and item.name.startswith("nodule_")],
        key=_nodule_index,
    )
    test_names = [f"nodule_{i}" for i in range(0, 30)]
    valid_names = [f"nodule_{i}" for i in range(30, 60)]
    train_names = [item for item in dirs if item not in set(test_names + valid_names)]

    splits = []
    for names in (train_names, valid_names, test_names):
        split_x, split_y = [], []
        for name in names:
            x, y = get_data(name)
            split_x += x
            split_y += y
        splits.append((split_x, split_y))

    return tuple(splits)


class LUNA16SliceDataset(Dataset):
    """2.5D dataset: previous, center, and next CT slices predict the center mask."""

    def __init__(self, images_path, masks_path, size, transform=None, use_25d=True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.use_25d = use_25d
        self.n_samples = len(images_path)

    def _get_neighbor_indices(self, index):
        center_dir = os.path.dirname(self.images_path[index])
        prev_idx = index
        next_idx = index

        if index - 1 >= 0 and os.path.dirname(self.images_path[index - 1]) == center_dir:
            prev_idx = index - 1

        if index + 1 < self.n_samples and os.path.dirname(self.images_path[index + 1]) == center_dir:
            next_idx = index + 1

        return prev_idx, index, next_idx

    @staticmethod
    def _read_gray(path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return image

    def __getitem__(self, index):
        if self.use_25d:
            prev_idx, center_idx, next_idx = self._get_neighbor_indices(index)
            slices = [self._read_gray(self.images_path[idx]) for idx in (prev_idx, center_idx, next_idx)]
            image = np.stack(slices, axis=-1)
        else:
            image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {self.images_path[index]}")

        mask = self._read_gray(self.masks_path[index])

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.use_25d:
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=1)
        else:
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            image = np.transpose(image, (2, 0, 1))

        image = image.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)
        return image, mask

    def __len__(self):
        return self.n_samples
