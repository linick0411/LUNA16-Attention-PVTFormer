import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from luna16_data import DEFAULT_DATA_DIR, LUNA16SliceDataset, load_data
from metrics import DiceBCELoss
from utils import calculate_metrics, create_dir, epoch_time, print_and_save, seeding, shuffling


@dataclass
class TrainingConfig:
    model_name: str
    checkpoint_name: str
    train_log_name: str
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoints_dir: Path = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    logs_dir: Path = Path(os.environ.get("LOG_DIR", "logs"))
    image_size: int = int(os.environ.get("IMAGE_SIZE", "192"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "16"))
    num_epochs: int = int(os.environ.get("NUM_EPOCHS", "500"))
    lr: float = float(os.environ.get("LR", "1e-4"))
    early_stopping_patience: int = 50
    num_workers: int = int(os.environ.get("NUM_WORKERS", "2"))
    use_25d: bool = os.environ.get("USE_25D", "1") != "0"
    seed: int = int(os.environ.get("SEED", "42"))


def _collect_metrics(y, y_pred):
    batch_scores = [calculate_metrics(yt, yp) for yt, yp in zip(y, y_pred)]
    return np.mean(batch_scores, axis=0)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_metrics = np.zeros(4, dtype=np.float64)

    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        y_pred = torch.sigmoid(logits)
        batch_metrics = _collect_metrics(y, y_pred)[:4]

        epoch_loss += loss.item()
        epoch_metrics += batch_metrics
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "F1": f"{batch_metrics[1]:.4f}"})

    return epoch_loss / len(loader), (epoch_metrics / len(loader)).tolist()


def evaluate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_metrics = np.zeros(4, dtype=np.float64)

    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            logits = model(x)
            loss = loss_fn(logits, y)
            y_pred = torch.sigmoid(logits)
            batch_metrics = _collect_metrics(y, y_pred)[:4]

            epoch_loss += loss.item()
            epoch_metrics += batch_metrics
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "F1": f"{batch_metrics[1]:.4f}"})

    return epoch_loss / len(loader), (epoch_metrics / len(loader)).tolist()


def run_training(model_factory, config):
    seeding(config.seed)
    create_dir(config.checkpoints_dir)
    create_dir(config.logs_dir)

    checkpoint_path = config.checkpoints_dir / config.checkpoint_name
    train_log_path = config.logs_dir / config.train_log_name
    if not train_log_path.exists():
        train_log_path.write_text("", encoding="utf-8")

    size = (config.image_size, config.image_size)
    print_and_save(str(train_log_path), str(datetime.datetime.now()))
    print_and_save(
        str(train_log_path),
        (
            f"Model: {config.model_name}\n"
            f"Data: {config.data_dir}\n"
            f"Image Size: {size}\n"
            f"Batch Size: {config.batch_size}\n"
            f"LR: {config.lr}\n"
            f"Epochs: {config.num_epochs}\n"
            f"Early Stopping Patience: {config.early_stopping_patience}\n"
            f"Use 2.5D: {config.use_25d}"
        ),
    )

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(config.data_dir)
    if len(train_x) == 0 or len(valid_x) == 0:
        raise RuntimeError(
            f"Dataset split is empty. train={len(train_x)}, valid={len(valid_x)}. "
            "Check LUNA16_TASK_DIR and Task03_lung structure."
        )

    train_x, train_y = shuffling(train_x, train_y)
    print_and_save(
        str(train_log_path),
        f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}",
    )

    transform = A.Compose(
        [
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
        ]
    )

    train_dataset = LUNA16SliceDataset(train_x, train_y, size, transform=transform, use_25d=config.use_25d)
    valid_dataset = LUNA16SliceDataset(valid_x, valid_y, size, transform=None, use_25d=config.use_25d)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    loss_fn = DiceBCELoss()
    print_and_save(str(train_log_path), "Optimizer: Adam\nLoss: BCE Dice Loss")

    best_valid_f1 = 0.0
    early_stopping_count = 0
    epoch_pbar = tqdm(range(config.num_epochs), desc="Epochs", unit="epoch")

    for epoch in epoch_pbar:
        start_time = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate_one_epoch(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if valid_metrics[1] > best_valid_f1:
            print_and_save(
                str(train_log_path),
                f"Valid F1 improved from {best_valid_f1:2.4f} to {valid_metrics[1]:2.4f}. "
                f"Saving checkpoint: {checkpoint_path}",
            )
            best_valid_f1 = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        epoch_mins, epoch_secs = epoch_time(start_time, time.time())
        epoch_pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Train F1": f"{train_metrics[1]:.4f}",
                "Valid Loss": f"{valid_loss:.4f}",
                "Valid F1": f"{valid_metrics[1]:.4f}",
                "Best F1": f"{best_valid_f1:.4f}",
                "LR": f"{current_lr:.2e}",
            }
        )

        log = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        log += (
            f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - "
            f"F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - "
            f"Precision: {train_metrics[3]:.4f}\n"
        )
        log += (
            f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - "
            f"F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - "
            f"Precision: {valid_metrics[3]:.4f}\n"
        )
        log += f"\tLearning Rate: {current_lr:.2e}\n"
        print_and_save(str(train_log_path), log)

        if early_stopping_count >= config.early_stopping_patience:
            print_and_save(
                str(train_log_path),
                f"Early stopping: validation F1 did not improve for {config.early_stopping_patience} epochs.",
            )
            break
