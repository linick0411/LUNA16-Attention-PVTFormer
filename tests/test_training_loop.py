import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from metrics import DiceBCELoss
from train_common import evaluate_one_epoch, train_one_epoch


def test_shared_training_and_validation_loop_updates_a_tiny_model():
    torch.manual_seed(7)
    images = torch.rand(3, 3, 8, 8)
    masks = (images[:, :1] > 0.5).float()
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = DiceBCELoss()
    before = model.weight.detach().clone()

    train_loss, train_metrics = train_one_epoch(model, loader, optimizer, loss_fn, torch.device("cpu"))
    valid_loss, valid_metrics = evaluate_one_epoch(model, loader, loss_fn, torch.device("cpu"))

    assert np.isfinite(train_loss)
    assert np.isfinite(valid_loss)
    assert all(0.0 <= value <= 1.0 for value in train_metrics + valid_metrics)
    assert not torch.equal(before, model.weight.detach())
