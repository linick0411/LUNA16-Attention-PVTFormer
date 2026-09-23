import math

import numpy as np
import torch

from metrics import DiceBCELoss, hd_dist


def test_hausdorff_distance_is_zero_for_identical_masks():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 3:6] = 1

    assert hd_dist(mask, mask) == 0.0


def test_hausdorff_distance_is_symmetric_and_spatial():
    left = np.zeros((8, 8), dtype=np.uint8)
    right = np.zeros((8, 8), dtype=np.uint8)
    left[2, 2] = 1
    right[5, 6] = 1

    expected = 5.0
    assert hd_dist(left, right) == expected
    assert hd_dist(right, left) == expected


def test_hausdorff_distance_handles_empty_masks():
    empty = np.zeros((8, 8), dtype=np.uint8)
    non_empty = empty.copy()
    non_empty[4, 4] = 1

    assert hd_dist(empty, empty) == 0.0
    assert hd_dist(empty, non_empty) == math.sqrt(7**2 + 7**2)


def test_dice_bce_loss_is_finite_for_extreme_logits():
    logits = torch.tensor([[[[-100.0, 100.0]]]], requires_grad=True)
    targets = torch.tensor([[[[0.0, 1.0]]]])

    loss = DiceBCELoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
