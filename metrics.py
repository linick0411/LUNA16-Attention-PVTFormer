import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

""" Loss Functions -------------------------------------- """
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        logits = inputs.view(-1)
        targets = targets.view(-1)
        probabilities = torch.sigmoid(logits)

        intersection = (probabilities * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(probabilities.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def confusion_counts(y_true, y_pred, threshold=0.5, valid_mask=None):
    """Return TP, FP, TN, and FN counts for binary segmentation arrays."""
    y_true = np.asarray(y_true) > 0.5
    y_pred = np.asarray(y_pred) > threshold
    if valid_mask is None:
        valid = np.ones_like(y_true, dtype=bool)
    else:
        valid = np.asarray(valid_mask) > 0.5
        if valid.shape != y_true.shape:
            valid = np.broadcast_to(valid, y_true.shape)

    tp = np.count_nonzero(valid & y_true & y_pred)
    fp = np.count_nonzero(valid & ~y_true & y_pred)
    tn = np.count_nonzero(valid & ~y_true & ~y_pred)
    fn = np.count_nonzero(valid & y_true & ~y_pred)
    return np.asarray([tp, fp, tn, fn], dtype=np.float64)


def metrics_from_confusion(counts):
    """Compute micro-averaged metrics from accumulated TP/FP/TN/FN counts."""
    tp, fp, tn, fn = [float(value) for value in counts]

    def ratio(numerator, denominator, empty_value=1.0):
        return numerator / denominator if denominator > 0 else empty_value

    return {
        "jaccard": ratio(tp, tp + fp + fn),
        "f1": ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "recall": ratio(tp, tp + fn),
        "precision": ratio(tp, tp + fp),
        "accuracy": ratio(tp + tn, tp + fp + tn + fn),
        "f2": ratio(5.0 * tp, 5.0 * tp + 4.0 * fn + fp),
    }

## https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/319452
def hd_dist(preds, targets):
    """Return the symmetric Hausdorff distance between binary mask contours.

    The distance is reported in resized-image pixels. When exactly one mask is
    empty, the image diagonal is returned as the maximum spatial penalty; two
    empty masks have zero distance.
    """
    preds = np.asarray(preds) > 0.5
    targets = np.asarray(targets) > 0.5

    pred_points = np.argwhere(preds)
    target_points = np.argwhere(targets)

    if pred_points.size == 0 and target_points.size == 0:
        return 0.0

    if pred_points.size == 0 or target_points.size == 0:
        return float(np.linalg.norm(np.asarray(preds.shape, dtype=np.float64) - 1.0))

    pred_to_target = directed_hausdorff(pred_points, target_points)[0]
    target_to_pred = directed_hausdorff(target_points, pred_points)[0]
    return float(max(pred_to_target, target_to_pred))
