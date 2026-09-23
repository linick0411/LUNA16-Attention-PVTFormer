import numpy as np

from eval_common import _update_diagnostics
from metrics import confusion_counts, metrics_from_confusion


def _empty_diagnostics():
    return {
        "counts": np.zeros(4, dtype=np.float64),
        "hausdorff_sum": 0.0,
        "hausdorff_count": 0,
        "auc_sum": 0.0,
        "auc_count": 0,
        "positive_slices": 0,
        "negative_slices": 0,
        "negative_slices_with_false_positive": 0,
    }


def test_micro_f1_is_not_inflated_by_empty_background_slice():
    truth = np.zeros((2, 4, 4), dtype=np.uint8)
    truth[0, 1, 1] = 1
    prediction = np.zeros_like(truth)

    summary = metrics_from_confusion(confusion_counts(truth, prediction))

    assert summary["f1"] == 0.0
    assert summary["recall"] == 0.0


def test_diagnostics_skip_empty_empty_hausdorff_but_track_negative_slices():
    diagnostics = _empty_diagnostics()
    empty = np.zeros((4, 4), dtype=np.float32)

    _update_diagnostics(empty, empty, np.ones_like(empty), diagnostics)

    assert diagnostics["negative_slices"] == 1
    assert diagnostics["hausdorff_count"] == 0
    assert diagnostics["negative_slices_with_false_positive"] == 0


def test_diagnostics_penalize_false_positive_only_slice():
    diagnostics = _empty_diagnostics()
    truth = np.zeros((4, 4), dtype=np.float32)
    score = np.zeros((4, 4), dtype=np.float32)
    score[1, 1] = 1.0

    _update_diagnostics(truth, score, np.ones_like(truth), diagnostics)

    assert diagnostics["negative_slices_with_false_positive"] == 1
    assert diagnostics["hausdorff_count"] == 1
    assert diagnostics["hausdorff_sum"] > 0
