# Recovered Legacy Weight Comparison

## Contract

All values below use the same corrected `sphere-v1` test set: 4,076 slices,
including 954 positive slices, at 192 x 192 pixels and threshold 0.5.  The
comparison therefore isolates checkpoint behavior and aggregation method; it
does not reproduce the report-era dataset construction or split.

## Recovered legacy checkpoints

| Model | Recovered checkpoint | Micro F1 | Legacy per-slice F1 | Recall | Precision | Positive slices predicted empty |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline candidate | `PVTFormer-main/files0/checkpoint.pth` | 0.5571 | 0.8399 | 0.8651 | 0.4108 | 259 |
| Voxel Attention | `PVTFormer-main/files/checkpoint_model2.pth` | 0.5620 | 0.8451 | 0.8631 | 0.4166 | 245 |
| Attention Gate | `PVTFormer-main/files/checkpoint_model3.pth` | 0.5630 | 0.8377 | 0.8378 | 0.4239 | 288 |
| Coordinate Attention | `PVTFormer-main/files/checkpoint_model4.pth` | 0.5541 | 0.8379 | 0.8806 | 0.4042 | 252 |

The recovered `files/checkpoint.pth` is not the report baseline.  On this test
set it predicts almost the whole image as foreground (6,680,285 false-positive
pixels and micro F1 0.000003), showing that the file was overwritten by or came
from a later incompatible experiment.  `files0/checkpoint.pth` is retained as
the plausible LUNA16 baseline candidate, but there is no commit/checkpoint
manifest proving it is the exact report-era file.

## Newly trained checkpoints

| Model | Micro F1 | Legacy per-slice F1 | Recall | Precision | Positive slices predicted empty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.6379 | 0.8363 | 0.5963 | 0.6857 | 418 |
| Voxel Attention | 0.6165 | 0.8301 | 0.5963 | 0.6382 | 389 |
| Attention Gate | 0.6518 | 0.8347 | 0.6001 | 0.7132 | 471 |
| Coordinate Attention | 0.6504 | 0.8383 | 0.6044 | 0.7039 | 429 |

## Interpretation

The recovered legacy weights do not restore the report table when evaluated
under the corrected contract.  With the historical per-slice aggregation, both
old and new weights score around 0.83 to 0.85 because 3,122 of 4,076 test slices
are negative and an empty/empty slice contributes F1 = 1.  Dataset-level micro
F1 removes that inflation.

Under micro aggregation, the newly trained checkpoints are better than the
recovered legacy checkpoints.  Legacy checkpoints favor recall (0.84 to 0.88)
at the cost of precision (0.40 to 0.42), whereas the new checkpoints have lower
recall (about 0.60) but substantially higher precision (0.64 to 0.71).  The
report-era values therefore cannot be compared directly without recovering the
exact report-era pseudo-mask construction, split manifest, checkpoint identity,
and metric implementation.
