# Results And Evidence Status

## Legacy Presentation Results

The table below is transcribed from `Final Presentation1.pptx`, slide 17. It
records the original project run and is the primary result used for the project
conclusion. The recovered checkpoints load successfully, but the exact
report-era dataset build and split manifest are not available, so the table
cannot be recreated bit-for-bit in the rebuilt environment.

| Model | Recall | F1 / Dice | Precision | Accuracy | Jaccard / IoU | F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PVTFormer baseline | 0.7953 | 0.7273 | 0.8979 | 0.9996 | 0.7070 | 0.7164 |
| Voxel Attention | 0.8130 | 0.7917 | 0.9509 | 0.9997 | 0.7757 | 0.7917 |
| Attention Gate | 0.8168 | **0.7994** | **0.9555** | **0.9997** | **0.7827** | **0.7983** |
| Coordinate Attention | **0.8208** | 0.7899 | 0.9384 | **0.9997** | 0.7718 | 0.7892 |

These numbers support the project statement that, in the original run,
Attention Gate had the highest F1, Precision, Jaccard, and F2, while Coordinate
Attention had the highest Recall. They are reported as the original experiment,
not as values reproduced by the corrected `sphere-v1` pipeline.

## Recovered Legacy Weights

The recovered Voxel Attention, Attention Gate, and Coordinate Attention weights
strictly match the current model tensor names and shapes. The baseline required
only semantic key renaming (`r1` to `residual`/`fuse`, `y` to `output`). The
usable baseline candidate is `PVTFormer-main/files0/checkpoint.pth`; the later
`files/checkpoint.pth` was overwritten by or belongs to a different experiment.

When the recovered weights are tested on the corrected 4,076-slice `sphere-v1`
test set, their legacy per-slice F1 is 0.8377 to 0.8451. Their foreground micro
F1 is 0.5541 to 0.5630 because they favor Recall (0.8378 to 0.8806) while
producing more false-positive pixels (Precision 0.4042 to 0.4239). Full values
and checkpoint hashes are recorded in `docs/LEGACY_WEIGHT_COMPARISON.md`.

Background-only slices are clinically relevant and are intentionally retained.
The two aggregations answer different questions:

- Per-slice mean evaluates the average complete slice and gives an empty target
  plus empty prediction a perfect score.
- Dataset-level micro F1 measures foreground overlap after pooling every valid
  pixel, so correctly empty slices do not inflate nodule overlap.

For the quick project conclusion, the original presentation table and its
slice-level interpretation are used. The corrected micro table is supplementary
analysis rather than a replacement for the original experiment.

## Why The Old Hausdorff Values Are Excluded

The presentation also listed Hausdorff Distance values, but the historical implementation did not compute symmetric spatial Hausdorff distance correctly. The repository now uses foreground coordinates in both directions and explicit empty-mask handling. Because the algorithm changed, the old HD values are intentionally omitted rather than mixed with the corrected metric.

## Fairness Limitation In The Historical Run

The historical Coordinate Attention training used an early-stopping patience of 20 epochs, while the other variants used 50. The code now sets the baseline and all three attention variants to 50. A fair conclusion about model ranking requires retraining under the same:

- scan-level train, validation, and test split;
- PVTv2-B3 initialization;
- image size and batch size;
- data augmentation;
- optimizer, learning rate, loss, scheduler, and stopping rule;
- metric implementation and decision threshold.

## Corrected Reproduction Status

The corrected `sphere-v1` reproduction is complete. It uses 601 cases and
44,550 image/mask pairs with a fixed scan-level split of 35,777 training, 4,697
validation, and 4,076 test slices. All five models have saved checkpoints and
test metrics.

| Model | Micro F1 | Recall | Precision |
| --- | ---: | ---: | ---: |
| 2D baseline | 0.6379 | 0.5963 | 0.6857 |
| Plain 2.5D control | 0.6398 | 0.6054 | 0.6783 |
| Attention Gate | **0.6518** | 0.6001 | **0.7132** |
| Voxel Attention | 0.6165 | 0.5963 | 0.6382 |
| Coordinate Attention | 0.6504 | **0.6044** | 0.7039 |

These values use a different pseudo-mask construction, split, and aggregation
from the original presentation. They must be cited as a separate reproduction.

The corresponding machine-readable metrics, merged training histories,
data/z-order validation report, and checkpoint SHA-256 manifest are published
under [`results/published/sphere-v1`](../results/published/sphere-v1). The model
weights themselves are retained outside Git because every best checkpoint is
larger than GitHub's normal 100 MB per-file limit.

## Evidence Included For The Final Report

The final report should include:

1. original presentation result table;
2. recovered-checkpoint audit under one fixed test contract;
3. corrected reproduction table and training curves;
4. saved prediction masks and joint visual comparisons;
5. an explanation of slice-level versus foreground-micro evaluation;
6. a limitation that LUNA16 supplies center and diameter annotations rather
   than expert pixel contours, so this project constructs pseudo-masks.

The rebuilt evaluator reports dataset-level micro overlap metrics alongside
negative-slice statistics. This complements the legacy whole-slice view without
removing clinically relevant background-only slices.
