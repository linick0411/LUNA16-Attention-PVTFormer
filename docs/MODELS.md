# Model Variants

All three variants use the same basic pipeline:

- CT input is prepared as 2.5D slices: previous, center, next.
- Each grayscale slice is repeated internally to 3 channels because PVTv2 is an RGB-pretrained backbone.
- PVTv2-B3 extracts multi-scale encoder features.
- The decoder upsamples and fuses multi-scale features to predict the center-slice nodule mask.

## Attention Gate

Files:

- `train_attention_gate.py`
- `eval_attention_gate.py`
- `model3.py`

This variant adds Attention U-Net style gates on decoder skip connections. The decoder feature acts as a gating signal and suppresses irrelevant encoder skip features before concatenation.

Use this when the question is: "Can decoder-guided skip filtering improve nodule segmentation?"

## Voxel Attention

Files:

- `train_voxel_attention.py`
- `eval_voxel_attention.py`
- `model2.py`

This variant processes the three neighboring slices through the same PVTv2 encoder, reshapes features back into `(B, S, C, H, W)`, and learns cross-slice attention weights at the feature level.

Voxel attention is applied at encoder feature levels:

- `e1`: 64 channels
- `e2`: 128 channels
- `e3`: 320 channels

Use this when the question is: "Can the model learn which neighboring CT slice is most useful at each spatial location?"

## Coordinate Attention

Files:

- `train_coordinate_attention.py`
- `eval_coordinate_attention.py`
- `model4.py`

This variant fuses the three slice features with coordinate attention. It pools features separately along height and width, then generates direction-aware attention maps that preserve positional information.

Use this when the question is: "Can direction-aware spatial/channel attention improve slice-feature fusion?"

## What Is Shared

The shared logic lives in:

- `luna16_data.py`
- `train_common.py`
- `eval_common.py`

Shared settings:

- Input size: `192`
- Batch size: `16`
- Learning rate: `1e-4`
- Loss: Dice + BCE
- Scheduler: ReduceLROnPlateau
- Augmentation: rotation, horizontal flip, vertical flip, coarse dropout
- Metrics: Jaccard, F1, Recall, Precision, Accuracy, F2, HD, AUC

Default early stopping patience:

- Attention Gate: `50`
- Voxel Attention: `50`
- Coordinate Attention: `20`
