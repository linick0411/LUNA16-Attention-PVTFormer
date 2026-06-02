# Reproducibility

## Environment

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Local syntax validation was run with:

```powershell
python -m py_compile <all repository python files>
```

On the local Windows machine used to prepare this repo, `torch`, `timm`, `opencv-python`, `pandas`, `scikit-learn`, and `scipy` were installed. `albumentations` and `SimpleITK` were missing locally, so full training/preprocessing was not executed on the local machine during repo cleanup.

## Data

Place the prepared data at `data/Task03_lung` or set:

```powershell
$env:LUNA16_TASK_DIR="D:\datasets\Task03_lung"
```

The repository ignores `data/` because LUNA16/LIDC-IDRI must not be redistributed through this repo.

## Weights

Place PVTv2-B3 weights at:

```text
checkpoints/pvt_v2_b3.pth
```

or set:

```powershell
$env:PVT_PRETRAINED_PATH="D:\weights\pvt_v2_b3.pth"
```

Training still runs without the pretrained weights, but the backbone will initialize randomly and results will not match pretrained experiments.

## Commands

Train:

```powershell
python train_attention_gate.py
python train_voxel_attention.py
python train_coordinate_attention.py
```

Evaluate:

```powershell
python eval_attention_gate.py
python eval_voxel_attention.py
python eval_coordinate_attention.py
```

Common overrides:

```powershell
$env:IMAGE_SIZE="192"
$env:BATCH_SIZE="16"
$env:NUM_EPOCHS="500"
$env:NUM_WORKERS="2"
$env:CHECKPOINT_DIR="checkpoints"
```

## Expected Outputs

Training:

```text
logs/train_log_attention_gate.txt
logs/train_log_voxel_attention.txt
logs/train_log_coordinate_attention.txt
checkpoints/checkpoint_attention_gate.pth
checkpoints/checkpoint_voxel_attention.pth
checkpoints/checkpoint_coordinate_attention.pth
```

Evaluation:

```text
results/attention_gate/mask
results/attention_gate/joint
results/voxel_attention/mask
results/voxel_attention/joint
results/coordinate_attention/mask
results/coordinate_attention/joint
```

## Metric Notes

- AUC is computed pixel-wise from predicted probabilities and binary mask labels.
- HD is computed from the current project implementation in `metrics.py`.
- Because LUNA16 labels are converted from center/diameter annotations into coarse masks, metric values should be interpreted as experiment-comparison signals, not clinical validation.
