# LUNA16 Attention PVTFormer

This repository organizes three PVTFormer-based 2.5D lung nodule segmentation variants for the LUNA16/LIDC-IDRI dataset:

| Entry point | Model file | Attention variant |
| --- | --- | --- |
| `train_attention_gate.py` / `eval_attention_gate.py` | `model3.py` | Attention Gate on decoder skip connections |
| `train_voxel_attention.py` / `eval_voxel_attention.py` | `model2.py` | Voxel Attention fusion across neighboring CT slices |
| `train_coordinate_attention.py` / `eval_coordinate_attention.py` | `model4.py` | Coordinate Attention fusion across neighboring CT slices |

The code is adapted from the PVTFormer architecture proposed for CT liver segmentation and extended here for LUNA16 lung nodule segmentation experiments. It is research code only and is not a clinical diagnostic tool.

## What Is Included

- 2.5D training and evaluation pipeline using previous, center, and next CT slices.
- Three attention variants with separate train/eval entry points.
- LUNA16 preprocessing scripts for converting `.mhd` CT volumes and `annotations.csv` into `Task03_lung` JPG folders.
- Evaluation metrics: Jaccard/IoU, F1/Dice, Recall, Precision, Accuracy, F2, Hausdorff distance, AUC, and FPS.
- Documentation for dataset preparation, model differences, reproducibility, and citation/third-party attribution.

## Dataset

The dataset is not included in this repository. Prepare it under:

```text
data/Task03_lung/
  nodule_0/
    images/*.jpg
    masks/nodule/*.jpg
  nodule_1/
    images/*.jpg
    masks/nodule/*.jpg
```

Or point to an existing folder:

```powershell
$env:LUNA16_TASK_DIR="D:\datasets\Task03_lung"
```

The prepared remote dataset used for this project contained `nodule_0` through `nodule_600`, with 45,461 image JPGs and 45,461 mask JPGs. The split policy matches the current project code:

- Test: `nodule_0` to `nodule_29`
- Validation: `nodule_30` to `nodule_59`
- Train: all remaining `nodule_*` folders

See [docs/DATASET.md](docs/DATASET.md) for the full preprocessing flow.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the PVTv2-B3 pretrained weights separately and place them at:

```text
checkpoints/pvt_v2_b3.pth
```

Alternatively:

```powershell
$env:PVT_PRETRAINED_PATH="D:\weights\pvt_v2_b3.pth"
```

The repository intentionally ignores datasets, checkpoints, and prediction outputs.

## Train

```powershell
python train_attention_gate.py
python train_voxel_attention.py
python train_coordinate_attention.py
```

Useful overrides:

```powershell
$env:LUNA16_TASK_DIR="D:\datasets\Task03_lung"
$env:IMAGE_SIZE="192"
$env:BATCH_SIZE="16"
$env:NUM_EPOCHS="500"
$env:NUM_WORKERS="2"
```

Checkpoints are saved under `checkpoints/`:

- `checkpoint_attention_gate.pth`
- `checkpoint_voxel_attention.pth`
- `checkpoint_coordinate_attention.pth`

## Evaluate

```powershell
python eval_attention_gate.py
python eval_voxel_attention.py
python eval_coordinate_attention.py
```

Outputs are written to:

```text
results/attention_gate/
results/voxel_attention/
results/coordinate_attention/
```

Each output folder contains `mask/` predictions and `joint/` visual comparisons.

## References

Key references are listed in [docs/REFERENCES.md](docs/REFERENCES.md) and [docs/REFERENCES.bib](docs/REFERENCES.bib). Also see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing derivative code.
