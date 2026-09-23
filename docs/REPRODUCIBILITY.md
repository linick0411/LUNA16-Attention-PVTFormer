# Reproducibility

## Environment

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The current reproduction environment is an isolated WSL project virtual environment. Validation was run with:

```powershell
python -m pytest tests -q
python -m py_compile <changed repository Python files>
```

The verified runtime is Python 3.10.12, PyTorch 2.14.0 with CUDA 13.0, and an NVIDIA GeForce RTX 5070 Ti with 15.92 GiB reported memory. CUDA tensor execution and forward passes for the baseline and all three attention variants completed successfully. The corrected `sphere-v1` preprocessing, five-model training, and fixed-contract evaluation have completed; compact evidence is published under `results/published/sphere-v1`.

## Training Hardware

The historical project workstation had:

- GPU: NVIDIA GeForce RTX 4060 Ti
- GPU memory: 16,380 MiB reported by `nvidia-smi`
- CPU: Intel Core i3-14100F
- CPU cores/threads: 4 cores / 8 threads
- System memory: 15 GiB RAM, 23 GiB swap
- OS: Ubuntu 24.04.3 LTS
- NVIDIA driver: 580.126.20
- CUDA version reported by `nvidia-smi`: 13.0

The default `IMAGE_SIZE=192` and `BATCH_SIZE=16` are hardware-aware defaults. They were selected to keep the experiments runnable on this workstation and to avoid CUDA out-of-memory failures. If using `256x256` or larger input resolution, reduce `BATCH_SIZE` first or use a GPU with more VRAM.

The rebuilt local environment uses the RTX 5070 Ti described above. Every new training run writes its exact Python, PyTorch, CUDA, and device settings into a JSON run record.

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

Comparable training stops with a clear error if pretrained weights are missing or incompatible. Set `ALLOW_RANDOM_BACKBONE=1` only for an explicitly non-comparable smoke test.

## Commands

Train:

```powershell
python train_attention_gate.py
python train_voxel_attention.py
python train_coordinate_attention.py
python train_baseline.py
```

Evaluate:

```powershell
python eval_attention_gate.py
python eval_voxel_attention.py
python eval_coordinate_attention.py
python eval_baseline.py
```

Common overrides:

```powershell
$env:IMAGE_SIZE="192"
$env:BATCH_SIZE="16"
$env:NUM_EPOCHS="500"
$env:NUM_WORKERS="2"
$env:CHECKPOINT_DIR="checkpoints"
$env:USE_AMP="1"
```

CUDA training and evaluation use automatic mixed precision by default to reduce memory use and runtime. Set `USE_AMP=0` for a full-float diagnostic run. The selected mode is recorded in every run and evaluation JSON file.

## Expected Outputs

Training:

```text
logs/train_log_attention_gate.txt
logs/train_log_voxel_attention.txt
logs/train_log_coordinate_attention.txt
checkpoints/checkpoint_attention_gate.pth
checkpoints/checkpoint_voxel_attention.pth
checkpoints/checkpoint_coordinate_attention.pth
checkpoints/checkpoint_baseline.pth
```

Evaluation:

```text
results/attention_gate/mask
results/attention_gate/joint
results/voxel_attention/mask
results/voxel_attention/joint
results/coordinate_attention/mask
results/coordinate_attention/joint
results/baseline/mask
results/baseline/joint
```

## Metric Notes

- Jaccard, F1/Dice, Recall, Precision, Accuracy, and F2 use dataset-level micro aggregation over TP/FP/TN/FN. This prevents empty background slices from receiving free perfect overlap scores and inflating the mean.
- AUC is computed pixel-wise from predicted probabilities and binary mask labels.
- HD is the symmetric Hausdorff distance between foreground-pixel coordinates and
  is reported in resized-image pixels. Empty/empty slices are excluded; a false-positive-only slice receives the resized image diagonal as its penalty. The evaluator records the number of contributing slices.
- The evaluator also records the false-positive rate among ground-truth-negative slices.
- Because LUNA16 labels are converted from center/diameter annotations into coarse masks, metric values should be interpreted as experiment-comparison signals, not clinical validation.
