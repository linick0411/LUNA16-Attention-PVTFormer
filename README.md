# LUNA16 Attention PVTFormer

This repository organizes a 2D PVTFormer baseline, a plain 2.5D control, and three 2.5D attention variants for the LUNA16/LIDC-IDRI dataset:

| Entry point | Model file | Attention variant |
| --- | --- | --- |
| `train_baseline.py` / `eval_baseline.py` | `model_baseline.py` | Original center-slice 2D PVTFormer baseline |
| `train_concat_25d.py` / `eval_concat_25d.py` | `model_concat_25d.py` | Three-slice concatenation control without added attention |
| `train_attention_gate.py` / `eval_attention_gate.py` | `model3.py` | Attention Gate on decoder skip connections |
| `train_voxel_attention.py` / `eval_voxel_attention.py` | `model2.py` | Voxel Attention fusion across neighboring CT slices |
| `train_coordinate_attention.py` / `eval_coordinate_attention.py` | `model4.py` | Coordinate Attention fusion across neighboring CT slices |

The code is adapted from the PVTFormer architecture proposed for CT liver segmentation and extended here for LUNA16 lung nodule segmentation experiments. It is research code only and is not a clinical diagnostic tool.

## Project Conclusion

The original project run remains the primary report result. Under its
slice-level evaluation protocol, Attention Gate produced the highest F1/Dice
(0.7994), Precision (0.9555), Jaccard/IoU (0.7827), and F2 (0.7983), while
Coordinate Attention produced the highest Recall (0.8208). These results
suggest that neighboring CT slices and attention-based feature fusion can help
PVTFormer segment LUNA16 lung-nodule targets.

Recovered legacy checkpoints have since been loaded successfully and audited.
Because background-only CT slices are part of the real screening workflow, the
legacy slice-level score remains useful as an overall slice-performance view.
The corrected reproduction additionally reports foreground micro metrics and
negative-slice false positives, which answer different questions and should not
be compared directly with the presentation table. See
[docs/QUICK_PROJECT_WRAP_UP.md](docs/QUICK_PROJECT_WRAP_UP.md) for the concise
report-ready conclusion and [docs/LEGACY_WEIGHT_COMPARISON.md](docs/LEGACY_WEIGHT_COMPARISON.md)
for the checkpoint audit.

## What Is Included

- A reproducible 2D upstream baseline and a plain 2.5D control using previous, center, and next CT slices.
- Three attention variants with separate train/eval entry points.
- LUNA16 preprocessing scripts for converting `.mhd` CT volumes and `annotations.csv` into lossless PNG `Task03_lung` folders.
- Evaluation metrics: Jaccard/IoU, F1/Dice, Recall, Precision, Accuracy, F2, Hausdorff distance, AUC, and FPS.
- Documentation for dataset preparation, model differences, reproducibility, and citation/third-party attribution.

## Dataset

The dataset is not included in this repository. Prepare it under:

```text
data/Task03_lung/
  nodule_0/
    images/*.(png|jpg)
    masks/nodule/*.(png|jpg)
  nodule_1/
    images/*.(png|jpg)
    masks/nodule/*.(png|jpg)
```

Or point to an existing folder:

```powershell
$env:LUNA16_TASK_DIR="D:\datasets\Task03_lung"
```

The historical prepared dataset contained `nodule_0` through `nodule_600`, with 45,461 image JPGs and 45,461 mask JPGs. When no manifest is present, the loader preserves its legacy split policy:

- Test: `nodule_0` to `nodule_29`
- Validation: `nodule_30` to `nodule_59`
- Train: all remaining `nodule_*` folders

The corrected reproduction writes `case_manifest.csv` and uses a fixed, subset-stratified 80/10/10 scan-level split instead.

See [docs/DATASET.md](docs/DATASET.md) for the full preprocessing flow.

Validate exact image/mask pairing and pixel contents before training:

```powershell
python -m scripts.validate_task03 --data-dir data/Task03_lung --check-pixels --output results/data_validation.json
```

The original presentation numbers, recovered-checkpoint evidence, and corrected
reproduction are documented in [docs/RESULTS.md](docs/RESULTS.md). A file-level
account of reused upstream code and this project's additions is in
[docs/UPSTREAM_COMPARISON.md](docs/UPSTREAM_COMPARISON.md).

Compact machine-readable evidence from the completed reproduction is committed
under [`results/published/sphere-v1`](results/published/sphere-v1). It includes
per-model metrics, merged training histories, the data and z-order audit, and
SHA-256 identifiers for the locally retained checkpoints. Regenerate it with:

```powershell
python -m scripts.publish_results `
  --validation results/reproduction-sphere-v1/data_validation_with_provenance.json
```

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

The repository intentionally ignores datasets, checkpoints, and prediction outputs. Comparable experiments require these pretrained weights; set `ALLOW_RANDOM_BACKBONE=1` only for an explicitly non-comparable smoke test.

## Train

```powershell
python train_attention_gate.py
python train_voxel_attention.py
python train_coordinate_attention.py
python train_baseline.py
```

Run the complete fair comparison sequentially with one command:

```powershell
python scripts/run_experiments.py --stage all --models all
```

Use `--dry-run` to inspect the exact order without starting training.

For a clean end-to-end corrected reproduction after all official subsets are present:

```powershell
python scripts/reproduce_all.py --luna-root data/luna16
```

This verifies all 888 official scans, builds versioned spherical pseudo-masks and lossless PNG data without deleting existing folders, validates every image/mask pair, then trains and evaluates all controls and attention variants. Best checkpoints are kept under `checkpoints/reproduction-sphere-v1/`; training curves and the final CSV/Markdown comparison tables are generated automatically.

Useful overrides:

```powershell
$env:LUNA16_TASK_DIR="D:\datasets\Task03_lung"
$env:IMAGE_SIZE="192"
$env:BATCH_SIZE="16"
$env:NUM_EPOCHS="500"
$env:NUM_WORKERS="2"
$env:USE_AMP="1"
```

## Hardware Note

The default `192x192` input size is a reproducible setting chosen for the available training workstation, not a claim that larger sizes are invalid. The project machine used an NVIDIA GeForce RTX 4060 Ti with 16 GB VRAM, an Intel Core i3-14100F CPU, and 15 GiB system RAM. Larger settings such as `256x256` may require reducing batch size or using a GPU with more VRAM to avoid CUDA out-of-memory errors.

Checkpoints are saved under `checkpoints/`:

- `checkpoint_attention_gate.pth`
- `checkpoint_voxel_attention.pth`
- `checkpoint_coordinate_attention.pth`
- `checkpoint_baseline.pth`
- `checkpoint_concat_25d.pth`

Every training run also writes a timestamped CSV history and JSON environment record under `logs/`. Generate report-ready loss and F1 curves with:

```powershell
python scripts/plot_training_curves.py logs/history_attention_gate_<run-id>.csv
```

## Evaluate

```powershell
python eval_attention_gate.py
python eval_voxel_attention.py
python eval_coordinate_attention.py
python eval_baseline.py
```

Outputs are written to:

```text
results/attention_gate/
results/voxel_attention/
results/coordinate_attention/
results/baseline/
results/concat_25d/
```

Each output folder contains `mask/` predictions and `joint/` visual comparisons.
It also contains `metrics.json`, including the corrected metrics, synchronized GPU inference time, and the environment used for evaluation.

## Visual Demo

For the shortest original-project demonstration, run:

```bash
./scripts/run_quick_legacy_demo.sh
```

This loads the recovered Attention Gate checkpoint, selects a representative
positive slice, and writes one labeled comparison image plus the individual
prediction artifacts.

Use `demo_infer.py` to show one actual model prediction as report-ready images:

```powershell
python demo_infer.py `
  --attention voxel_attention `
  --checkpoint checkpoints/checkpoint_voxel_attention.pth `
  --case-dir data/Task03_lung/nodule_0 `
  --slice-index 5 `
  --output-dir demo_outputs
```

The demo writes the center CT slice, probability heatmap, binary prediction mask, overlay, and a joint comparison image. See [docs/DEMO.md](docs/DEMO.md) for all options.

## References

Key references are listed in [docs/REFERENCES.md](docs/REFERENCES.md) and [docs/REFERENCES.bib](docs/REFERENCES.bib). Also see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing derivative code.
