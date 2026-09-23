# Visual Demo

`demo_infer.py` runs one trained model on one LUNA16 case or one manually selected CT slice stack. It creates images that can be used in reports, slides, or live demonstrations.

## Quick Original-Project Demo

The shortest project-ending demonstration uses the recovered original Attention
Gate weight and automatically selects the largest annotated nodule slice from a
test case:

```bash
./scripts/run_quick_legacy_demo.sh
```

The default case is `nodule_592`. To select another case:

```bash
./scripts/run_quick_legacy_demo.sh data/Task03_lung_sphere-v1/nodule_573
```

This is a visual demonstration of the recovered original-project model on the
currently available `sphere-v1` data. It does not claim to recreate the missing
report-era split and pseudo-mask build.

## Outputs

For one selected center slice, the script writes:

- `*_input.png`: center CT slice.
- `*_probability.png`: model probability heatmap.
- `*_prediction.png`: binary predicted nodule mask.
- `*_overlay.png`: predicted mask overlaid on the CT slice.
- `*_joint.png`: side-by-side comparison. If a ground-truth mask is available, the joint image includes it.
- `*_summary.json`: selected paths, checkpoint, threshold, and per-slice Dice,
  IoU, Recall, and Precision when a reference mask is available.

## Attention Choices

Use one of:

```text
attention_gate
voxel_attention
coordinate_attention
```

`--weight-set legacy` selects the recovered original-project checkpoint for the
chosen model. `--auto-positive` selects the slice with the largest available
reference nodule mask. Omit it and pass `--slice-index` to demonstrate a chosen
positive or background-only slice.

Default checkpoints:

```text
checkpoints/checkpoint_attention_gate.pth
checkpoints/checkpoint_voxel_attention.pth
checkpoints/checkpoint_coordinate_attention.pth
```

You can also pass any checkpoint path with `--checkpoint`.

## Demo From A Task03_lung Case Folder

Example:

```powershell
python demo_infer.py `
  --attention voxel_attention `
  --checkpoint checkpoints/checkpoint_voxel_attention.pth `
  --case-dir data/Task03_lung/nodule_0 `
  --slice-index 5 `
  --output-dir demo_outputs
```

If `--slice-index` is omitted, the middle slice in `images/` is used.

The script automatically looks for the matching mask at:

```text
<case-dir>/masks/nodule/<center-slice-name>.jpg
```

## Demo From Three Manually Selected Slices

Example:

```powershell
python demo_infer.py `
  --attention attention_gate `
  --checkpoint checkpoints/checkpoint_attention_gate.pth `
  --prev data/Task03_lung/nodule_0/images/0_0004.jpg `
  --center data/Task03_lung/nodule_0/images/0_0005.jpg `
  --next data/Task03_lung/nodule_0/images/0_0006.jpg `
  --mask data/Task03_lung/nodule_0/masks/nodule/0_0005.jpg `
  --output-dir demo_outputs
```

If only `--center` is provided, the center slice is duplicated as previous and next. This is useful for a quick test, but a real 2.5D demonstration should provide neighboring slices or use `--case-dir`.

## Pipeline Test Without A Checkpoint

For checking that image loading and output writing work:

```powershell
python demo_infer.py `
  --attention voxel_attention `
  --case-dir data/Task03_lung/nodule_0 `
  --allow-random-weights
```

This produces images, but they are not meaningful model results. Use this only to test the demo pipeline.

## Suggested Report Usage

Use `*_joint.png` as the main visual result:

```text
Input CT | Ground Truth | Probability Heatmap | Prediction Mask | Overlay
```

This is easier to explain than raw metric numbers because it shows what the model actually predicted on one CT slice.
