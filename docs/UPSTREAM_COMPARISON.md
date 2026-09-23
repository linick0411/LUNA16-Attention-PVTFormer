# Comparison With The Original PVTFormer Repository

## Upstream Basis

This project starts from Debesh Jha and collaborators' PVTFormer implementation for liver CT segmentation:

- Upstream repository: `https://github.com/DebeshJha/PVTFormer`
- PVTv2 backbone source: `https://github.com/whai362/PVT`

The PVTv2 encoder and the hierarchical PVTFormer decoder are the architectural foundation. They should be cited and described as reused components rather than claimed as original work.

## What This Project Adds

| Area | Upstream PVTFormer | This project |
| --- | --- | --- |
| Medical task | LiTS liver segmentation | LUNA16 pulmonary-nodule experiments |
| Input | One 2D RGB/three-channel slice | Previous, center, and next grayscale CT slices for the 2.5D variants |
| Labels | Liver segmentation masks | Coarse masks constructed from LUNA16 nodule center and diameter annotations |
| Control model | Original 2D PVTFormer | Reproducible `model_baseline.py` using the shared pipeline |
| 2.5D control | None | Plain three-slice concatenation without added attention |
| Added models | None | Voxel Attention, decoder Attention Gate, and Coordinate Attention variants |
| Data pipeline | LiTS-specific scripts | Official LUNA16 downloader, annotation-to-mask conversion, slice extraction, and scan-level loading |
| Experiment tooling | Separate original train/test scripts | Shared train/evaluation configuration, visual outputs, corrected metrics, and tests |

## File-Level Mapping

- `pvtv2.py`: retained PVTv2 backbone code derived from upstream.
- `model_baseline.py`: upstream PVTFormer model adapted to optional pretrained-weight loading and the shared experiment interface.
- `model_concat_25d.py`: adds a plain three-slice concatenation control to separate the value of neighboring slices from attention effects.
- `model2.py`: adds feature-level attention across three neighboring slices.
- `model3.py`: adds three-slice feature fusion and Attention Gate decoder skips.
- `model4.py`: adds Coordinate Attention fusion across three-slice features.
- `luna16_data.py`: adds the LUNA16/Task03_lung loader and neighbor-slice assembly.
- `scripts/preprocess_luna16/`: adds LUNA16 annotation conversion and 2D preprocessing.
- `train_common.py` and `eval_common.py`: add shared experiment execution so model comparisons use the same settings.

## Accurate Contribution Statement

A concise description for a report or interview is:

> I used the public PVTFormer architecture as the base network, transferred the task from liver CT segmentation to LUNA16 pulmonary-nodule analysis, built a 2.5D neighboring-slice data pipeline, and implemented three attention-integration strategies under a shared evaluation workflow.

This distinguishes task transfer, data engineering, model integration, and comparative experimentation from the upstream backbone itself.
