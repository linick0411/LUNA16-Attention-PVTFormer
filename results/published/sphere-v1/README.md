# Published sphere-v1 evidence

This directory contains the compact evidence needed to review the corrected
five-model reproduction without committing LUNA16 data, prediction masks, raw
logs, or large model weights.

- `summary.csv`: fixed-contract test comparison for all five models.
- `metrics/*.json`: per-model test metrics and runtime metadata.
- `training/*.csv`: the run containing each model's highest recorded validation F1.
- `training/runs/*.csv`: every retained run, including the Coordinate Attention warm restart.
- `training-summary.csv`: best validation F1 and final losses.
- `data-validation.json`: pairing, binary-mask, split, and z-order provenance audit.
- `aggregation-audit.json`: micro versus per-slice F1 diagnostic.
- `checkpoint-manifest.json`: exact checkpoint filenames, sizes, and SHA-256 hashes.

The targets are spherical pseudo-masks generated from LUNA16 center coordinates
and diameter annotations. They are not expert-drawn clinical contours. The
checkpoint files exceed GitHub's normal 100 MB file limit and are therefore
identified by hash rather than stored in this repository.
