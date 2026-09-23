# Dataset And Preprocessing

## Source Dataset

This project uses LUNA16, a lung nodule detection challenge derived from the LIDC-IDRI public chest CT collection. LUNA16 provides CT scans in `.mhd/.raw` format and nodule annotations in `annotations.csv`.

The repository does not redistribute LUNA16/LIDC-IDRI data. Download the dataset from the official challenge/data source and keep it outside git.

The complete official release is about 61.8 GiB compressed. A resumable downloader with official Zenodo checksums is included:

```powershell
python scripts/download_luna16.py --output-root data/luna16 --subsets all
```

Use `--list` to inspect the manifest without downloading, or `--subsets 0,1` to prepare only selected subsets for a pipeline smoke test. Complete experiments require all ten subsets.

## Important Label Meaning

LUNA16 is originally a lung nodule detection dataset, not a pixel-perfect segmentation dataset. The preprocessing scripts build coarse 2D nodule masks from `annotations.csv` center coordinates and `diameter_mm`.

That means the target mask is an approximation of each annotated nodule region. It is suitable for student/research experiments, but it is not the same as expert contour segmentation.

The default generator uses a sphere derived from the annotated physical diameter. This is a closer geometric approximation than the project's older cuboid method, but it remains a generated pseudo-mask rather than an expert contour.

## Pipeline Used Here

### 1. Generate 3D Masks From LUNA16 Annotations

Script:

```powershell
python scripts/preprocess_luna16/LUNA_mask_extraction.py --luna-root data/luna16
```

Input:

```text
data/luna16/
  annotations.csv
  subset0/*.mhd + *.raw
  subset1/*.mhd + *.raw
  ...
  subset9/*.mhd + *.raw
```

Method:

- Read each CT series with SimpleITK.
- Match rows in `annotations.csv` by `seriesuid`.
- Convert world coordinates `(coordX, coordY, coordZ)` into voxel coordinates using CT origin and spacing.
- Convert `diameter_mm` into voxel radius per axis.
- Fill a spherical pseudo-mask using the annotated center and diameter in physical millimetres. Use `--mask-shape box` only to reproduce the legacy cuboid labels.
- Save one `_segmentation.mhd` mask per CT volume.

Output:

```text
data/luna16/mask/subset0/*_segmentation.mhd
...
data/luna16/mask/subset9/*_segmentation.mhd
```

### 2. Convert 3D CT And Mask Volumes Into 2D Slices

Script:

```powershell
python scripts/preprocess_luna16/luna2D_mask.py --luna-root data/luna16
```

Method:

- Clip CT intensity with HU range `[-1000, 600]`.
- Map the fixed HU window `[-1000, 600]` linearly to `[0, 255]` so the same HU value has the same intensity in every scan.
- If slice spacing on the z-axis is larger than 1.0 mm, resample CT and mask to z-spacing 1.0 mm.
- Find the half-open z-depth range containing foreground mask pixels, including nodules that occupy only one slice.
- Keep the foreground range plus 13 extra slices before and after.
- Save CT slices as lossless PNG under `process/image` by default. Legacy JPG output remains available with `--image-format jpg`.
- Always save binary nodule masks as lossless PNG under `process/mask`.

Output:

```text
data/luna16/process/
  image/0/*.png
  mask/0/*.png
  image/1/*.png
  mask/1/*.png
```

### 3. Convert Process Folders Into Task03_lung

Script:

```powershell
python scripts/preprocess_luna16/convert_process_to_task03.py `
  --process-root data/luna16/process `
  --output-root data/Task03_lung
```

Output:

```text
data/Task03_lung/
  nodule_0/
    images/0_0000.png
    masks/nodule/0_0000.png
  nodule_1/
    images/1_0000.png
    masks/nodule/1_0000.png
```

## Alternative Direct Conversion

`convert_luna_to_jpg.py` directly extracts slices from CT volumes and generated masks into `Task03_lung`. The process-based path above is closer to the current project history.

```powershell
python scripts/preprocess_luna16/convert_luna_to_jpg.py `
  --luna-root data/luna16 `
  --output-root data/Task03_lung
```

## Training Data Loader

`luna16_data.py` loads:

```text
images/*.(png|jpg)
masks/nodule/*.(png|jpg)
```

For each center slice, the model input is a 2.5D stack:

```text
previous slice + center slice + next slice
```

The tensor shape is:

```text
image: (B, 3, 1, 192, 192)
mask:  (B, 1, 192, 192)
```

Image and mask files are paired by their exact filename stem, preventing a missing file from shifting every later pair. Masks are resized with nearest-neighbor interpolation and thresholded back to strict binary values.

Before training, validate the final folder structure and exact pairing:

```powershell
python -m scripts.validate_task03 --data-dir data/Task03_lung --check-pixels --output results/data_validation.json
```

The pixel check also verifies that every image and mask is readable, has the same shape, and that masks contain only binary values.

To verify the full z-order provenance from unpadded process filenames into the
zero-padded Task03 names, include the process directory:

```powershell
python -m scripts.validate_task03 `
  --data-dir data/Task03_lung `
  --process-root data/luna16/process `
  --check-pixels `
  --output results/data_validation.json
```

This checks every case for contiguous numeric source slices, manifest counts,
UID consistency, and the rule `original_z = first_kept_slice + exported_index`.
Three deterministic cases are also compared byte-for-byte from source image and
mask files to their renamed Task03 copies.

The preprocessing flow writes `series_manifest.csv` and `case_manifest.csv`. These preserve the mapping from every `nodule_*` folder back to its official LUNA16 scan UID and subset. The corrected pipeline assigns a deterministic 80/10/10 scan-level split, stratified by the ten official subsets, and records the assignment in `case_manifest.csv`. This makes the split auditable and avoids concentrating test cases in the first downloaded subset.

## Project Dataset Snapshot

The prepared remote dataset previously inspected for this project had:

- Root: `/home/project/Desktop/Task03_lung`
- Folders: `nodule_0` through `nodule_600`
- Images: 45,461 JPG files
- Masks: 45,461 JPG files
- Test folders present: 30/30
- Validation folders present: 30/30
