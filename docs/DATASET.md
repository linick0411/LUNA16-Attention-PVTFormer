# Dataset And Preprocessing

## Source Dataset

This project uses LUNA16, a lung nodule detection challenge derived from the LIDC-IDRI public chest CT collection. LUNA16 provides CT scans in `.mhd/.raw` format and nodule annotations in `annotations.csv`.

The repository does not redistribute LUNA16/LIDC-IDRI data. Download the dataset from the official challenge/data source and keep it outside git.

## Important Label Meaning

LUNA16 is originally a lung nodule detection dataset, not a pixel-perfect segmentation dataset. The preprocessing scripts build coarse 2D nodule masks from `annotations.csv` center coordinates and `diameter_mm`.

That means the target mask is an approximation of each annotated nodule region. It is suitable for student/research experiments, but it is not the same as expert contour segmentation.

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
- Fill a cuboid nodule mask around the converted center.
- Save one `_segmentation.mhd` mask per CT volume.

Output:

```text
data/luna16/mask/subset0/*_segmentation.mhd
...
data/luna16/mask/subset9/*_segmentation.mhd
```

### 2. Convert 3D CT And Mask Volumes Into 2D JPG Slices

Script:

```powershell
python scripts/preprocess_luna16/luna2D_mask.py --luna-root data/luna16
```

Method:

- Clip CT intensity with HU range `[-1000, 600]`.
- Rescale clipped CT values to `[0, 255]`.
- If slice spacing on the z-axis is larger than 1.0 mm, resample CT and mask to z-spacing 1.0 mm.
- Find the z-depth range containing foreground mask pixels.
- Keep the foreground range plus 13 extra slices before and after.
- Save CT slices as JPG under `process/image`.
- Save binary nodule masks as JPG under `process/mask`.

Output:

```text
data/luna16/process/
  image/0/*.jpg
  mask/0/*.jpg
  image/1/*.jpg
  mask/1/*.jpg
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
    images/0_0000.jpg
    masks/nodule/0_0000.jpg
  nodule_1/
    images/1_0000.jpg
    masks/nodule/1_0000.jpg
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
images/*.jpg
masks/nodule/*.jpg
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

Masks are resized with nearest-neighbor interpolation and thresholded back to strict binary values.

## Project Dataset Snapshot

The prepared remote dataset inspected for this project had:

- Root: `/home/project/Desktop/Task03_lung`
- Folders: `nodule_0` through `nodule_600`
- Images: 45,461 JPG files
- Masks: 45,461 JPG files
- Test folders present: 30/30
- Validation folders present: 30/30
