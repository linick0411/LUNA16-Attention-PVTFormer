from pathlib import Path

import numpy as np

from scripts.preprocess_luna16.LUNA_mask_extraction import (
    _add_sphere_mask,
    _match_series_file,
    _physical_to_continuous_index,
)
from scripts.preprocess_luna16.convert_process_to_task03 import _slice_number
import SimpleITK as sitk

from scripts.preprocess_luna16.luna2D_mask import get_mask_depth_range, load_ct_with_truncation
from luna16_data import _media_by_stem, _slice_index


def test_series_uid_with_dots_matches_full_filename():
    uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.12345"
    paths = [f"/data/subset0/{uid}"]

    assert _match_series_file(paths, uid) == paths[0]


def test_sphere_mask_uses_physical_spacing():
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    _add_sphere_mask(
        mask,
        center_xyz=(4.0, 4.0, 8.0),
        diameter_mm=4.0,
        origin_xyz=(0.0, 0.0, 0.0),
        spacing_xyz=(1.0, 1.0, 2.0),
    )

    assert mask[4, 4, 4] == 1
    assert mask[4, 4, 6] == 1
    assert mask[5, 4, 4] == 1
    assert mask[6, 4, 4] == 0


def test_world_coordinate_conversion_respects_image_direction():
    direction = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    index = _physical_to_continuous_index(
        center_xyz=(-4.0, -6.0, 8.0),
        origin_xyz=(0.0, 0.0, 0.0),
        spacing_xyz=(1.0, 2.0, 4.0),
        direction_xyz=direction,
    )

    np.testing.assert_allclose(index, (4.0, 3.0, 2.0))


def test_single_foreground_slice_is_preserved_as_half_open_range():
    mask = np.zeros((5, 4, 4), dtype=np.uint8)
    mask[2, 1:3, 1:3] = 1

    assert get_mask_depth_range(mask) == (2, 3)
    assert get_mask_depth_range(np.zeros_like(mask)) == (None, None)


def test_ct_window_uses_fixed_hu_mapping_and_preserves_geometry(tmp_path):
    array = np.array([[[-1200.0, -1000.0, -200.0, 600.0, 800.0]]], dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 0.8, 1.5))
    image.SetOrigin((1.0, 2.0, 3.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "ct.mhd"
    sitk.WriteImage(image, str(path))

    result = load_ct_with_truncation(path, upper=600, lower=-1000)
    values = sitk.GetArrayFromImage(result)

    np.testing.assert_allclose(values[0, 0], (0.0, 0.0, 127.5, 255.0, 255.0), atol=1e-4)
    assert result.GetSpacing() == image.GetSpacing()
    assert result.GetOrigin() == image.GetOrigin()
    assert result.GetDirection() == image.GetDirection()


def test_unpadded_slice_names_sort_numerically():
    files = ["10.jpg", "2.jpg", "1.jpg"]

    assert sorted(files, key=_slice_number) == ["1.jpg", "2.jpg", "10.jpg"]


def test_mixed_numeric_and_text_slice_names_have_stable_sort_order():
    files = ["slice.jpg", "10.jpg", "2.jpg"]

    assert sorted(files, key=_slice_number) == ["2.jpg", "10.jpg", "slice.jpg"]


def test_loader_accepts_png_and_jpg_and_sorts_slice_suffix(tmp_path):
    for name in ("case_10.png", "case_2.jpg", "case_1.png"):
        (tmp_path / name).write_bytes(b"test")

    media = _media_by_stem(tmp_path)
    stems = sorted(media, key=_slice_index)

    assert stems == ["case_1", "case_2", "case_10"]
