import pandas as pd
import pytest

from scripts.reproduce_all import verify_luna_ready


def test_verify_luna_ready_counts_official_pairs(tmp_path):
    for index in range(10):
        (tmp_path / f"subset{index}").mkdir()
    (tmp_path / "subset0" / "scan.mhd").write_text("header", encoding="utf-8")
    (tmp_path / "subset0" / "scan.raw").write_bytes(b"pixels")
    pd.DataFrame([{"seriesuid": "scan"}]).to_csv(tmp_path / "annotations.csv", index=False)

    report = verify_luna_ready(tmp_path, expected_scans=1)

    assert report == {"scan_count": 1, "raw_count": 1, "annotated_scan_count": 1}


def test_verify_luna_ready_rejects_missing_subset(tmp_path):
    pd.DataFrame([{"seriesuid": "scan"}]).to_csv(tmp_path / "annotations.csv", index=False)

    with pytest.raises(RuntimeError, match="Missing extracted"):
        verify_luna_ready(tmp_path, expected_scans=1)
