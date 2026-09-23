from scripts.preprocess_luna16.convert_process_to_task03 import _assign_scan_splits


def test_scan_split_is_deterministic_and_stratified_by_subset():
    rows = [
        {
            "case_name": f"nodule_{index}",
            "source_series_index": str(index),
            "seriesuid": f"uid-{index}",
            "subset": str(index // 10),
            "slice_count": 1,
        }
        for index in range(20)
    ]

    first = _assign_scan_splits([dict(row) for row in rows], seed=42)
    second = _assign_scan_splits([dict(row) for row in rows], seed=42)

    assert [row["split"] for row in first] == [row["split"] for row in second]
    for subset in ("0", "1"):
        splits = [row["split"] for row in first if row["subset"] == subset]
        assert splits.count("test") == 1
        assert splits.count("validation") == 1
        assert splits.count("train") == 8
