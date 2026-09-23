import torch

from scripts.compare_legacy_weights import _remap_baseline


def test_remap_baseline_renames_only_legacy_module_paths():
    tensor = torch.tensor([1.0])
    legacy = {
        "backbone.block.weight": tensor,
        "d1.r1.conv.weight": tensor,
        "u3.r1.shortcut.weight": tensor,
        "r1.conv.weight": tensor,
        "y.weight": tensor,
    }

    remapped = _remap_baseline(legacy)

    assert set(remapped) == {
        "backbone.block.weight",
        "d1.residual.conv.weight",
        "u3.residual.shortcut.weight",
        "fuse.conv.weight",
        "output.weight",
    }
    assert all(value is tensor for value in remapped.values())
