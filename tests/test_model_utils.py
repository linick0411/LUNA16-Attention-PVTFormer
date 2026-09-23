import torch

from model_utils import load_pvtv2_b3_weights


def test_pretrained_loader_loads_compatible_tensor(tmp_path, monkeypatch):
    backbone = torch.nn.Linear(2, 1)
    expected = {
        "weight": torch.tensor([[2.0, 3.0]]),
        "bias": torch.tensor([4.0]),
    }
    path = tmp_path / "weights.pth"
    torch.save(expected, path)
    monkeypatch.setenv("PVT_PRETRAINED_PATH", str(path))

    loaded_path = load_pvtv2_b3_weights(backbone)

    assert loaded_path == path
    assert torch.equal(backbone.weight.detach(), expected["weight"])
    assert torch.equal(backbone.bias.detach(), expected["bias"])


def test_missing_pretrained_weights_fail_unless_explicitly_allowed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PVT_PRETRAINED_PATH", raising=False)
    monkeypatch.delenv("ALLOW_RANDOM_BACKBONE", raising=False)

    try:
        load_pvtv2_b3_weights(torch.nn.Linear(2, 1))
    except FileNotFoundError as exc:
        assert "ALLOW_RANDOM_BACKBONE=1" in str(exc)
    else:
        raise AssertionError("Missing weights should not silently start a comparable experiment")
