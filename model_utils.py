import os
from pathlib import Path

import torch


def load_pvtv2_b3_weights(backbone):
    """Load optional PVTv2-B3 ImageNet weights without requiring them in git."""
    candidates = [
        Path(os.environ.get("PVT_PRETRAINED_PATH", "")),
        Path("checkpoints/pvt_v2_b3.pth"),
        Path("pvt_v2_b3.pth"),
    ]
    weight_path = next((path for path in candidates if str(path) and path.exists()), None)
    if weight_path is None:
        print("Warning: PVTv2-B3 pretrained weights not found. Initializing backbone randomly.")
        return

    try:
        save_model = torch.load(weight_path, map_location="cpu")
        model_dict = backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict}
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
        print(f"Loaded PVTv2-B3 weights: {weight_path}")
    except Exception as exc:
        print(f"Warning: failed to load PVTv2-B3 weights from {weight_path}: {exc}")
