import os
from pathlib import Path

import torch


def load_pvtv2_b3_weights(backbone):
    """Load PVTv2-B3 ImageNet weights and reject silent random initialization."""
    candidates = []
    env_path = os.environ.get("PVT_PRETRAINED_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([Path("checkpoints/pvt_v2_b3.pth"), Path("pvt_v2_b3.pth")])

    weight_path = next((path for path in candidates if path.exists() and path.is_file()), None)
    if weight_path is None:
        if os.environ.get("ALLOW_RANDOM_BACKBONE") == "1":
            print("Warning: PVTv2-B3 weights not found; ALLOW_RANDOM_BACKBONE=1 permits random initialization.")
            return None
        raise FileNotFoundError(
            "PVTv2-B3 pretrained weights were not found. Place them at "
            "checkpoints/pvt_v2_b3.pth, set PVT_PRETRAINED_PATH, or explicitly set "
            "ALLOW_RANDOM_BACKBONE=1 for a non-comparable smoke test."
        )

    try:
        try:
            save_model = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            save_model = torch.load(weight_path, map_location="cpu")
        if isinstance(save_model, dict) and isinstance(save_model.get("state_dict"), dict):
            save_model = save_model["state_dict"]
        if not isinstance(save_model, dict):
            raise TypeError(f"Expected a state-dict mapping, got {type(save_model).__name__}")

        model_dict = backbone.state_dict()
        state_dict = {
            key: value
            for key, value in save_model.items()
            if key in model_dict and hasattr(value, "shape") and value.shape == model_dict[key].shape
        }
        if not state_dict:
            raise RuntimeError("No compatible backbone tensors were found in the weights file")
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)
        print(f"Loaded {len(state_dict)} PVTv2-B3 tensors: {weight_path}")
        return weight_path
    except Exception as exc:
        raise RuntimeError(f"Failed to load PVTv2-B3 weights from {weight_path}: {exc}") from exc
