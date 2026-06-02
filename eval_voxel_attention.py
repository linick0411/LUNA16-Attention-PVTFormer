from pathlib import Path

from eval_common import EvaluationConfig, run_evaluation
from model2 import PVTFormer_Voxel


if __name__ == "__main__":
    run_evaluation(
        PVTFormer_Voxel,
        EvaluationConfig(
            model_name="voxel_attention",
            checkpoint_name="checkpoint_voxel_attention.pth",
            output_dir=Path("results/voxel_attention"),
        ),
    )
