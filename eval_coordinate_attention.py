from pathlib import Path

from eval_common import EvaluationConfig, run_evaluation
from model4 import PVTFormer


if __name__ == "__main__":
    run_evaluation(
        PVTFormer,
        EvaluationConfig(
            model_name="coordinate_attention",
            checkpoint_name="checkpoint_coordinate_attention.pth",
            output_dir=Path("results/coordinate_attention"),
        ),
    )
