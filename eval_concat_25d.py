import os
from pathlib import Path

from eval_common import EvaluationConfig, run_evaluation
from model_concat_25d import PVTFormerConcat25D


if __name__ == "__main__":
    run_evaluation(
        PVTFormerConcat25D,
        EvaluationConfig(
            model_name="concat_25d_control",
            checkpoint_name="checkpoint_concat_25d.pth",
            output_dir=Path(os.environ.get("RESULTS_DIR", "results")) / "concat_25d",
        ),
    )
