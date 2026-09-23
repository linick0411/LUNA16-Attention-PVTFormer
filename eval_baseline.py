import os
from pathlib import Path

from eval_common import EvaluationConfig, run_evaluation
from model_baseline import PVTFormerBaseline


if __name__ == "__main__":
    run_evaluation(
        PVTFormerBaseline,
        EvaluationConfig(
            model_name="pvtformer_baseline_2d",
            checkpoint_name="checkpoint_baseline.pth",
            output_dir=Path(os.environ.get("RESULTS_DIR", "results")) / "baseline",
            use_25d=False,
        ),
    )
