from pathlib import Path

from eval_common import EvaluationConfig, run_evaluation
from model3 import PVTFormer


if __name__ == "__main__":
    run_evaluation(
        PVTFormer,
        EvaluationConfig(
            model_name="attention_gate",
            checkpoint_name="checkpoint_attention_gate.pth",
            output_dir=Path("results/attention_gate"),
        ),
    )
