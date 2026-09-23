from model_baseline import PVTFormerBaseline
from train_common import TrainingConfig, run_training


if __name__ == "__main__":
    run_training(
        PVTFormerBaseline,
        TrainingConfig(
            model_name="pvtformer_baseline_2d",
            checkpoint_name="checkpoint_baseline.pth",
            train_log_name="train_log_baseline.txt",
            early_stopping_patience=50,
            use_25d=False,
        ),
    )
