from model4 import PVTFormer
from train_common import TrainingConfig, run_training


if __name__ == "__main__":
    run_training(
        PVTFormer,
        TrainingConfig(
            model_name="coordinate_attention",
            checkpoint_name="checkpoint_coordinate_attention.pth",
            train_log_name="train_log_coordinate_attention.txt",
            early_stopping_patience=20,
        ),
    )
