from model_concat_25d import PVTFormerConcat25D
from train_common import TrainingConfig, run_training


if __name__ == "__main__":
    run_training(
        PVTFormerConcat25D,
        TrainingConfig(
            model_name="concat_25d_control",
            checkpoint_name="checkpoint_concat_25d.pth",
            train_log_name="train_log_concat_25d.txt",
            early_stopping_patience=50,
        ),
    )
