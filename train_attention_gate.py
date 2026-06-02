from model3 import PVTFormer
from train_common import TrainingConfig, run_training


if __name__ == "__main__":
    run_training(
        PVTFormer,
        TrainingConfig(
            model_name="attention_gate",
            checkpoint_name="checkpoint_attention_gate.pth",
            train_log_name="train_log_attention_gate.txt",
            early_stopping_patience=50,
        ),
    )
