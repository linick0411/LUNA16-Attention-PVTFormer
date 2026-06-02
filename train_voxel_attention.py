from model2 import PVTFormer_Voxel
from train_common import TrainingConfig, run_training


if __name__ == "__main__":
    run_training(
        PVTFormer_Voxel,
        TrainingConfig(
            model_name="voxel_attention",
            checkpoint_name="checkpoint_voxel_attention.pth",
            train_log_name="train_log_voxel_attention.txt",
            early_stopping_patience=50,
        ),
    )
