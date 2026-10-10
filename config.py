"""
Configuration file for Medical Image Segmentation Benchmark
Optimized for Google Colab with NVIDIA A100 GPU (and compatible with CPU/standard GPUs)
"""

import os
import torch

class Config:
    # Experiment identifier
    EXPERIMENT_NAME = "EffiBoundNet_Medical_Segmentation"
    
    # Dataset Configuration
    DATASET_ROOT = os.path.join(os.path.dirname(__file__), "data", "Dataset_BUSI_with_GT")
    IMG_SIZE = (256, 256)
    NUM_CLASSES = 1  # Binary lesion segmentation (0: Background, 1: Lesion)
    IN_CHANNELS = 3
    
    # Train / Val / Test Split
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RANDOM_SEED = 42
    
    # Training Hyperparameters (Default optimized for A100 GPU)
    BATCH_SIZE = 16  # For A100, can be set to 16, 32, or 64
    NUM_WORKERS = 4  # Set to 2-4 on Colab
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER = "CosineAnnealingWarmRestarts"
    T_0 = 10
    T_MULT = 2
    MIN_LR = 1e-6
    
    # Mixed Precision & Acceleration
    USE_AMP = True  # Automatic Mixed Precision (torch.cuda.amp) for A100 Tensor Cores
    PIN_MEMORY = True
    
    # Loss Weights
    DICE_WEIGHT = 1.0
    BCE_WEIGHT = 1.0
    BOUNDARY_WEIGHT = 0.5
    AUX_BOUNDARY_WEIGHT = 0.3
    
    # Models to benchmark
    BENCHMARK_MODELS = [
        "FCN",
        "UNet",
        "AttUNet",
        "DeepLabV3Plus",
        "TransUNet",
        "EffiBoundNet"  # Proposed Method
    ]
    
    # Output Directories
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
    FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

    @classmethod
    def get_device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
