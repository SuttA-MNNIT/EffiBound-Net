# EffiBound-Net: Frequency-Guided and Boundary-Aware Network for Medical Image Segmentation

EffiBound-Net is an ultra-lightweight, frequency-guided, and boundary-aware neural architecture designed for real-time and robust medical image segmentation (e.g., breast ultrasound lesion delineation). 

The architecture decouples high-frequency acoustic boundaries from low-frequency structural context, utilizing depthwise separable receptive field blocks and frequency operators to achieve fine boundary fidelity with low computational footprint.

---

## Key Architectural Modules

1. **Multi-Scale Depthwise Receptive Field Block (MS-DRFB)**:
   Deploys parallel depthwise separable convolutions across multiple dilation factors ($d \in \{1, 2, 4\}$) followed by Squeeze-and-Excitation channel recalibration to expand multi-scale receptive field coverage while minimizing parameter overhead.

2. **High-Frequency Gating Module (HFGM)**:
   Incorporates discrete 2D Laplacian operator approximations directly into the encoder hierarchy to attenuate low-frequency acoustic clutter and accentuate high-frequency boundary harmonics at soft-tissue interfaces.

3. **Cross-Scale Boundary-Aware Attention Skip Fusion (CS-BASF)**:
   Cross-projects deep semantic representations onto shallow encoder features through spatial and channel gating, filtering out background noise and artifact propagation before decoder reconstruction.

4. **Dual-Head Boundary Supervision**:
   Equipped with dual prediction heads—a primary segmentation mask head and an auxiliary boundary contour stream—jointly supervised via a compound loss (Soft Dice + Binary Cross-Entropy + Boundary Contour Loss).

---

## Benchmark Baseline Architectures

This repository includes full implementations and benchmarking pipelines for standard and state-of-the-art segmentation baselines:
- **FCN-8s** (Long et al., CVPR 2015)
- **Standard U-Net** (Ronneberger et al., MICCAI 2015)
- **Attention U-Net** (Oktay et al., MIDL 2018)
- **DeepLabV3+** (Chen et al., ECCV 2018)
- **TransUNet** (Chen et al., arXiv 2021)
- **EffiBound-Net** (Proposed Method)

---

## Repository Structure

```
EffiBound-Net/
├── config.py              # Centralized configuration (dataset paths, hyperparameters)
├── requirements.txt       # Dependencies and package specifications
├── colab_run.ipynb        # Ready-to-run Google Colab Notebook
├── test_models.py         # Forward pass and architecture integrity test
├── train.py               # Distributed/AMP training script
├── evaluate.py            # Comprehensive evaluation and metric computation script
├── visualize.py           # Qualitative visual predictions & contour overlay generator
├── datasets/
│   └── busi_dataset.py    # Custom DataLoader with boundary extraction & domain shifts
├── models/
│   ├── baseline_fcn.py
│   ├── baseline_unet.py
│   ├── baseline_att_unet.py
│   ├── baseline_deeplabv3p.py
│   ├── baseline_transunet.py
│   └── effibound_net.py   # Proposed EffiBound-Net architecture
├── losses/
│   └── compound_loss.py   # Soft Dice + BCE + Boundary Contour Loss
├── utils/
│   ├── metrics.py         # Dice, IoU, HD95, ASD, BIoU, Precision, Recall, ECE
│   └── complexity.py      # Parameters, FLOPs, Latency, FPS profiler
└── data/                  # Dataset storage directory
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SuttA-MNNIT/EffiBound-Net.git
cd EffiBound-Net
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Recommended environment: Python 3.9+, PyTorch 2.0+ (CUDA support recommended).

---

## Dataset Preparation

This repository uses the Breast Ultrasound Images (BUSI) benchmark dataset:
1. Download the Dataset BUSI with GT from Mendeley Data or Kaggle.
2. Extract the dataset into the `data/` folder such that the directory structure is organized as:
```
data/
└── Dataset_BUSI_with_GT/
    ├── benign/
    │   ├── benign (1).png
    │   └── benign (1)_mask.png
    ├── malignant/
    │   ├── malignant (1).png
    │   └── malignant (1)_mask.png
    └── normal/
        ├── normal (1).png
        └── normal (1)_mask.png
```

---

## Quickstart & Usage

### 1. Architecture Verification & Sanity Check
Verify model instantiation, output tensor shapes, parameter counts, and loss calculations across all architectures:
```bash
python test_models.py
```

### 2. Training
Train **EffiBound-Net** using Automatic Mixed Precision (AMP):
```bash
python train.py --model EffiBoundNet --batch_size 16 --epochs 100
```

To train any baseline model:
```bash
python train.py --model UNet --batch_size 16 --epochs 100
python train.py --model AttUNet --batch_size 16 --epochs 100
python train.py --model DeepLabV3Plus --batch_size 16 --epochs 100
python train.py --model TransUNet --batch_size 16 --epochs 100
```

Checkpoints will be saved in the `checkpoints/` directory.

### 3. Evaluation & Benchmarking
Evaluate a trained model checkpoint on the test split:
```bash
python evaluate.py --model EffiBoundNet --checkpoint checkpoints/EffiBoundNet_best.pth
```
This computes Dice, IoU, 95% Hausdorff Distance (HD95), Average Surface Distance (ASD), Boundary-IoU (BIoU), Precision, Recall, parameter counts, FLOPs, and latency. Results are saved as a CSV report in `results/`.

### 4. Qualitative Visual Predictions
Generate visual comparison overlays (Input Image, Ground Truth Mask, Predicted Mask, Boundary Map, and Contour Overlays):
```bash
python visualize.py --model EffiBoundNet --checkpoint checkpoints/EffiBoundNet_best.pth
```
Output images are saved to `visualizations/qualitative_predictions.png`.

### 5. Google Colab Quickstart
Open `colab_run.ipynb` in Google Colab to train and evaluate on cloud GPUs.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
