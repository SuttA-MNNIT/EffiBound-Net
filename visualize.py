"""
Inference and Qualitative Visualization Script
Generates visual prediction overlays comparing Ground Truth annotations against Model predictions.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from datasets.busi_dataset import BUSIDataset
from models import get_model

def visualize_predictions(model, dataset, device, is_effibound, output_dir, num_samples=4):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    num_cols = 4 if not is_effibound else 5
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3.5, num_samples * 3.2))

    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["Input Ultrasound", "Ground Truth", "Predicted Mask", "Overlay"]
    if is_effibound:
        col_titles.insert(3, "Predicted Boundary")

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, pad=10, fontweight="bold")

    with torch.no_grad():
        for row_idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            img_tensor = sample["image"].unsqueeze(0).to(device)
            mask_np = sample["mask"].numpy().squeeze()

            if is_effibound:
                seg_out, bound_out = model(img_tensor, return_boundary=True)
                pred_mask = (torch.sigmoid(seg_out).cpu().numpy().squeeze() > 0.5).astype(np.float32)
                pred_bound = torch.sigmoid(bound_out).cpu().numpy().squeeze()
            else:
                seg_out = model(img_tensor)
                pred_mask = (torch.sigmoid(seg_out).cpu().numpy().squeeze() > 0.5).astype(np.float32)
                pred_bound = None

            # Un-normalize image for visualization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            img_np = sample["image"].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_vis = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

            # Col 0: Input Image
            axes[row_idx, 0].imshow(img_vis)
            axes[row_idx, 0].axis("off")

            # Col 1: Ground Truth Mask
            axes[row_idx, 1].imshow(mask_np, cmap="gray")
            axes[row_idx, 1].axis("off")

            # Col 2: Predicted Mask
            axes[row_idx, 2].imshow(pred_mask, cmap="gray")
            axes[row_idx, 2].axis("off")

            col_curr = 3
            if is_effibound:
                # Boundary prediction
                axes[row_idx, col_curr].imshow(pred_bound, cmap="magma")
                axes[row_idx, col_curr].axis("off")
                col_curr += 1

            # Overlay: Green for Ground Truth, Red for Prediction
            overlay = img_vis.copy()
            gt_contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pred_contours, _ = cv2.findContours((pred_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)   # Green = GT
            cv2.drawContours(overlay, pred_contours, -1, (255, 50, 50), 2) # Red = Pred

            axes[row_idx, col_curr].imshow(overlay)
            axes[row_idx, col_curr].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "qualitative_predictions.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative predictions visualization to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Qualitative Visual Predictions")
    parser.add_argument("--data_dir", type=str, default=Config.DATASET_ROOT)
    parser.add_argument("--model", type=str, default="EffiBoundNet", choices=Config.BENCHMARK_MODELS)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=Config.OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "visualizations"))
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading test split from: {args.data_dir}")
    test_dataset = BUSIDataset(args.data_dir, split="test", img_size=Config.IMG_SIZE)

    print(f"Initializing {args.model} on {device}...")
    model = get_model(args.model, in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)
    is_effibound = (args.model == "EffiBoundNet")

    ckpt_path = args.checkpoint or os.path.join(args.checkpoints_dir, f"{args.model}_best.pth")
    if os.path.exists(ckpt_path):
        print(f"Loading weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"Notice: Checkpoint not found at {ckpt_path}. Visualizing with initialized weights.")

    visualize_predictions(model, test_dataset, device, is_effibound, args.output_dir, num_samples=args.num_samples)

if __name__ == "__main__":
    main()
