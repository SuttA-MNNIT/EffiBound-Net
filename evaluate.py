"""
Evaluation Script for EffiBound-Net and Baseline Models
Evaluates trained model checkpoints on the test split and computes evaluation metrics:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- 95% Hausdorff Distance (HD95)
- Average Surface Distance (ASD)
- Boundary-IoU (BIoU)
- Precision, Recall, Specificity
- Model Parameters, FLOPs, Inference Latency, and FPS
"""

import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from datasets.busi_dataset import BUSIDataset
from models import get_model
from utils.metrics import compute_batch_metrics
from utils.complexity import profile_model

def evaluate_model(model, dataloader, device, is_effibound, max_batches=None):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            if is_effibound:
                seg_out = model(images, return_boundary=False)
            else:
                seg_out = model(images)

            _, batch_indiv = compute_batch_metrics(seg_out, masks, return_individual=True)
            all_metrics.extend(batch_indiv)

    return pd.DataFrame(all_metrics)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Models on Test Split")
    parser.add_argument("--data_dir", type=str, default=Config.DATASET_ROOT)
    parser.add_argument("--model", type=str, default="EffiBoundNet", choices=Config.BENCHMARK_MODELS)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=Config.OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None, help="Optional limit on batches for quick evaluation")
    parser.add_argument("--output_dir", type=str, default=Config.RESULTS_DIR)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading test dataset from: {args.data_dir}")
    test_dataset = BUSIDataset(args.data_dir, split="test", img_size=Config.IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Evaluating model: {args.model} on {device}")
    model = get_model(args.model, in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)
    is_effibound = (args.model == "EffiBoundNet")

    ckpt_path = args.checkpoint or os.path.join(args.checkpoints_dir, f"{args.model}_best.pth")
    if os.path.exists(ckpt_path):
        print(f"Loading weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"Notice: Checkpoint not found at {ckpt_path}. Running with initialized weights.")

    prof = profile_model(model, input_size=(1, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1]), device=str(device))
    print(f"Model Complexity: Params = {prof['params_m']:.2f}M | FLOPs = {prof['gflops']:.2f}G | Latency = {prof['latency_ms']:.2f}ms | FPS = {prof['fps']:.1f}")

    results_df = evaluate_model(model, test_loader, device, is_effibound, max_batches=args.max_batches)

    summary = {
        "Model": args.model,
        "Params_M": round(prof["params_m"], 3),
        "FLOPs_G": round(prof["gflops"], 3),
        "Latency_ms": round(prof["latency_ms"], 2),
        "FPS": round(prof["fps"], 1),
        "Dice": round(float(results_df["dice"].mean()) * 100, 2),
        "IoU": round(float(results_df["iou"].mean()) * 100, 2),
        "HD95": round(float(results_df["hd95"].mean()), 2),
        "ASD": round(float(results_df["asd"].mean()), 2),
        "BIoU": round(float(results_df["biou"].mean()) * 100, 2),
        "Precision": round(float(results_df["precision"].mean()) * 100, 2),
        "Recall": round(float(results_df["recall"].mean()) * 100, 2)
    }

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_csv = os.path.join(args.output_dir, f"{args.model}_test_metrics.csv")
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print(f"\nSaved evaluation metrics to: {out_csv}")

if __name__ == "__main__":
    main()
