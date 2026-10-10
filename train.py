"""
Training Script for Medical Image Segmentation
Optimized for NVIDIA A100 GPU on Google Colab with AMP (Automatic Mixed Precision)
Supports all benchmark models and proposed EffiBound-Net.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from datasets.busi_dataset import BUSIDataset, get_dataloaders
from models import get_model
from losses.compound_loss import CompoundSegmentationLoss
from utils.metrics import compute_batch_metrics

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, is_effibound, use_amp):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        boundaries = batch["boundary"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                if is_effibound:
                    seg_out, bound_out = model(images, return_boundary=True)
                    loss_dict = criterion(seg_out, masks, bound_out, boundaries)
                else:
                    seg_out = model(images)
                    loss_dict = criterion(seg_out, masks)
                loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if is_effibound:
                seg_out, bound_out = model(images, return_boundary=True)
                loss_dict = criterion(seg_out, masks, bound_out, boundaries)
            else:
                seg_out = model(images)
                loss_dict = criterion(seg_out, masks)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        
        # Fast batch metrics
        with torch.no_grad():
            b_metrics = compute_batch_metrics(seg_out, masks)
            total_dice += b_metrics["dice"]
            total_iou += b_metrics["iou"]

    return {
        "loss": total_loss / max(1, num_batches),
        "dice": total_dice / max(1, num_batches),
        "iou": total_iou / max(1, num_batches)
    }

def validate(model, dataloader, criterion, device, is_effibound):
    model.eval()
    total_loss = 0.0
    all_metrics = {"dice": [], "iou": [], "precision": [], "recall": [], "specificity": [], "biou": [], "hd95": []}
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            boundaries = batch["boundary"].to(device, non_blocking=True)

            if is_effibound:
                seg_out, bound_out = model(images, return_boundary=True)
                loss_dict = criterion(seg_out, masks, bound_out, boundaries)
            else:
                seg_out = model(images)
                loss_dict = criterion(seg_out, masks)

            total_loss += loss_dict["loss"].item()
            m = compute_batch_metrics(seg_out, masks)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    avg_metrics["loss"] = total_loss / max(1, len(dataloader))
    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Medical Image Segmentation Models on A100 GPU")
    parser.add_argument("--model", type=str, default="EffiBoundNet", choices=Config.BENCHMARK_MODELS)
    parser.add_argument("--data_dir", type=str, default=Config.DATASET_ROOT)
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--save_dir", type=str, default=Config.OUTPUT_DIR)
    parser.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = (not args.no_amp) and (device.type == "cuda")
    is_effibound = (args.model == "EffiBoundNet")

    print("=" * 65)
    print(f"TRAINING CONFIGURATION: {args.model}")
    print("=" * 65)
    print(f"Device: {device} | AMP: {use_amp} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Dataset: {args.data_dir}")
    print(f"Checkpoints: {args.save_dir}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("-" * 65)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=Config.IMG_SIZE,
        pin_memory=(device.type == "cuda")
    )

    print(f"Data Splits: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Model
    model = get_model(args.model, in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Initialized {args.model} with {total_params:.2f}M parameters")

    # Optimizer, Loss, Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.MIN_LR)
    criterion = CompoundSegmentationLoss(w_dice=Config.DICE_WEIGHT, w_bce=Config.BCE_WEIGHT, w_boundary=Config.BOUNDARY_WEIGHT)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_dice = 0.0
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": [], "val_iou": [], "val_hd95": []}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, is_effibound, use_amp)
        val_res = validate(model, val_loader, criterion, device, is_effibound)
        scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(train_res["loss"])
        history["train_dice"].append(train_res["dice"])
        history["val_loss"].append(val_res["loss"])
        history["val_dice"].append(val_res["dice"])
        history["val_iou"].append(val_res["iou"])
        history["val_hd95"].append(val_res["hd95"])

        is_best = val_res["dice"] > best_val_dice
        if is_best:
            best_val_dice = val_res["dice"]
            best_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
                "val_metrics": val_res
            }, best_path)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            mark = " [*BEST*]" if is_best else ""
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] ({elapsed:.1f}s) | "
                  f"Train Loss: {train_res['loss']:.4f} | Train Dice: {train_res['dice']:.4f} | "
                  f"Val Loss: {val_res['loss']:.4f} | Val Dice: {val_res['dice']:.4f} | "
                  f"Val IoU: {val_res['iou']:.4f} | Val HD95: {val_res['hd95']:.2f}{mark}")

    total_time = (time.time() - start_time) / 60.0
    print("-" * 65)
    print(f"Training completed in {total_time:.2f} mins! Best Val Dice: {best_val_dice:.4f}")

    # Save training history JSON
    history_path = os.path.join(Config.RESULTS_DIR, f"{args.model}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_path}")

if __name__ == "__main__":
    main()
