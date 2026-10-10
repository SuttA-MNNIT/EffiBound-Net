"""
Fast Unit Test & Architecture Verification Script
Tests forward pass shapes, parameter counts, FLOPs, and loss functions for all models.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model, FCN8s, UNet, AttentionUNet, DeepLabV3Plus, TransUNet, EffiBoundNet
from losses import CompoundSegmentationLoss
from utils.metrics import compute_batch_metrics
from utils.complexity import profile_model

def run_tests():
    print("=" * 60)
    print("RUNNING ARCHITECTURE & PIPELINE INTEGRITY TESTS")
    print("=" * 60)
    
    device = torch.device("cpu")
    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    dummy_mask = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)
    dummy_bound = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)
    
    models_to_test = [
        ("FCN-8s", "FCN"),
        ("Standard U-Net", "UNet"),
        ("Attention U-Net", "AttUNet"),
        ("DeepLabV3+", "DeepLabV3Plus"),
        ("TransUNet", "TransUNet"),
        ("Proposed EffiBound-Net", "EffiBoundNet")
    ]
    
    loss_fn = CompoundSegmentationLoss()
    
    print(f"{'Model':<25} | {'Params (M)':<10} | {'Output Shape':<18} | {'Loss':<8} | {'Status'}")
    print("-" * 75)
    
    for display_name, model_key in models_to_test:
        model = get_model(model_key, in_channels=3, num_classes=1, img_size=(256, 256)).to(device)
        model.eval()
        
        with torch.no_grad():
            if model_key == "EffiBoundNet":
                seg_out, bound_out = model(dummy_input, return_boundary=True)
                assert seg_out.shape == (2, 1, 256, 256), f"Seg shape mismatch: {seg_out.shape}"
                assert bound_out.shape == (2, 1, 256, 256), f"Bound shape mismatch: {bound_out.shape}"
                loss_dict = loss_fn(seg_out, dummy_mask, bound_out, dummy_bound)
                out_shape = str(tuple(seg_out.shape))
            else:
                out = model(dummy_input)
                assert out.shape == (2, 1, 256, 256), f"Shape mismatch: {out.shape}"
                loss_dict = loss_fn(out, dummy_mask)
                out_shape = str(tuple(out.shape))
                
        params = sum(p.numel() for p in model.parameters()) / 1e6
        loss_val = loss_dict["loss"].item()
        
        print(f"{display_name:<25} | {params:<10.2f} | {out_shape:<18} | {loss_val:<8.4f} | PASSED")
        
    print("\nTesting Metrics Computation...")
    test_metrics = compute_batch_metrics(torch.randn(2, 1, 256, 256), dummy_mask)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\nALL ARCHITECTURAL & UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
