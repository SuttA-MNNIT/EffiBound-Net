"""
Model Complexity and Efficiency Profiler
Measures:
- Parameter count (M)
- FLOPs / MACs (G)
- Latency (ms) and Throughput (FPS)
"""

import time
import torch
import numpy as np

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def estimate_flops(model, input_size=(1, 3, 256, 256), device="cpu"):
    """
    Hook-based FLOPs estimator for standard Conv2d, Linear, BatchNorm layers
    """
    model.eval()
    flops = []
    handles = []

    def conv_hook(module, input, output):
        # input is tuple of tensors, output is tensor
        b, c_out, h_out, w_out = output.shape
        c_in = module.in_channels
        k_h, k_w = module.kernel_size
        groups = module.groups
        # FLOPs per output element: 2 * (c_in / groups) * k_h * k_w
        flop = b * c_out * h_out * w_out * (2 * (c_in // groups) * k_h * k_w)
        flops.append(flop)

    def linear_hook(module, input, output):
        in_features = module.in_features
        out_features = module.out_features
        b = input[0].shape[0]
        flop = b * (2 * in_features) * out_features
        flops.append(flop)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    dummy = torch.randn(*input_size).to(device)
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception:
            pass

    for h in handles:
        h.remove()

    total_flops = sum(flops)
    return total_flops

def measure_latency(model, input_size=(1, 3, 256, 256), device="cpu", warmups=10, runs=50):
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(dummy)
            
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
            
        t0 = time.time()
        for _ in range(runs):
            _ = model(dummy)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
        t1 = time.time()
        
    avg_latency_ms = ((t1 - t0) / runs) * 1000.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    return avg_latency_ms, fps

def profile_model(model, input_size=(1, 3, 256, 256), device="cpu"):
    params_total, params_train = count_parameters(model)
    flops = estimate_flops(model, input_size, device)
    warmups = 5 if "cuda" in str(device) else 1
    runs = 20 if "cuda" in str(device) else 2
    latency_ms, fps = measure_latency(model, input_size, device, warmups=warmups, runs=runs)
    
    return {
        "params_m": params_total / 1e6,
        "gflops": flops / 1e9,
        "latency_ms": latency_ms,
        "fps": fps
    }
