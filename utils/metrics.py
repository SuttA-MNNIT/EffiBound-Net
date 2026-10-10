"""
Evaluation Metrics for Medical Image Segmentation
Implements:
- Dice Similarity Coefficient (DSC)
- Intersection-over-Union (IoU) / Jaccard Index
- True 95% Hausdorff Distance (HD95) via Euclidean Distance Transform
- Average Symmetric Surface Distance (ASD / ASSD)
- Boundary-IoU (BIoU)
- Precision, Recall, Specificity
- Expected Calibration Error (ECE) & Brier Score
- Per-sample and batch-level aggregation
"""

import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt

def compute_surface_distances(p_bound, t_bound):
    """
    Computes exact bidirectional Euclidean surface distances using distance_transform_edt.
    Returns: hd95 (95th percentile Hausdorff Distance) and asd (Average Surface Distance).
    """
    has_p = np.any(p_bound > 0)
    has_t = np.any(t_bound > 0)
    
    if not has_p and not has_t:
        return 0.0, 0.0
    if not has_p or not has_t:
        # Missed detection penalty (bounded by diagonal in 256x256 ~ 50.0 px standardized)
        return 50.0, 50.0

    # Distance map from target boundary
    dt_target = distance_transform_edt(1 - t_bound)
    dists_p_to_t = dt_target[p_bound > 0]

    # Distance map from pred boundary
    dt_pred = distance_transform_edt(1 - p_bound)
    dists_t_to_p = dt_pred[t_bound > 0]

    all_dists = np.concatenate([dists_p_to_t, dists_t_to_p])
    hd95 = float(np.percentile(all_dists, 95))
    asd = float(np.mean(all_dists))
    return hd95, asd

def compute_expected_calibration_error(probs_flat, targets_flat, n_bins=10):
    """
    Expected Calibration Error (ECE) measuring confidence reliability.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs_flat)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs_flat >= bin_lower) & (probs_flat < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(probs_flat[in_bin])
            avg_accuracy = np.mean(targets_flat[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
            
    return float(ece)

def compute_single_sample_metrics(prob_2d, target_2d, threshold=0.5):
    """
    Calculates all metrics for a single 2D slice (H, W).
    """
    pred_2d = (prob_2d > threshold).astype(np.uint8)
    target_2d = target_2d.astype(np.uint8)

    tp = np.sum((pred_2d == 1) & (target_2d == 1))
    fp = np.sum((pred_2d == 1) & (target_2d == 0))
    fn = np.sum((pred_2d == 0) & (target_2d == 1))
    tn = np.sum((pred_2d == 0) & (target_2d == 0))

    # Dice
    denom_dice = 2 * tp + fp + fn
    dice = (2.0 * tp / denom_dice) if denom_dice > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)

    # IoU
    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)

    # Precision, Recall, Specificity
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fn == 0 else 0.0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    # Boundary Contours (d=3 pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    p_bound = cv2.dilate(pred_2d, kernel) - cv2.erode(pred_2d, kernel)
    t_bound = cv2.dilate(target_2d, kernel) - cv2.erode(target_2d, kernel)

    # Boundary-IoU
    b_tp = np.sum((p_bound == 1) & (t_bound == 1))
    b_union = np.sum((p_bound == 1) | (t_bound == 1))
    biou = (b_tp / b_union) if b_union > 0 else (1.0 if (np.sum(p_bound) == 0 and np.sum(t_bound) == 0) else 0.0)

    # True HD95 and ASD
    hd95, asd = compute_surface_distances(p_bound, t_bound)

    # Brier Score & ECE
    brier = float(np.mean((prob_2d - target_2d) ** 2))
    ece = compute_expected_calibration_error(prob_2d.flatten(), target_2d.flatten())

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "biou": float(biou),
        "hd95": float(hd95),
        "asd": float(asd),
        "brier": float(brier),
        "ece": float(ece)
    }

def compute_batch_metrics(pred_logits, targets, threshold=0.5, return_individual=False):
    """
    pred_logits: Tensor of shape (B, 1, H, W)
    targets: Tensor of shape (B, 1, H, W)
    """
    probs = torch.sigmoid(pred_logits)
    probs_np = probs.detach().cpu().numpy().squeeze(1)
    targets_np = targets.detach().cpu().numpy().squeeze(1)

    batch_size = probs_np.shape[0]
    individual_metrics = []

    for i in range(batch_size):
        m = compute_single_sample_metrics(probs_np[i], targets_np[i], threshold=threshold)
        individual_metrics.append(m)

    # Aggregate means
    avg_metrics = {}
    for k in individual_metrics[0].keys():
        avg_metrics[k] = float(np.mean([m[k] for m in individual_metrics]))

    if return_individual:
        return avg_metrics, individual_metrics
    return avg_metrics
