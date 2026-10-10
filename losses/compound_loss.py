"""
Loss Functions for Medical Image Segmentation
Includes:
- Soft Dice Loss (numerically stable with Laplace smoothing)
- Binary Cross-Entropy Loss
- Boundary Surface Contour Loss
- Compound Boundary-Supervised Hybrid Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W) raw network output before sigmoid
        targets: (B, 1, H, W) binary ground truth (0 or 1)
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()

class BoundaryContourLoss(nn.Module):
    """
    Penalizes deviations on lesion perimeter edges and fine contours
    Combines Weighted BCE and Contour Dice
    """
    def __init__(self, smooth=1e-5):
        super(BoundaryContourLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))

    def forward(self, boundary_logits, boundary_targets):
        device = boundary_logits.device
        if self.bce.pos_weight.device != device:
            self.bce.pos_weight = self.bce.pos_weight.to(device)
            
        bce_loss = self.bce(boundary_logits, boundary_targets)
        
        probs = torch.sigmoid(boundary_logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = boundary_targets.view(boundary_targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_boundary = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        contour_dice_loss = (1.0 - dice_boundary).mean()
        
        return 0.5 * bce_loss + 0.5 * contour_dice_loss

class CompoundSegmentationLoss(nn.Module):
    """
    Total Compound Loss:
    L = w_bce * L_bce + w_dice * L_dice + w_bound * L_boundary
    """
    def __init__(self, w_dice=1.0, w_bce=1.0, w_boundary=0.5):
        super(CompoundSegmentationLoss, self).__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.w_boundary = w_boundary
        
        self.dice_fn = SoftDiceLoss()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.boundary_fn = BoundaryContourLoss()

    def forward(self, seg_logits, target_mask, boundary_logits=None, target_boundary=None):
        l_dice = self.dice_fn(seg_logits, target_mask)
        l_bce = self.bce_fn(seg_logits, target_mask)
        
        loss = self.w_dice * l_dice + self.w_bce * l_bce
        
        l_bound = torch.tensor(0.0, device=seg_logits.device)
        if boundary_logits is not None and target_boundary is not None:
            l_bound = self.boundary_fn(boundary_logits, target_boundary)
            loss = loss + self.w_boundary * l_bound
            
        return {
            "loss": loss,
            "loss_dice": l_dice.item(),
            "loss_bce": l_bce.item(),
            "loss_boundary": l_bound.item() if isinstance(l_bound, torch.Tensor) else 0.0
        }
