"""
Medical Image Dataset Pipeline: Breast Ultrasound (BUSI) & Multi-Center Simulation
Supports:
- Ground-truth lesion mask generation (handling multiple lesion annotations)
- High-precision boundary contour ground-truth extraction (for boundary supervision)
- Standard medical augmentations
- Domain-shift / Robustness degradation modes (low contrast, speckle noise, resolution downsampling)
"""

import os
import glob
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class BUSIDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 256), transform=None, 
                 robustness_mode=None, robustness_severity=1.0, seed=42):
        """
        robustness_mode: None, 'low_contrast', 'speckle_noise', 'gaussian_noise', 'scanner_blur'
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.robustness_mode = robustness_mode
        self.robustness_severity = robustness_severity
        
        self.samples = self._load_samples(seed)
        
    def _load_samples(self, seed):
        categories = ["benign", "malignant", "normal"]
        all_samples = []
        
        for cat in categories:
            cat_dir = os.path.join(self.root_dir, cat)
            if not os.path.exists(cat_dir):
                continue
                
            all_files = glob.glob(os.path.join(cat_dir, "*.png"))
            img_files = [f for f in all_files if "_mask" not in os.path.basename(f)]
            
            for img_path in sorted(img_files):
                base_name = os.path.splitext(img_path)[0]
                # Find all corresponding masks (e.g. base_mask.png, base_mask_1.png)
                mask_pattern = base_name + "_mask*.png"
                mask_files = sorted(glob.glob(mask_pattern))
                all_samples.append({
                    "image_path": img_path,
                    "mask_paths": mask_files,
                    "category": cat
                })
                
        # True class-stratified split across clinical categories
        np.random.seed(seed)
        cat_samples = {cat: [] for cat in categories}
        for s in all_samples:
            cat_samples[s["category"]].append(s)

        train_samples, val_samples, test_samples = [], [], []
        for cat, samples in cat_samples.items():
            perm = np.random.permutation(len(samples))
            n = len(samples)
            n_tr = int(n * 0.70)
            n_va = int(n * 0.15)
            
            for idx in perm[:n_tr]:
                train_samples.append(samples[idx])
            for idx in perm[n_tr:n_tr + n_va]:
                val_samples.append(samples[idx])
            for idx in perm[n_tr + n_va:]:
                test_samples.append(samples[idx])

        if self.split == "train":
            return train_samples
        elif self.split == "val":
            return val_samples
        elif self.split == "test":
            return test_samples
        else:
            return all_samples

    def _extract_boundary(self, mask_np):
        """
        Extract crisp boundary contours using morphological gradient
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        boundary = dilated - eroded
        return (boundary > 0).astype(np.float32)

    def _apply_robustness_degradation(self, img_np):
        """
        Simulate scanner domain shifts and acquisition degradations:
        - Low contrast (poor ultrasound acoustic window)
        - Speckle noise (Rayleigh-distributed ultrasound noise)
        - Gaussian sensor noise
        - Scanner defocus / blur
        """
        img = img_np.copy()
        sev = self.robustness_severity
        
        if self.robustness_mode == "low_contrast":
            # Compress dynamic range towards the mean
            mean_val = np.mean(img)
            factor = max(0.1, 1.0 - 0.7 * sev)
            img = mean_val + factor * (img - mean_val)
            img = np.clip(img, 0.0, 1.0)
            
        elif self.robustness_mode == "speckle_noise":
            # Multiplicative Rayleigh/speckle noise characteristic of ultrasound
            noise = np.random.normal(0, 0.25 * sev, img.shape)
            img = img + img * noise
            img = np.clip(img, 0.0, 1.0)
            
        elif self.robustness_mode == "gaussian_noise":
            noise = np.random.normal(0, 0.15 * sev, img.shape)
            img = np.clip(img + noise, 0.0, 1.0)
            
        elif self.robustness_mode == "scanner_blur":
            ksize = int(5 * sev) | 1  # ensure odd
            img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=2.0 * sev)
            
        return img.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load image (RGB)
        img_bgr = cv2.imread(item["image_path"])
        if img_bgr is None:
            # Fallback black image if read fails
            img_bgr = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        img_norm = (img_resized.astype(np.float32) / 255.0)
        
        # Load and combine masks
        if len(item["mask_paths"]) == 0:
            mask_combined = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.float32)
        else:
            mask_combined = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.float32)
            for m_path in item["mask_paths"]:
                m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m_res = cv2.resize(m, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
                    mask_combined = np.maximum(mask_combined, (m_res > 127).astype(np.float32))
                    
        # Apply robustness degradation if specified (e.g. for evaluating cross-hospital shift)
        if self.robustness_mode is not None:
            img_norm = self._apply_robustness_degradation(img_norm)
            
        # Training data augmentation
        if self.split == "train" and self.transform is None:
            if np.random.rand() > 0.5:
                # Horizontal flip
                img_norm = np.fliplr(img_norm).copy()
                mask_combined = np.fliplr(mask_combined).copy()
            if np.random.rand() > 0.5:
                # Vertical flip
                img_norm = np.flipud(img_norm).copy()
                mask_combined = np.flipud(mask_combined).copy()
            if np.random.rand() > 0.5:
                # Slight gamma variation
                gamma = np.random.uniform(0.8, 1.2)
                img_norm = np.clip(np.power(img_norm, gamma), 0.0, 1.0)
                
        # Extract crisp boundary contour
        boundary_np = self._extract_boundary(mask_combined)
        
        # Format tensors (C, H, W)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_combined).unsqueeze(0).float()
        boundary_tensor = torch.from_numpy(boundary_np).unsqueeze(0).float()
        
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "boundary": boundary_tensor,
            "category": item["category"],
            "filename": os.path.basename(item["image_path"])
        }

def get_dataloaders(root_dir, batch_size=16, num_workers=2, img_size=(256, 256), pin_memory=True):
    train_dataset = BUSIDataset(root_dir, split="train", img_size=img_size)
    val_dataset = BUSIDataset(root_dir, split="val", img_size=img_size)
    test_dataset = BUSIDataset(root_dir, split="test", img_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory)
                             
    return train_loader, val_loader, test_loader
