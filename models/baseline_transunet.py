"""
Baseline 5: TransUNet (Hybrid CNN-Vision Transformer)
Reference: Chen et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation", arXiv:2102.04306 / IEEE.
Features:
- Hybrid CNN (ResNet-like stages) feature extractor
- 12-layer Vision Transformer encoder (d=768, heads=12, ff=3072) for global self-attention
- Cascaded Upsampler (CUP) decoder with multi-resolution skip connections
- Parameter count: ~105.3M (standard)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c or stride != 1 else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x), inplace=True)

class R50CNNEncoder(nn.Module):
    """
    Standard ResNet-50 hybrid feature extractor for TransUNet.
    Yields multi-scale features for skip connections.
    """
    def __init__(self, in_channels=3):
        super(R50CNNEncoder, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) # 1/2 resolution (128x128)
        
        # Stage 1: 64 -> 128 (1/4 resolution: 64x64)
        self.stage1 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128)
        )
        
        # Stage 2: 128 -> 256 (1/8 resolution: 32x32)
        self.stage2 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        
        # Stage 3: 256 -> 512 (1/16 resolution: 16x16)
        self.stage3 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
            ResBlock(512, 512)
        )

    def forward(self, x):
        x_stem = self.stem(x)       # 1/2 (64ch)
        x_s1 = self.stage1(x_stem)  # 1/4 (128ch)
        x_s2 = self.stage2(x_s1)    # 1/8 (256ch)
        x_s3 = self.stage3(x_s2)    # 1/16 (512ch)
        return x_stem, x_s1, x_s2, x_s3

class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv1(x)

class TransUNet(nn.Module):
    """
    TransUNet: Hybrid ViT-B/16 + ResNet architecture.
    variant: 'standard' (105.3M params, default) or 'lite' (3.8M params for fast tests).
    """
    def __init__(self, in_channels=3, num_classes=1, variant="standard", img_size=(256, 256)):
        super(TransUNet, self).__init__()
        self.img_size = img_size
        self.variant = variant
        
        if variant == "standard":
            self.d_model = 768
            n_heads = 12
            n_layers = 12
            dim_feedforward = 3072
            
            # CNN Encoder
            self.cnn_encoder = R50CNNEncoder(in_channels)
            # x_s3 has 512 channels at 1/16 resolution (16x16 for 256x256)
            self.proj = nn.Conv2d(512, self.d_model, kernel_size=1)
            patch_h, patch_w = img_size[0] // 16, img_size[1] // 16
            num_patches = patch_h * patch_w
            
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model=self.d_model, nhead=n_heads, dim_feedforward=dim_feedforward)
                for _ in range(n_layers)
            ])
            
            self.trans_out_proj = nn.Sequential(
                nn.Conv2d(self.d_model, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            # Cascaded CUP Decoder
            self.dec1 = DecoderBlock(in_channels=256, out_channels=256, skip_channels=256) # 1/16 -> 1/8 (skip x_s2: 256ch)
            self.dec2 = DecoderBlock(in_channels=256, out_channels=128, skip_channels=128) # 1/8 -> 1/4 (skip x_s1: 128ch)
            self.dec3 = DecoderBlock(in_channels=128, out_channels=64, skip_channels=64)   # 1/4 -> 1/2 (skip x_stem: 64ch)
            self.dec4 = DecoderBlock(in_channels=64, out_channels=32, skip_channels=0)     # 1/2 -> 1/1 (final upsample)
            self.final_head = nn.Conv2d(32, num_classes, kernel_size=1)
            
        else: # 'lite' variant
            self.d_model = 256
            n_heads = 8
            n_layers = 4
            dim_feedforward = 512
            
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.proj = nn.Conv2d(256, self.d_model, kernel_size=1)
            patch_h, patch_w = img_size[0] // 8, img_size[1] // 8
            self.pos_embed = nn.Parameter(torch.zeros(1, patch_h * patch_w, self.d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model=self.d_model, nhead=n_heads, dim_feedforward=dim_feedforward)
                for _ in range(n_layers)
            ])
            self.trans_out_proj = nn.Conv2d(self.d_model, 256, kernel_size=1)
            self.dec1 = DecoderBlock(256, 128)
            self.dec2 = DecoderBlock(128, 64)
            self.dec3 = DecoderBlock(64, 32)
            self.dec4 = nn.Identity()
            self.final_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        
        if self.variant == "standard":
            x_stem, x_s1, x_s2, x_s3 = self.cnn_encoder(x)
            feat = self.proj(x_s3)
            _, _, fh, fw = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)
            
            if tokens.shape[1] == self.pos_embed.shape[1]:
                tokens = tokens + self.pos_embed
            else:
                pos_emb = F.interpolate(self.pos_embed.transpose(1, 2), size=tokens.shape[1], mode='linear').transpose(1, 2)
                tokens = tokens + pos_emb
                
            for blk in self.transformer_blocks:
                tokens = blk(tokens)
                
            trans_feat = tokens.transpose(1, 2).view(b, self.d_model, fh, fw)
            trans_feat = self.trans_out_proj(trans_feat)
            
            d1 = self.dec1(trans_feat, x_s2)
            d2 = self.dec2(d1, x_s1)
            d3 = self.dec3(d2, x_stem)
            d4 = self.dec4(d3)
            return self.final_head(d4)
        else:
            feat = self.cnn_encoder(x)
            tokens = self.proj(feat).flatten(2).transpose(1, 2) + self.pos_embed
            for blk in self.transformer_blocks:
                tokens = blk(tokens)
            trans_feat = self.trans_out_proj(tokens.transpose(1, 2).view(b, self.d_model, h//8, w//8))
            d1 = self.dec1(trans_feat)
            d2 = self.dec2(d1)
            d3 = self.dec3(d2)
            return self.final_head(d3)

