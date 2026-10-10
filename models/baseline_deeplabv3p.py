"""
Baseline 4: DeepLabV3+
Reference: Chen et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", ECCV 2018 / IEEE TPAMI.
Uses an Atrous Spatial Pyramid Pooling (ASPP) module with multi-rate atrous convolutions and a low-level feature decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, backbone_channels=[64, 128, 256, 512]):
        super(DeepLabV3Plus, self).__init__()
        
        # Lightweight ResNet-like feature encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, backbone_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels[0], backbone_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling stages
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(backbone_channels[0], backbone_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(backbone_channels[1], backbone_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(backbone_channels[2], backbone_channels[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # ASPP on deepest stage
        self.aspp = ASPP(backbone_channels[3], atrous_rates=[6, 12, 18], out_channels=256)
        
        # Low-level projection from layer1 (1/2 scale)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(backbone_channels[0], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        
        low_level = self.stem(x)
        x_enc = self.layer1(low_level)
        x_enc = self.layer2(x_enc)
        x_enc = self.layer3(x_enc)
        
        aspp_out = self.aspp(x_enc)
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[2:], mode="bilinear", align_corners=True)
        
        low_proj = self.low_level_proj(low_level)
        fused = torch.cat([aspp_up, low_proj], dim=1)
        
        out = self.decoder(fused)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
            
        return out
