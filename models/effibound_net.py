"""
EffiBound-Net: A Frequency-Guided and Boundary-Aware Neural Network for Medical Image Segmentation

Key Components:
1. Multi-Scale Depthwise Receptive Field Block (MS-DRFB):
   Captures multi-scale receptive field coverage using parallel dilated depthwise separable convolutions.
2. High-Frequency Gating Module (HFGM):
   Decomposes features into high-frequency edge components via 2D discrete Laplacian operators to enhance boundary representations.
3. Cross-Scale Boundary-Aware Attention Skip Fusion (CS-BASF):
   Selectively filters background noise and artifacts across skip connections using semantic-boundary gating.
4. Dual-Head Boundary-Supervised Architecture:
   Jointly predicts lesion masks and boundary contours.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SqueezeExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)  # Swish activation for smooth gradient flow

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class MultiScaleReceptiveBlock(nn.Module):
    """
    MS-DRFB: Parallel multi-dilation depthwise convolutions with SE channel recalibration
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleReceptiveBlock, self).__init__()
        branch_channels = out_channels // 4
        
        self.branch1 = DepthwiseSeparableConv(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1)
        self.branch2 = DepthwiseSeparableConv(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2)
        self.branch3 = DepthwiseSeparableConv(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True)
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.se = SqueezeExcitation(out_channels)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fuse(out)
        out = self.se(out)
        return out + res

class HighFrequencyGatingModule(nn.Module):
    """
    HFGM: Extracts high-frequency boundary contours using fixed 2D Laplacian operator,
    then applies learnable frequency gating to amplify lesion margins.
    """
    def __init__(self, channels):
        super(HighFrequencyGatingModule, self).__init__()
        # 2D Laplacian operator kernel: [0, 1, 0; 1, -4, 1; 0, 1, 0]
        laplacian_weight = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, -4.0, 1.0],
               [0.0, 1.0, 0.0]]]], dtype=torch.float32
        )
        self.register_buffer("laplacian_kernel", laplacian_weight)
        
        self.edge_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        # Apply depthwise Laplacian filter across all channels
        k = self.laplacian_kernel.repeat(c, 1, 1, 1)
        high_freq = F.conv2d(x, k, padding=1, groups=c)
        high_freq_abs = torch.abs(high_freq)
        
        gate = self.edge_proj(high_freq_abs)
        # Modulate feature map: retain original content and amplify boundary high frequencies
        return x + x * gate

class CrossScaleBoundaryAwareSkipFusion(nn.Module):
    """
    CS-BASF: Replaces naive U-Net skip concatenation with semantic-boundary cross attention
    """
    def __init__(self, enc_channels, dec_channels, out_channels):
        super(CrossScaleBoundaryAwareSkipFusion, self).__init__()
        self.conv_dec = nn.Sequential(
            nn.Conv2d(dec_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_enc = nn.Sequential(
            nn.Conv2d(enc_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, enc_feat, dec_feat):
        # Resize dec_feat if needed to match enc_feat spatial resolution
        if dec_feat.shape[2:] != enc_feat.shape[2:]:
            dec_feat = F.interpolate(dec_feat, size=enc_feat.shape[2:], mode="bilinear", align_corners=True)
            
        enc_proj = self.conv_enc(enc_feat)
        dec_proj = self.conv_dec(dec_feat)
        
        # Cross-gating attention
        concat_cross = torch.cat([enc_proj, dec_proj], dim=1)
        gate = self.spatial_gate(concat_cross)
        
        # Filtered encoder skip features
        filtered_enc = enc_proj * gate
        
        # Fused combination
        fused = torch.cat([dec_proj, filtered_enc], dim=1)
        return self.fuse(fused)

class EffiBoundNet(nn.Module):
    """
    EffiBound-Net:
    Encoder-Decoder with Multi-Scale Depthwise Blocks, High-Frequency Gating,
    Cross-Scale Attention Skips, and Auxiliary Boundary Contour Supervision.
    """
    def __init__(self, in_channels=3, num_classes=1, channels=[32, 64, 128, 256]):
        super(EffiBoundNet, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        # Encoder Stages
        self.enc1 = MultiScaleReceptiveBlock(channels[0], channels[0])
        self.hfgm1 = HighFrequencyGatingModule(channels[0])
        
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = MultiScaleReceptiveBlock(channels[0], channels[1])
        self.hfgm2 = HighFrequencyGatingModule(channels[1])
        
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = MultiScaleReceptiveBlock(channels[1], channels[2])
        self.hfgm3 = HighFrequencyGatingModule(channels[2])
        
        self.down3 = nn.MaxPool2d(2)
        self.enc4 = MultiScaleReceptiveBlock(channels[2], channels[3])
        self.hfgm4 = HighFrequencyGatingModule(channels[3])
        
        # Bottleneck (Dilated MS-DRFB)
        self.down4 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            MultiScaleReceptiveBlock(channels[3], channels[3]),
            HighFrequencyGatingModule(channels[3])
        )
        
        # Decoder with Cross-Scale Skip Fusions
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cs_skip4 = CrossScaleBoundaryAwareSkipFusion(channels[3], channels[3], channels[3])
        self.dec4 = MultiScaleReceptiveBlock(channels[3], channels[2])
        
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cs_skip3 = CrossScaleBoundaryAwareSkipFusion(channels[2], channels[2], channels[2])
        self.dec3 = MultiScaleReceptiveBlock(channels[2], channels[1])
        
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cs_skip2 = CrossScaleBoundaryAwareSkipFusion(channels[1], channels[1], channels[1])
        self.dec2 = MultiScaleReceptiveBlock(channels[1], channels[0])
        
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cs_skip1 = CrossScaleBoundaryAwareSkipFusion(channels[0], channels[0], channels[0])
        self.dec1 = MultiScaleReceptiveBlock(channels[0], channels[0])
        
        # Auxiliary Boundary Contour Head
        self.boundary_head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0] // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0] // 2, num_classes, kernel_size=1)
        )
        
        # Primary Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0] // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0] // 2, num_classes, kernel_size=1)
        )

    def forward(self, x, return_boundary=False):
        # Stem
        x0 = self.stem(x)
        
        # Encoder
        e1 = self.hfgm1(self.enc1(x0))
        e2 = self.hfgm2(self.enc2(self.down1(e1)))
        e3 = self.hfgm3(self.enc3(self.down2(e2)))
        e4 = self.hfgm4(self.enc4(self.down3(e3)))
        
        # Bottleneck
        b = self.bottleneck(self.down4(e4))
        
        # Decoder with Cross-Scale Skip Fusions
        d4 = self.dec4(self.cs_skip4(e4, self.up4(b)))
        d3 = self.dec3(self.cs_skip3(e3, self.up3(d4)))
        d2 = self.dec2(self.cs_skip2(e2, self.up2(d3)))
        d1 = self.dec1(self.cs_skip1(e1, self.up1(d2)))
        
        seg_out = self.seg_head(d1)
        
        if self.training or return_boundary:
            bound_out = self.boundary_head(d1)
            return seg_out, bound_out
            
        return seg_out
