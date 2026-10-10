"""
Baseline 1: Fully Convolutional Network (FCN-8s)
Reference: Long, Shelhamer, Darrell. "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015 / IEEE TPAMI 2017.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class FCN8s(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(FCN8s, self).__init__()
        # Encoder based on VGG16 backbone
        vgg = models.vgg16(weights=None)
        features = list(vgg.features.children())
        
        self.conv1 = nn.Sequential(*features[0:5])    # 1/2
        self.conv2 = nn.Sequential(*features[5:10])   # 1/4
        self.conv3 = nn.Sequential(*features[10:17])  # 1/8  (pool3)
        self.conv4 = nn.Sequential(*features[17:24])  # 1/16 (pool4)
        self.conv5 = nn.Sequential(*features[24:31])  # 1/32 (pool5)
        
        # FC layers converted to conv
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
        self.score_fr = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Transposed convs
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        pool3 = h
        
        h = self.conv4(h)
        pool4 = h
        
        h = self.conv5(h)
        h = self.fc6(h)
        h = self.fc7(h)
        score_fr = self.score_fr(h)
        
        # FCN-8s multi-resolution skip connections
        upscore2 = self.upscore2(score_fr)
        score_pool4c = self.score_pool4(pool4)
        fuse_pool4 = upscore2 + score_pool4c
        
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3c = self.score_pool3(pool3)
        fuse_pool3 = upscore_pool4 + score_pool3c
        
        out = self.upscore8(fuse_pool3)
        return out
