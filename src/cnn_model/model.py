import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

class HighPassFilter(nn.Module):
    def __init__(self):
        super(HighPassFilter, self).__init__()
        # Standard KV kernel (5x5) used in steganalysis/forensics
        kernel = np.array([[-1, 2, -2, 2, -1],
                           [ 2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [ 2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
        
        # Reshape to (Out=1, In=1, H=5, W=5)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        
        # We want to apply this to each of the 3 RGB channels independently.
        # So we repeat it 3 times to get (Out=3, In=1, H=5, W=5)
        # And use groups=3 in conv2d.
        self.weight = nn.Parameter(data=kernel.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        # Apply the high-pass filter to each channel
        return F.conv2d(x, self.weight, padding=2, groups=3)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SimpleCNN, self).__init__()
        
        # 1. Preprocessing Layer: Fixed High-Pass Filter
        self.hpf = HighPassFilter()
        
        # 2. Backbone: ResNet18
        # We use ResNet18 as a powerful feature extractor.
        # We don't use pretrained weights because the domain (noise residuals) is very different from ImageNet.
        # However, the architecture is excellent for this task.
        self.backbone = resnet18(weights=None)
        
        # Replace the first conv layer to accept 3 channels (which it does by default, but let's be explicit if needed)
        # ResNet18 conv1 is: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Our HPF outputs 3 channels, so this matches perfectly.
        
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Step 1: Extract Residuals
        x = self.hpf(x)
        
        # Step 2: ResNet Backbone (Features + Classification)
        x = self.backbone(x)
        
        return x