import torch
import torch.nn as nn

class SimplePainClassifier(nn.Module):
    """Simplified 2D CNN for pain recognition (more suitable for small datasets)"""
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(SimplePainClassifier, self).__init__()
        
        # Process each modality independently with 2D CNN
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.thermal_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, rgb, depth, thermal):
        # rgb: [B, 3, T, H, W] -> Average over time -> [B, 3, H, W]
        rgb = rgb.mean(dim=2)
        depth = depth.mean(dim=2)
        thermal = thermal.mean(dim=2)
        
        # Process each branch
        rgb_feat = self.rgb_branch(rgb).view(rgb.size(0), -1)
        depth_feat = self.depth_branch(depth).view(depth.size(0), -1)
        thermal_feat = self.thermal_branch(thermal).view(thermal.size(0), -1)
        
        # Concatenate and classify
        fused = torch.cat([rgb_feat, depth_feat, thermal_feat], dim=1)
        logits = self.fusion(fused)
        return logits
