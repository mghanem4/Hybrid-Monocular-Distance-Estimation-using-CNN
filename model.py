import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HybridDistanceModel(nn.Module):
    def __init__(self, num_classes: int = 3, backbone: str = 'custom'):
        super(HybridDistanceModel, self).__init__()
        
        self.backbone_type = backbone

        if backbone == 'resnet18':
            # Load pretrained ResNet18
            resnet = models.resnet18(pretrained=True)
            # Remove the final FC layer (fc) and keep everything else
            # ResNet18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
            # usage of children()[:-1] grabs everything up to avgpool
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            # ResNet18 output after avgpool is 512 channels
            self.flatten_dim = 512
            
        else: # 'custom'
            # ---------------------------------------------------------
            # CNN Backbone (Lightweight Feature Extractor)
            # Input: (Batch, 3, 128, 128)
            # ---------------------------------------------------------
            self.features = nn.Sequential(
                # Block 1: 128x128 -> 64x64 - Raw pixels (edges, lines)
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 2: 64x64 -> 32x32
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 3: 32x32 -> 16x16
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 4: 16x16 -> 8x8
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Flattened dimension: 256 channels * 8 * 8 spatial
            self.flatten_dim = 256 * 8 * 8

        # ---------------------------------------------------------
        # Classification Head 
        # Goal: Predict probabilities p_c for {Car, Ped, Cyc}
        # ---------------------------------------------------------
        self.cls_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes) # Output: Logits
        )

        # ---------------------------------------------------------
        # Regression Head (Residual)
        # Goal: Predict scalar correction delta_Z
        # ---------------------------------------------------------
        self.reg_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1) # Output: Scalar Delta Z
        )

    def forward(self, x):
        """
        Returns:
            cls_logits: Raw scores for classification (Batch, 3)
            delta_z: Predicted residual distance (Batch, 1)
        """
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        
        # Independent Heads
        cls_logits = self.cls_head(x)
        delta_z = self.reg_head(x)
        
        return cls_logits, delta_z

    def get_pinhole_estimate(self, focal_length, bbox_height_px, avg_height_m):
        """
        Calculates Z_pin based on the geometric formula.
        All inputs should be Tensors of shape (Batch,) or (Batch, 1)
        """
        # Avoid division by zero with a small epsilon
        epsilon = 1e-6
        Z_pin = (focal_length * avg_height_m) / (bbox_height_px + epsilon)
        return Z_pin
