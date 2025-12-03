import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class HybridDistanceModel(nn.Module):
    def __init__(self, num_classes: int = 3, backbone: str = 'resnet18', 
                 hidden_dim: int = 512, dropout: float = 0.5):
        """
        Args:
            num_classes: Number of output classes (e.g., 8 for KITTI).
            backbone: 'resnet18' or 'custom'.
            hidden_dim: Size of the hidden layer in the heads (default 512).
            dropout: Dropout probability (default 0.5).
        """
        super(HybridDistanceModel, self).__init__()
        
        self.backbone_type = backbone
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # ---------------------------------------------------------
        # 1. Image Backbone (The "Eye")
        # ---------------------------------------------------------
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            img_feature_dim = 512  # ResNet output size
            
        else: # 'custom'
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                # Block 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            )
            img_feature_dim = 256 * 8 * 8 # Custom output size

        # ---------------------------------------------------------
        # 2. Geometric MLP
        # Processes normalized coordinates (x, y, w, h)
        # ---------------------------------------------------------
        self.geo_dim = 32
        self.geo_mlp = nn.Sequential(
            nn.Linear(4, self.geo_dim), # Input: 4 coords -> Output: 32 features
            nn.ReLU()
        )

        # Calculate Total Dimension for the Heads
        # Image Features + Geometric Features
        self.flatten_dim = img_feature_dim + self.geo_dim

        # ---------------------------------------------------------
        # 3. Heads (Now taking the combined input)
        # ---------------------------------------------------------
        self.cls_head = nn.Sequential(
            nn.Linear(self.flatten_dim, self.hidden_dim), # <--- Dynamic Hidden Dim
            nn.ReLU(),
            nn.Dropout(self.dropout),                     # <--- Dynamic Dropout
            nn.Linear(self.hidden_dim, num_classes)
        )

        self.reg_head = nn.Sequential(
            nn.Linear(self.flatten_dim, self.hidden_dim), # <--- Dynamic Hidden Dim
            nn.ReLU(),
            nn.Dropout(self.dropout),                     # <--- Dynamic Dropout
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, box_coords):
        """
        Args:
            x: Image crop batch (B, 3, 128, 128)
            box_coords: Normalized box coordinates (B, 4)
                        [x_norm, y_norm, w_norm, h_norm]
        """
        # A. Process Image
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten image features
        
        # B. Process Coordinates
        geo = self.geo_mlp(box_coords) # (B, 32)
        
        # C. Fusion (Concatenate)
        combined = torch.cat((x, geo), dim=1) # (B, img_dim + 32)
        
        # D. Predict
        cls_logits = self.cls_head(combined)
        delta_z = self.reg_head(combined)
        
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