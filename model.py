import torch
import torch.nn as nn
import torchvision.models as models

class HybridDistanceNet(nn.Module):
    def __init__(self, pretrained=True):
        super(HybridDistanceNet, self).__init__()

        # Backbone: ResNet18
        # We strip the last FC layer
# instead of initializing the ResNet18 model with random, garbage numbers, you are downloading and loading a set of weights (parameters) that have already been optimized on a massive dataset.
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Head 1: Classification (Car, Ped, Cyc)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Dropout(0.5)
        )

        # Head 2: Residual Regression (Delta Z)
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Dropout(0.5)
        )

        # Average Heights buffer (Car, Ped, Cyc)
        # We register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer('avg_heights', torch.tensor([1.53, 1.76, 1.73]))

    def forward(self, x, bbox_heights, focal_lengths):
        """
        x: [B, 3, H, W] images
        bbox_heights: [B] pixel heights
        focal_lengths: [B] focal lengths in pixels
        """
        # 1. Feature Extraction
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten [B, 512]

        # 2. Heads
        class_logits = self.classifier(x) # [B, 3]
        delta_z = self.regressor(x)       # [B, 1]

        # 3. Physics Anchor (Pinhole)
        # Softmax to get class probabilities
        probs = torch.softmax(class_logits, dim=1) # [B, 3]

        # Expected real-world height: sum(prob_c * H_c)
        # Broadcasting: [B,3] * [3] -> [B,3] -> sum -> [B]
        avg_h_pred = torch.sum(probs * self.avg_heights, dim=1)

        # Z_pin = (f * H) / h
        # Ensure proper shapes for division
        z_pinhole = (focal_lengths * avg_h_pred) / bbox_heights

        # 4. Final Combination
        z_final = z_pinhole.view(-1, 1) + delta_z

        return z_final, class_logits, z_pinhole
