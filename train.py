import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dataset import KittiObjectDataset
from model import HybridDistanceModel

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DECAY = 1e-4 # Regularization
LAMBDA_REG = 1.0    # Balance between Class Loss and Distance Loss

def get_class_height_tensor(dataset: KittiObjectDataset, device):
    """
    Creates a tensor of average heights ordered by class ID [0, 1, 2...]
    Needed for the Hybrid Pinhole calculation.
    """
    cls_map = dataset.class_to_idx 
    avg_heights = dataset.avg_heights 
    
    num_classes = len(cls_map)
    height_vector = torch.zeros(num_classes).to(device)
    
    for name, idx in cls_map.items():
        h = avg_heights.get(name, 1.5) 
        height_vector[idx] = h
        
    return height_vector

def train_one_epoch(model, loader, optimizer, class_heights, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0 
    
    total_batches = len(loader)
    
    for i, batch in enumerate(loader):
        # 1. Move data to GPU/Device
        imgs = batch['image'].float().to(device) 
        imgs = imgs / 255.0 
        
        # Targets
        cls_gt = batch['class_idx'].to(device)
        z_gt = batch['z_gt'].to(device).view(-1, 1) 
        
        # Metadata for Pinhole
        f_x = batch['focal_length'].to(device).view(-1, 1)
        h_pixel = batch['bbox_height'].to(device).view(-1, 1)

        # 2. Forward Pass
        optimizer.zero_grad()
        cls_logits, delta_z = model(imgs)
        
        # -----------------------------------------------------------
        # 3. HYBRID LOGIC
        # -----------------------------------------------------------
        
        # A. Calculate Class Probabilities
        probs = F.softmax(cls_logits, dim=1) 
        
        # B. Calculate Weighted Average Height (H_bar)
        H_bar = torch.sum(probs * class_heights.unsqueeze(0), dim=1, keepdim=True)
        
        # C. Calculate Pinhole Anchor (Z_pin)
        Z_pin = (f_x * H_bar) / (h_pixel + 1e-6) 
        
        # D. Final Prediction
        Z_pred = Z_pin + delta_z

        # -----------------------------------------------------------
        # 4. LOSS CALCULATION
        # -----------------------------------------------------------
        loss_cls = F.cross_entropy(cls_logits, cls_gt)
        loss_reg = F.smooth_l1_loss(Z_pred, z_gt)
        
        total_loss = loss_cls + (LAMBDA_REG * loss_reg)
        
        # 5. Backward Pass
        total_loss.backward()
        optimizer.step()
        
        # Stats
        loss_item = total_loss.item()
        mae_item = torch.abs(Z_pred - z_gt).mean().item()
        running_loss += loss_item
        running_mae += mae_item
        
        # Simple logging every 10 batches
        if i % 10 == 0:
            print(f"  Batch {i}/{total_batches} | Loss: {loss_item:.4f} | MAE: {mae_item:.2f}m")

    return running_loss / total_batches, running_mae / total_batches

def evaluate(model, loader, class_heights, device):
    model.eval()
    total_mae = 0.0
    correct_cls = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].float().to(device) / 255.0
            cls_gt = batch['class_idx'].to(device)
            z_gt = batch['z_gt'].to(device).view(-1, 1)
            
            f_x = batch['focal_length'].to(device).view(-1, 1)
            h_pixel = batch['bbox_height'].to(device).view(-1, 1)
            
            # Forward
            cls_logits, delta_z = model(imgs)
            
            # Hybrid Calculation
            probs = F.softmax(cls_logits, dim=1)
            H_bar = torch.sum(probs * class_heights.unsqueeze(0), dim=1, keepdim=True)
            Z_pin = (f_x * H_bar) / (h_pixel + 1e-6)
            Z_pred = Z_pin + delta_z
            
            # Metrics
            total_mae += torch.abs(Z_pred - z_gt).sum().item()
            preds = torch.argmax(probs, dim=1)
            correct_cls += (preds == cls_gt).sum().item()
            total_samples += z_gt.size(0)

    avg_mae = total_mae / total_samples
    accuracy = correct_cls / total_samples
    return avg_mae, accuracy

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
def main():
    print(f"Starting Training on {DEVICE}")
    
    # 1. Prepare Data
    print("Loading Training Set...")
    train_ds = KittiObjectDataset(mode='train')
    print("Loading Validation Set...")
    val_ds = KittiObjectDataset(mode='test')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Prepare Model
    num_classes = len(train_ds.class_to_idx)
    model = HybridDistanceModel(num_classes=num_classes).to(DEVICE)
    
    # Get the class heights vector for the Hybrid formula
    class_heights = get_class_height_tensor(train_ds, DEVICE)
    print(f"Class Heights Vector: {class_heights}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 3. Training Loop
    best_mae = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{EPOCHS} {'='*20}")
        
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, class_heights, DEVICE)
        val_mae, val_acc = evaluate(model, val_loader, class_heights, DEVICE)
        
        scheduler.step()
        
        print(f"\n[Epoch {epoch+1} RESULT]")
        print(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f}m")
        print(f"Val MAE:    {val_mae:.2f}m   | Val Acc:   {val_acc*100:.1f}%")
        
        # Save Best Model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print(">>> New Best Model Saved!")

if __name__ == "__main__":
    main()