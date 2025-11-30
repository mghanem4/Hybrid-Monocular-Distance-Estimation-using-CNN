import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from loguru import logger

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
BACKBONE = 'custom' # Options: 'custom', 'resnet18'

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

def train_one_epoch(model:HybridDistanceModel, loader: DataLoader, optimizer: optim, class_heights: torch.Tensor, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0 
    
    total_batches = len(loader)
    
    for i, batch in enumerate(loader):
        # 1. Move data to GPU/Device
        # Note: Dataset with transforms already returns FloatTensor [0,1], no need for / 255.0
        imgs = batch['image'].to(device)
        
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
            logger.info(f"  Batch {i}/{total_batches} | Loss: {loss_item:.4f} | MAE: {mae_item:.2f}m")

    return running_loss / total_batches, running_mae / total_batches

def evaluate(model, loader, class_heights, device):
    model.eval()
    
    # Storage for metrics calculation
    all_preds_z = []
    all_gt_z = []
    all_gt_cls = []
    all_pred_cls = []
    
    # For Nearest Object Accuracy
    # Store tuples: (img_id, distance_pred, distance_gt, is_this_the_closest_object)
    # Actually, we need to group by img_id later.
    predictions_by_img = {} # {img_id: [{'z_pred': ..., 'z_gt': ...}]}
    
    total_samples = 0
    running_mae = 0.0

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            cls_gt = batch['class_idx'].to(device)
            z_gt = batch['z_gt'].to(device).view(-1, 1)
            img_ids = batch['img_id'] # List of strings/ints
            
            f_x = batch['focal_length'].to(device).view(-1, 1)
            h_pixel = batch['bbox_height'].to(device).view(-1, 1)
            
            # Forward
            cls_logits, delta_z = model(imgs)
            
            # Hybrid Calculation
            probs = F.softmax(cls_logits, dim=1)
            H_bar = torch.sum(probs * class_heights.unsqueeze(0), dim=1, keepdim=True)
            Z_pin = (f_x * H_bar) / (h_pixel + 1e-6)
            Z_pred = Z_pin + delta_z
            
            # Accumulate Results
            preds_cls = torch.argmax(probs, dim=1)
            
            # Move to CPU for metrics
            z_pred_np = Z_pred.cpu().numpy().flatten()
            z_gt_np = z_gt.cpu().numpy().flatten()
            cls_gt_np = cls_gt.cpu().numpy()
            preds_cls_np = preds_cls.cpu().numpy()
            
            all_preds_z.extend(z_pred_np)
            all_gt_z.extend(z_gt_np)
            all_gt_cls.extend(cls_gt_np)
            all_pred_cls.extend(preds_cls_np)
            
            # Organize for Nearest Object Accuracy
            for i, img_id in enumerate(img_ids):
                if img_id not in predictions_by_img:
                    predictions_by_img[img_id] = []
                predictions_by_img[img_id].append({
                    'z_pred': float(z_pred_np[i]),
                    'z_gt': float(z_gt_np[i])
                })

            running_mae += np.sum(np.abs(z_pred_np - z_gt_np))
            total_samples += len(z_gt_np)

    # -----------------------
    # COMPUTE METRICS
    # -----------------------
    all_preds_z = np.array(all_preds_z)
    all_gt_z = np.array(all_gt_z)
    all_gt_cls = np.array(all_gt_cls)
    all_pred_cls = np.array(all_pred_cls)
    
    # 1. Global MAE
    avg_mae = running_mae / total_samples
    
    # 2. RMSE
    rmse = np.sqrt(np.mean((all_preds_z - all_gt_z)**2))
    
    # 3. R2 Score
    ss_res = np.sum((all_gt_z - all_preds_z)**2)
    ss_tot = np.sum((all_gt_z - np.mean(all_gt_z))**2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    # 4. Classification Accuracy
    acc = np.mean(all_gt_cls == all_pred_cls)
    
    # 5. Per-Class MAE
    unique_classes = np.unique(all_gt_cls)
    per_class_mae = {}
    for cls_idx in unique_classes:
        mask = (all_gt_cls == cls_idx)
        if np.sum(mask) > 0:
            c_mae = np.mean(np.abs(all_preds_z[mask] - all_gt_z[mask]))
            per_class_mae[int(cls_idx)] = c_mae
            
    # 6. Nearest Object Accuracy
    # For each image, find the object with min GT distance.
    # Check if the object with min Predicted distance is the same object (index matching).
    # Since we don't have unique object IDs, we rely on the list order within the image.
    # However, 'predictions_by_img' stores a list of predictions for that image.
    # We assume the order of objects in the list corresponds to distinct objects.
    nearest_correct_count = 0
    images_with_multiple_objects = 0
    
    for img_id, objs in predictions_by_img.items():
        if len(objs) < 1: continue
        
        # Find index of closest object by Ground Truth
        # (Using stable sort logic or argmin)
        objs_gt = [o['z_gt'] for o in objs]
        idx_closest_gt = np.argmin(objs_gt)
        
        # Find index of closest object by Prediction
        objs_pred = [o['z_pred'] for o in objs]
        idx_closest_pred = np.argmin(objs_pred)
        
        if idx_closest_gt == idx_closest_pred:
            nearest_correct_count += 1
            
        images_with_multiple_objects += 1

    nearest_acc = nearest_correct_count / max(1, images_with_multiple_objects)

    return {
        'mae': avg_mae,
        'rmse': rmse,
        'r2': r2_score,
        'accuracy': acc,
        'per_class_mae': per_class_mae,
        'nearest_acc': nearest_acc
    }

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
def main():
    logger.info(f"Starting Training on {DEVICE}")
    logger.info(f"Backbone: {BACKBONE}")
    
    # 1. Define Transforms
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), # Converts to [0,1], CHW
        # Optionally add Normalize if using Pretrained models strictly, 
        # but ResNet handles 0-1 reasonably well or learns the shift. 
        # Ideally: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Prepare Data
    logger.info("Loading Training Set...")
    train_ds = KittiObjectDataset(mode='train', transform=train_transform)
    logger.info("Loading Validation Set...")
    val_ds = KittiObjectDataset(mode='test', transform=val_transform)
    
    # Note: colalte_fn default handles strings in a list, which is what we need for img_id
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 3. Prepare Model
    num_classes = len(train_ds.class_to_idx)
    model = HybridDistanceModel(num_classes=num_classes, backbone=BACKBONE).to(DEVICE)
    
    # Get the class heights vector for the Hybrid formula
    class_heights = get_class_height_tensor(train_ds, DEVICE)
    logger.info(f"Class Heights Vector: {class_heights}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Training Loop
    best_mae = float('inf')
    
    for epoch in range(EPOCHS):
        logger.info(f"\n{'='*20} Epoch {epoch+1}/{EPOCHS} {'='*20}")
        
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, class_heights, DEVICE)
        metrics = evaluate(model, val_loader, class_heights, DEVICE)
        
        val_mae = metrics['mae']
        val_acc = metrics['accuracy']
        
        scheduler.step()
        
        logger.info(f"\n[Epoch {epoch+1} RESULT]")
        logger.info(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f}m")
        logger.info(f"Val MAE:    {val_mae:.2f}m   | Val Acc:   {val_acc*100:.1f}%")
        logger.info(f"Val RMSE:   {metrics['rmse']:.2f}m   | R2 Score:  {metrics['r2']:.3f}")
        logger.info(f"Nearest Obj Acc: {metrics['nearest_acc']*100:.1f}%")
        logger.info(f"Per Class MAE: {metrics['per_class_mae']}")
        
        # Save Best Model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            logger.debug(">>> New Best Model Saved!")
if __name__ == "__main__":
    main()
