import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import csv
import time

from dataset import KittiObjectDataset
from model import HybridDistanceModel


from loguru import logger
# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
# SUPERMODEL CONFIG
DEFAULT_CONFIG = {
    'batch_size': 64,
    'lr': 0.0001,
    'epochs': 20,
    'weight_decay': 0.0,
    'lambda_reg': 10,
    'backbone': 'resnet18',
    'step_size': 10,
    'dropout': 0.5,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'save_dir': '.' # Default to current dir
}
# DEFAULT_CONFIG = {
#     'batch_size': 32,
#     'lr': 0.0005,
#     'epochs': 20,
#     'weight_decay': 1e-4,
#     'lambda_reg': 1.0,
#     'backbone': 'resnet18',
#     'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     'save_dir': '.' # Default to current dir
# }
def compute_dataset_stats(dataset):
    """
    Computes the mean and standard deviation of the dataset for normalization.
    """
    logger.info("Computing dataset statistics...")
    loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for batch in loader:
        data = batch['image'] 
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    logger.info(f"Dataset Mean: {mean}, Std: {std}")
    return mean, std

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

def train_one_epoch(model, loader, optimizer, class_heights, device, lambda_reg):
    model.train()
    running_loss = 0.0
    running_mae = 0.0 
    
    total_batches = len(loader)
    
    for i, batch in enumerate(loader):
        imgs = batch['image'].to(device)
        cls_gt = batch['class_idx'].to(device)
        z_gt = batch['z_gt'].to(device).view(-1, 1) 
        f_x = batch['focal_length'].to(device).view(-1, 1)
        h_pixel = batch['bbox_height'].to(device).view(-1, 1)
        
        # Dynamic Image Dimensions
        W = batch['img_width'].to(device) 
        H = batch['img_height'].to(device)
        
        # --- Prepare Box Coords ---
        if 'box' not in batch:
            raise ValueError("Your dataset.py is not returning 'box', check __getitem__.")
        raw_boxes = batch['box'].to(device)
        
        # Normalize to 0-1 range
        bn_x1 = raw_boxes[:, 0] / W
        bn_y1 = raw_boxes[:, 1] / H
        bn_w  = (raw_boxes[:, 2] - raw_boxes[:, 0]) / W
        bn_h  = (raw_boxes[:, 3] - raw_boxes[:, 1]) / H
        
        box_coords = torch.stack([bn_x1, bn_y1, bn_w, bn_h], dim=1)

        optimizer.zero_grad()

        # --- Forward Pass ---
        cls_logits, delta_z = model(imgs, box_coords)
        
        # --- Hybrid Logic ---
        probs = F.softmax(cls_logits, dim=1) 
        H_bar = torch.sum(probs * class_heights.unsqueeze(0), dim=1, keepdim=True)
        Z_pin = (f_x * H_bar) / (h_pixel + 1e-6) 
        Z_pred = Z_pin + delta_z

        # --- Loss Calculation (Distance-Weighted) ---
        loss_cls = F.cross_entropy(cls_logits, cls_gt)
        
        # 1. Calculate raw error per sample (reduction='none' keeps the batch shape)
        raw_reg_loss = F.smooth_l1_loss(Z_pred, z_gt, reduction='none')
        
        # 2. Create Weights: Inverse of Ground Truth Distance
        # We add +1.0 to z_gt to avoid division by zero for objects at 0m
        # Effect: 
        #   - 4m away  -> weight 1/5  = 0.20
        #   - 49m away -> weight 1/50 = 0.02
        # The model now cares 10x more about the 4m object.
        dist_weights = 1.0 / (z_gt + 1.0)
        
        # 3. Apply weights and take the mean
        loss_reg = (raw_reg_loss * dist_weights).mean()
        
        total_loss = loss_cls + (lambda_reg * loss_reg)
        
        total_loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += total_loss.item()
        running_mae += torch.abs(Z_pred - z_gt).mean().item()
        
    return running_loss / total_batches, running_mae / total_batches
def evaluate(model, loader, class_heights, device):
    model.eval()
    
    all_preds_z = []
    all_gt_z = []
    all_gt_cls = []
    all_pred_cls = []
    predictions_by_img = {} 
    
    running_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            cls_gt = batch['class_idx'].to(device)
            z_gt = batch['z_gt'].to(device).view(-1, 1)
            img_ids = batch['img_id']
            f_x = batch['focal_length'].to(device).view(-1, 1)
            h_pixel = batch['bbox_height'].to(device).view(-1, 1)
            
            # ---------------------------------------------------------
            # NEW: Prepare Box Coords (Copied from train_one_epoch)
            # ---------------------------------------------------------
            W = batch['img_width'].to(device) 
            H = batch['img_height'].to(device)
            
            raw_boxes = batch['box'].to(device) 
            if raw_boxes.dim() > 2:
                raw_boxes = raw_boxes[:, :, 0]
                
            bn_x1 = raw_boxes[:, 0] / W
            bn_y1 = raw_boxes[:, 1] / H
            bn_w  = (raw_boxes[:, 2] - raw_boxes[:, 0]) / W
            bn_h  = (raw_boxes[:, 3] - raw_boxes[:, 1]) / H
            
            box_coords = torch.stack([bn_x1, bn_y1, bn_w, bn_h], dim=1)
            # ---------------------------------------------------------

            # UPDATE: Pass both inputs to the model
            cls_logits, delta_z = model(imgs, box_coords)
            # softmax prob to get class preds
            probs = F.softmax(cls_logits, dim=1)
            # get the height estimate of each class by weighting with probability, (i.e 50% car, 30% pedestrian, %20 truk so we get:)
            # H_bar = 0.5 * H_car + 0.3 * H_pedestrian + 0.2 * truck, where H is the average height of each class
            H_bar = torch.sum(probs * class_heights.unsqueeze(0), dim=1, keepdim=True)
            # then we calculate the z_pinhole estimate, using the equation.
            # f_x: focal length in pixels
            # H_bar, the estimated height
            # h_pixel: height of box in pixels
            # add epsilon to avoid division by zero
            Z_pin = (f_x * H_bar) / (h_pixel + 1e-6)
            Z_pred = Z_pin + delta_z
            # argmax of probability to get class 
            preds_cls = torch.argmax(probs, dim=1)
            # Send to cpu to convert tensors to numpy for compatiability with other libraries
            z_pred_np = Z_pred.cpu().numpy().flatten()
            z_gt_np = z_gt.cpu().numpy().flatten()
            cls_gt_np = cls_gt.cpu().numpy()
            preds_cls_np = preds_cls.cpu().numpy()
            
            all_preds_z.extend(z_pred_np)
            all_gt_z.extend(z_gt_np)
            all_gt_cls.extend(cls_gt_np)
            all_pred_cls.extend(preds_cls_np)
            
            for i, img_id in enumerate(img_ids):
                if img_id not in predictions_by_img:
                    predictions_by_img[img_id] = []
                predictions_by_img[img_id].append({
                    'z_pred': float(z_pred_np[i]),
                    'z_gt': float(z_gt_np[i])
                })

            running_mae += np.sum(np.abs(z_pred_np - z_gt_np))
            total_samples += len(z_gt_np)

    all_preds_z = np.array(all_preds_z)
    all_gt_z = np.array(all_gt_z)
    all_gt_cls = np.array(all_gt_cls)
    all_pred_cls = np.array(all_pred_cls)
    
    # Calculate statistics for eval
    # average mae
    avg_mae = running_mae / total_samples
    # RMSE for distances 
    rmse = np.sqrt(np.mean((all_preds_z - all_gt_z)**2))
    # SS Residual and SSTO for R^2 calculation
    ss_res = np.sum((all_gt_z - all_preds_z)**2)
    ss_tot = np.sum((all_gt_z - np.mean(all_gt_z))**2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    # overall acc of classification
    acc = np.mean(all_gt_cls == all_pred_cls)
    
    unique_classes = np.unique(all_gt_cls)
    per_class_mae = {}
    for cls_idx in unique_classes:
        mask = (all_gt_cls == cls_idx)
        if np.sum(mask) > 0:
            c_mae = np.mean(np.abs(all_preds_z[mask] - all_gt_z[mask]))
            per_class_mae[int(cls_idx)] = c_mae
            
    nearest_correct_count = 0
    images_with_multiple_objects = 0
    
    for img_id, objs in predictions_by_img.items():
        if len(objs) < 1: continue
        objs_gt = [o['z_gt'] for o in objs]
        idx_closest_gt = np.argmin(objs_gt)
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
def train_model(config=DEFAULT_CONFIG):
    """
    Main reusable training function.
    """
    device = config['device']
    logger.info(f"Starting Training with config: {config}")

    # 1. Transforms
    
    # --- STEP 1: Compute Stats ---
    if config['backbone'] == 'custom':
        logger.info("Computing dataset statistics for custom backbone...")
        # Basic transform without Normalize
        temp_transform = transforms.Compose([
            transforms.Resize((128, 128)), # Ensure size matches
            transforms.ToTensor()
        ])
        temp_ds = KittiObjectDataset(mode='train', transform=temp_transform)
        mean, std = compute_dataset_stats(temp_ds)
        norm_mean = mean.tolist()
        norm_std = std.tolist()
    else:
        # Use ImageNet stats for ResNet
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    logger.info(f"Using Normalization -> Mean: {norm_mean}, Std: {norm_std}")

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # 2. Data
    train_ds = KittiObjectDataset(mode='train', transform=train_transform)
    val_ds = KittiObjectDataset(mode='test', transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    h_dim = config.get('hidden_dim', 512)
    drop_p = config.get('dropout', 0.5)
    # 3. Model
    num_classes = len(train_ds.class_to_idx)
    model = HybridDistanceModel(num_classes=num_classes, hidden_dim=h_dim,
        dropout=drop_p,
        backbone=config['backbone']).to(device)
    class_heights = get_class_height_tensor(train_ds, device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get('step_size', 7), gamma=0.1)

    # Prepare CSV Logging
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'metrics.csv')
    
    best_mae = float('inf')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_mae', 'val_mae', 'val_acc', 'val_rmse', 'val_r2', 'val_nearest_acc'])

        for epoch in range(config['epochs']):
            logger.info(f"Epoch {epoch+1}/{config['epochs']}")
            
            train_loss, train_mae = train_one_epoch(
                model, train_loader, optimizer, class_heights, device, config['lambda_reg']
            )
            
            metrics = evaluate(model, val_loader, class_heights, device)
            
            scheduler.step()
            
            # Log to CSV
            writer.writerow([
                epoch+1, 
                train_loss, 
                train_mae, 
                metrics['mae'], 
                metrics['accuracy'],
                metrics['rmse'],
                metrics['r2'],
                metrics['nearest_acc']
            ])
            
            logger.debug(f"  Train MAE: {train_mae:.2f}m | Val MAE: {metrics['mae']:.2f}m | Val Acc: {metrics['accuracy']:.2f}")

            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                
                # Create a comprehensive dictionary
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mae': best_mae,
                    # --- SAVE METADATA ---
                    'class_to_idx': train_ds.class_to_idx,
                    'avg_heights': train_ds.avg_heights,
                    'config': config
                }
                logger.info(checkpoint_data)                
                torch.save(checkpoint_data, os.path.join(save_dir, 'best_model_supermodel.pth'))
    logger.debug("Training Complete.")
    return best_mae

def main():
    train_model(DEFAULT_CONFIG)

if __name__ == "__main__":
    main()
