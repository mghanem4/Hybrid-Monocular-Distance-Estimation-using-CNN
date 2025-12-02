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
DEFAULT_CONFIG = {
    'batch_size': 32,
    'lr': 0.001,
    'epochs': 20,
    'weight_decay': 1e-4,
    'lambda_reg': 1.0,
    'backbone': 'resnet18',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'save_dir': '.' # Default to current dir
}

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

        optimizer.zero_grad()
        # get class logits, and delta Z from the model.
        cls_logits, delta_z = model(imgs)
        # Hybrid Logic
        probs = F.softmax(cls_logits, dim=1) 
        H_bar = torch.sum(probs * class_heights.unsqueeze(0), dim=1, keepdim=True)
        # start with the pinhole estimate to get Z
        Z_pin = (f_x * H_bar) / (h_pixel + 1e-6) 
        # add Z_pin to the residual to get pred Z
        Z_pred = Z_pin + delta_z

        loss_cls = F.cross_entropy(cls_logits, cls_gt)
        loss_reg = F.smooth_l1_loss(Z_pred, z_gt)
        total_loss = loss_cls + (lambda_reg * loss_reg)
        
        total_loss.backward()
        optimizer.step()
        
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
            
            cls_logits, delta_z = model(imgs)
            
            probs = F.softmax(cls_logits, dim=1)
            H_bar = torch.sum(probs * class_heights.unsqueeze(0), dim=1, keepdim=True)
            Z_pin = (f_x * H_bar) / (h_pixel + 1e-6)
            Z_pred = Z_pin + delta_z
            
            preds_cls = torch.argmax(probs, dim=1)
            
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
    
    avg_mae = running_mae / total_samples
    rmse = np.sqrt(np.mean((all_preds_z - all_gt_z)**2))
    ss_res = np.sum((all_gt_z - all_preds_z)**2)
    ss_tot = np.sum((all_gt_z - np.mean(all_gt_z))**2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
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
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Data
    train_ds = KittiObjectDataset(mode='train', transform=train_transform)
    val_ds = KittiObjectDataset(mode='test', transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # 3. Model
    num_classes = len(train_ds.class_to_idx)
    model = HybridDistanceModel(num_classes=num_classes, backbone=config['backbone']).to(device)
    class_heights = get_class_height_tensor(train_ds, device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
            
            logger.info(f"  Train MAE: {train_mae:.2f}m | Val MAE: {metrics['mae']:.2f}m | Val Acc: {metrics['accuracy']:.2f}")

            # Save Best
            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
    
    logger.info("Training Complete.")
    return best_mae

def main():
    train_model(DEFAULT_CONFIG)

if __name__ == "__main__":
    main()
