import torch
import cv2
import numpy as np
import argparse
import time
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

from model import HybridDistanceModel

# --------------------------------------------------------------------------
# CONSTANTS & MAPPINGS
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
'''

NOTE: THIS USSES MOBILE DETECTER FOR SPEED
'''
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------


KITTI_CLASS_TO_IDX = {
    'Car': 0, 'Cyclist': 1, 'Misc': 2, 'Pedestrian': 3, 
    'Person_sitting': 4, 'Tram': 5, 'Truck': 6, 'Van': 7
}

AVG_HEIGHTS = {
    'Pedestrian': 1.76, 'Car': 1.53, 'Van': 2.21, 
    'Person_sitting': 1.27, 'Cyclist': 1.74, 'Truck': 3.25, 
    'Misc': 1.92, 'Tram': 3.53
}

COCO_TO_KITTI = {
    1: 'Pedestrian', 2: 'Cyclist', 3: 'Car', 
    4: 'Cyclist', 6: 'Truck', 8: 'Truck'
}

CLASS_COLORS = {
    'Car': (0, 255, 0), 'Van': (0, 200, 200), 'Truck': (0, 165, 255),
    'Pedestrian': (0, 0, 255), 'Person_sitting': (50, 50, 200),
    'Cyclist': (255, 0, 0), 'Tram': (255, 0, 255), 'Misc': (128, 128, 128)
}

# --------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------

# 1. Update Imports
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, 
    SSDLite320_MobileNet_V3_Large_Weights
)

def get_object_detector(device):
    print("Loading Object Detector (SSDLite 320)...")
    
    # This model is optimized for 320x320 resolution inference
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    
    model.to(device)
    model.eval()
    return model
def load_distance_model(checkpoint_path, device):
    print(f"Loading Distance Model from {checkpoint_path}...")
    # NOTE: Ensure this matches your trained model (e.g. hidden_dim=512)
    model = HybridDistanceModel(num_classes=8, backbone='resnet18') 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def preprocess_crop(crop_img):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(crop_img).unsqueeze(0)

# --------------------------------------------------------------------------
# PROCESSING PIPELINE
# --------------------------------------------------------------------------
def process_frame(frame, detector, dist_model, device, focal_length):
    # 1. Prepare Image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb).to(device).unsqueeze(0)

    # 2. Object Detection
    with torch.no_grad():
        detections = detector(img_tensor)[0]

    # 3. Filter Results
    threshold = 0.5
    keep_indices = detections['scores'] > threshold
    boxes = detections['boxes'][keep_indices].cpu().numpy()
    labels = detections['labels'][keep_indices].cpu().numpy()
    
    h_img, w_img, _ = frame.shape
    
    # --- BATCH PREPARATION ---
    valid_detections = [] # To store metadata for later drawing
    crop_tensors = []     # To store images for batch inference
    box_tensors = []      # To store coords for batch inference
    
    for i, (box, label_idx) in enumerate(zip(boxes, labels)):
        if label_idx not in COCO_TO_KITTI:
            continue
            
        kitti_class = COCO_TO_KITTI[int(label_idx)]
        avg_h = AVG_HEIGHTS[kitti_class]

        # Coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Safe Crop Logic
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        x1_p, y1_p = max(0, x1 - pad_w), max(0, y1 - pad_h)
        x2_p, y2_p = min(w_img, x2 + pad_w), min(h_img, y2 + pad_h)
        
        crop = Image.fromarray(img_rgb[y1_p:y2_p, x1_p:x2_p])
        if crop.size[0] == 0 or crop.size[1] == 0: continue
        
        # Preprocess Crop
        crop_t = preprocess_crop(crop) # Returns (1, 3, 128, 128)
        crop_tensors.append(crop_t)
        
        # Preprocess Coords
        norm_x = x1 / w_img
        norm_y = y1 / h_img
        norm_w = (x2 - x1) / w_img
        norm_h = (y2 - y1) / h_img
        box_t = torch.tensor([norm_x, norm_y, norm_w, norm_h], dtype=torch.float32).unsqueeze(0)
        box_tensors.append(box_t)
        
        # Store metadata to draw later
        valid_detections.append({
            'box': (x1, y1, x2, y2),
            'class': kitti_class,
            'avg_h': avg_h,
            'h_pixel': max(1, y2 - y1),
            'id': i+1
        })

    # If no objects found, return early
    if not valid_detections:
        return frame

    # --- BATCH INFERENCE (1 Run instead of N) ---
    # Stack lists into tensors: (Batch_Size, C, H, W)
    batch_crops = torch.cat(crop_tensors).to(device)
    batch_boxes = torch.cat(box_tensors).to(device)

    with torch.no_grad():
        cls_logits, delta_z = dist_model(batch_crops, batch_boxes)
        # delta_z is shape (Batch, 1)

    # --- DRAWING LOOP ---
    for i, det in enumerate(valid_detections):
        # Retrieve batch result
        d_z = delta_z[i].item()
        
        # Calculate Pinhole
        Z_pin = (focal_length * det['avg_h']) / det['h_pixel']
        Z_final = Z_pin + d_z
        
        # Draw
        x1, y1, x2, y2 = det['box']
        kitti_class = det['class']
        color = CLASS_COLORS.get(kitti_class, (255, 255, 255))
        label_text = f"{kitti_class} {Z_final:.1f}m"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label_text, font, 0.5, 1)
        
        lx, ly = x1, y1 - 5
        if ly - th < 0: ly = y1 + th + 10
            
        cv2.rectangle(frame, (lx, ly - th - baseline), (lx + tw, ly + baseline), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (lx, ly), font, 0.5, (255, 255, 255), 1)

    return frame

# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help="Camera ID (0) or Video Path")
    parser.add_argument('--checkpoint', type=str, default='experiments/exp_01_lr0.001_resnet18/best_model.pth')
    # Use a lower focal length for webcams (usually ~600-800 pixels width)
    parser.add_argument('--focal', type=float, default=800.0, help="Approx focal length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Load Models
    detector = get_object_detector(device)
    dist_model = load_distance_model(args.checkpoint, device)

    # Handle Source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Starting Live Demo... Press 'q' to exit.")
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Resize for speed if needed (e.g. 640 width)
        # frame = cv2.resize(frame, (640, 480)) 
        
        result_frame = process_frame(frame, detector, dist_model, device, args.focal)

        # Draw FPS
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hybrid Distance Estimation Demo', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()