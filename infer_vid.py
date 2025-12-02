import torch
import cv2
import numpy as np
import argparse
import time
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torch.nn.functional as F

from model import HybridDistanceModel

# --------------------------------------------------------------------------
# CONSTANTS & MAPPINGS
# --------------------------------------------------------------------------
# (Keep your existing mappings - they are correct)
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
def get_object_detector(device):
    print("Loading Object Detector (Faster R-CNN)...")
    # Using the default weights (ResNet50)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model

def load_distance_model(checkpoint_path, device):
    print(f"Loading Distance Model from {checkpoint_path}...")
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
    """
    Takes a raw BGR frame (from opencv), runs detection + distance, 
    and returns the annotated frame.
    """
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
    
    # 4. Process Each Object
    for i, (box, label_idx) in enumerate(zip(boxes, labels)):
        if label_idx not in COCO_TO_KITTI:
            continue
            
        kitti_class = COCO_TO_KITTI[int(label_idx)] # Safety cast
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
        
        # Distance Estimation
        crop_tensor = preprocess_crop(crop).to(device)
        h_pixel = max(1, y2 - y1) 

        with torch.no_grad():
            cls_logits, delta_z = dist_model(crop_tensor)
            Z_pin = (focal_length * avg_h) / h_pixel
            Z_final = Z_pin + delta_z.item()
        
        # --- DRAWING ---
        color = CLASS_COLORS.get(kitti_class, (255, 255, 255))
        label_text = f"{kitti_class} {Z_final:.1f}m"

        # Text Background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Label Position (Smart)
        lx, ly = x1, y1 - 5
        if ly - th < 0: ly = y1 + th + 10 # Move inside if off-screen
            
        cv2.rectangle(frame, (lx, ly - th - baseline), (lx + tw, ly + baseline), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (lx, ly), font, font_scale, (255, 255, 255), thickness)

    return frame

# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help="Camera ID (0) or Video Path")
    parser.add_argument('--checkpoint', type=str, default='best_model.pth')
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

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Process Frame
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