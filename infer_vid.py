import torch
import cv2
import numpy as np
import argparse
import time
from torchvision import transforms
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, 
    SSDLite320_MobileNet_V3_Large_Weights
)
from PIL import Image
import torch.nn.functional as F

# Ensure model.py is in the same directory
try:
    from model import HybridDistanceModel
except ImportError:
    print("Error: Could not import 'HybridDistanceModel'. Make sure model.py is in this folder.")
    exit(1)

# --------------------------------------------------------------------------
# CONSTANTS & GLOBALS
# --------------------------------------------------------------------------

# These will be populated by the load_custom_model function from the checkpoint
IDX_TO_CLASS = {}
AVG_HEIGHTS = {}
CLASS_COLORS = {}

# COCO IDs to accept from the Region Proposer (SSDLite)
# We accept these blobs, but let OUR model decide the specific class and height.
COCO_INTEREST_IDS = [
    1, # Person
    2, # Bicycle
    3, # Car
    4, # Motorcycle
    6, # Bus
    8, # Truck
]

def generate_colors(classes):
    """
    Generates distinct colors for the loaded classes.
    """
    colors = {}
    # Standard palette for common classes to keep it readable
    standard_palette = {
        'Car': (0, 255, 0),        'Van': (0, 255, 255),
        'Truck': (0, 165, 255),    'Pedestrian': (0, 0, 255),
        'Person_sitting': (0, 0, 128), 'Cyclist': (255, 0, 0),
        'Tram': (255, 0, 255),     'Misc': (128, 128, 128)
    }
    
    for cls_name in classes:
        if cls_name in standard_palette:
            colors[cls_name] = standard_palette[cls_name]
        else:
            # Random color for unknown classes
            np.random.seed(sum(map(ord, cls_name)))
            colors[cls_name] = tuple(np.random.randint(0, 255, 3).tolist())
    return colors

# --------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------

def get_object_detector(device):
    """
    Loads SSDLite (MobileNetV3) for fast region proposal.
    """
    print("Loading Object Detector (SSDLite 320)...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.to(device)
    model.eval()
    return model

def load_custom_model(checkpoint_path, device, backbone_type='resnet18'):
    """
    Loads your trained HybridDistanceModel and extracts metadata.
    """
    print(f"Loading Custom Model from {checkpoint_path}...")
    
    # Load the file first to check metadata
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ---------------------------------------------------------
    # 1. EXTRACT METADATA
    # ---------------------------------------------------------
    global IDX_TO_CLASS, AVG_HEIGHTS, CLASS_COLORS
    
    if 'class_to_idx' in checkpoint and 'avg_heights' in checkpoint:
        print("Metadata found in checkpoint. Updating globals...")
        class_to_idx = checkpoint['class_to_idx']
        AVG_HEIGHTS = checkpoint['avg_heights']
        
        # Create reverse mapping
        IDX_TO_CLASS = {v: k for k, v in class_to_idx.items()}
        # Generate colors
        CLASS_COLORS = generate_colors(class_to_idx.keys())
        
        num_classes = len(class_to_idx)
    else:
        # Fallback if using an old checkpoint without metadata
        print("[WARNING] No metadata in checkpoint. Using hardcoded defaults.")
        KITTI_CLASSES = ['Car', 'Cyclist', 'Misc', 'Pedestrian', 'Person_sitting', 'Tram', 'Truck', 'Van']
        IDX_TO_CLASS = {i: n for i, n in enumerate(KITTI_CLASSES)}
        num_classes = 8
        # Fill AVG_HEIGHTS with generic defaults if missing
        AVG_HEIGHTS = {'Car': 1.53, 'Pedestrian': 1.76, 'Cyclist': 1.74} 
        CLASS_COLORS = generate_colors(KITTI_CLASSES)

    # ---------------------------------------------------------
    # 2. INITIALIZE MODEL
    # ---------------------------------------------------------
    # Initialize with same params as training
    model = HybridDistanceModel(
        num_classes=num_classes, 
        backbone=backbone_type,
        hidden_dim=512, 
        dropout=0.5
    ) 
    
    # ---------------------------------------------------------
    # 3. PREPARE WEIGHTS 
    # ---------------------------------------------------------
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint.copy()
        # Remove metadata keys that would confuse load_state_dict
        keys_to_remove = ['class_to_idx', 'avg_heights', 'config', 'epoch', 'best_mae']
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]

    # Load Weights
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed. Retrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
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

    # 2. Object Detection (Region Proposals)
    with torch.no_grad():
        detections = detector(img_tensor)[0]

    # 3. Filter Results
    threshold = 0.5
    keep_indices = detections['scores'] > threshold
    boxes = detections['boxes'][keep_indices].cpu().numpy()
    labels = detections['labels'][keep_indices].cpu().numpy()
    
    h_img, w_img, _ = frame.shape
    
    # --- BATCH PREPARATION ---
    valid_detections = [] # Metadata for drawing
    crop_tensors = []     # For batch inference
    box_tensors = []      # For batch inference
    
    for i, (box, label_idx) in enumerate(zip(boxes, labels)):
        # Filter: Only process if it is a road object (Car, Person, Bus, etc.)
        if int(label_idx) not in COCO_INTEREST_IDS:
            continue
            
        # Coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Safe Crop Logic
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        x1_p, y1_p = max(0, x1 - pad_w), max(0, y1 - pad_h)
        x2_p, y2_p = min(w_img, x2 + pad_w), min(h_img, y2 + pad_h)
        
        crop = Image.fromarray(img_rgb[y1_p:y2_p, x1_p:x2_p])
        if crop.size[0] == 0 or crop.size[1] == 0: continue
        
        # A. Preprocess Crop
        crop_t = preprocess_crop(crop) # (1, 3, 128, 128)
        crop_tensors.append(crop_t)
        
        # B. Preprocess Coords
        norm_x = x1 / w_img
        norm_y = y1 / h_img
        norm_w = (x2 - x1) / w_img
        norm_h = (y2 - y1) / h_img
        box_t = torch.tensor([norm_x, norm_y, norm_w, norm_h], dtype=torch.float32).unsqueeze(0)
        box_tensors.append(box_t)
        
        # Store box for drawing later
        valid_detections.append({
            'box': (x1, y1, x2, y2),
            'h_pixel': max(1, y2 - y1)
        })

    # If no objects found, return early
    if not valid_detections:
        return frame

    # --- BATCH INFERENCE (1 Run) ---
    batch_crops = torch.cat(crop_tensors).to(device)
    batch_boxes = torch.cat(box_tensors).to(device)

    with torch.no_grad():
        # Run your Custom Model
        cls_logits, delta_z = dist_model(batch_crops, batch_boxes)
        
        # 1. Get Probabilities & Predictions
        probs = F.softmax(cls_logits, dim=1)
        pred_indices = torch.argmax(probs, dim=1).cpu().numpy()
        delta_z = delta_z.cpu().numpy().flatten()

    # --- DRAWING LOOP ---
    for i, det in enumerate(valid_detections):
        # 1. Get Predicted Class from Custom Model
        pred_idx = pred_indices[i]
        class_name = IDX_TO_CLASS.get(pred_idx, 'Unknown')
        
        # 2. Get True Average Height for that class
        avg_h = AVG_HEIGHTS.get(class_name, 1.5)
        
        # 3. Calculate Distance
        d_z = delta_z[i]
        Z_pin = (focal_length * avg_h) / det['h_pixel']
        Z_final = Z_pin + d_z
        
        # 4. Draw
        x1, y1, x2, y2 = det['box']
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        label_text = f"{class_name} {Z_final:.1f}m"
        
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
    parser.add_argument('--checkpoint', type=str, default='best_model_supermodel.pth', help="Path to your .pth file")
    parser.add_argument('--backbone', type=str, default='resnet18', help="Backbone type (resnet18/custom)")
    # Use a lower focal length for webcams (usually ~600-800 pixels width)
    parser.add_argument('--focal', type=float, default=800.0, help="Approx focal length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Load Models
    detector = get_object_detector(device)
    # This will load weights + metadata (heights/classes)
    dist_model = load_custom_model(args.checkpoint, device, args.backbone)

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
        
        # Optional: Resize for speed/consistency
        # frame = cv2.resize(frame, (1280, 720)) 
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

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