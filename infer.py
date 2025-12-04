import torch
import os
import cv2
import numpy as np
import argparse
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torch.nn.functional as F

# Assumed local imports (based on your snippet)
# If these files are missing in this environment, this script is for your local use.
try:
    from model import HybridDistanceModel
    from dataset import KittiObjectDataset
except ImportError:
    # Dummy mocks for syntax checking if files aren't present
    print("Warning: local modules 'model' or 'dataset' not found. Using placeholders.")
    HybridDistanceModel = None
    KittiObjectDataset = lambda mode: None

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

# Updated to match your training log (8 classes)
KITTI_CLASS_TO_IDX = {
    'Car': 0, 
    'Cyclist': 1, 
    'Misc': 2, 
    'Pedestrian': 3, 
    'Person_sitting': 4, 
    'Tram': 5, 
    'Truck': 6, 
    'Van': 7
}

AVG_HEIGHTS = {
    'Pedestrian': 1.76, 
    'Car': 1.53,
    'Van': 2.21, 
    'Person_sitting': 1.27, 
    'Cyclist': 1.74, 
    'Truck': 3.25, 
    'Misc': 1.92, 
    'Tram': 3.53
}

# Update COCO Map to point to these keys
COCO_TO_KITTI = {
    1: 'Pedestrian',
    2: 'Cyclist',
    3: 'Car',      # maps to 'Car' 
    4: 'Cyclist',
    6: 'Truck',    # Bus -> Truck
    8: 'Truck'     # Truck -> Truck
}

# Define distinct colors for classes (BGR format for OpenCV)
CLASS_COLORS = {
    'Car': (1, 50, 32),        # Green
    'Van': (0, 200, 200),      # Yellowish-Green
    'Truck': (0, 165, 255),    # Orange
    'Pedestrian': (0, 0, 255), # Red
    'Person_sitting': (50, 50, 200), # Dark Red
    'Cyclist': (255, 0, 0),    # Blue
    'Tram': (255, 0, 255),     # Magenta
    'Misc': (128, 128, 128)    # Gray
}

def get_object_detector(device):
    """
    Loads a pre-trained Faster R-CNN to find objects in the image.
    """
    print("Loading Object Detector (Faster R-CNN)...")
    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model

def load_distance_model(checkpoint_path, device):
    """
    Loads your custom trained HybridDistanceModel.
    """
    print(f"Loading Distance Model from {checkpoint_path}...")
    if HybridDistanceModel is None:
        raise ImportError("HybridDistanceModel class is missing.")
        
    # Initialize model structure
    model = HybridDistanceModel(num_classes=8, backbone='resnet18') 
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def preprocess_crop(crop_img):
    """
    Matches the preprocessing in dataset.py:
    Resize to 128x128 -> ToTensor -> Normalize
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(crop_img).unsqueeze(0) # Add batch dimension

def check_overlap(rect1, rect2):
    """
    Checks if two rectangles overlap.
    Rect format: (x, y, w, h)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def draw_smart_labels(image, detections):
    """
    Draws labels with collision detection. If a label overlaps, moves it and draws a leader line.
    detections: list of dicts {'box': [x1,y1,x2,y2], 'text': str, 'color': tuple}
    """
    img_h, img_w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Store occupied regions (x, y, w, h)
    occupied_rects = []
    
    # Sort detections by Y coordinate (bottom-up) or X to establish priority?
    # Usually sorting by X helps scan left-to-right.
    detections.sort(key=lambda x: x['box'][0]) 

    for det in detections:
        x1, y1, x2, y2 = det['box']
        text = det['text']
        color = det['color']
        
        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        total_h = text_h + baseline + 4 # padding
        total_w = text_w + 4

        # --- Strategy: Define Candidate Positions ---
        # 1. Top of box (Preferred)
        # 2. Bottom of box
        # 3. Far Above (with line)
        # 4. Far Below (with line)
        
        candidates = [
            {'pos': (x1, y1 - total_h), 'line': False},          # Standard Top
            {'pos': (x1, y2 + 5), 'line': False},               # Standard Bottom
            {'pos': (x1, y1 - total_h - 20), 'line': True},     # Floating Top
            {'pos': (x1, y2 + 25), 'line': True},               # Floating Bottom
            {'pos': (x1 - 20, y1 - total_h - 10), 'line': True}, # Offset Left Top
            {'pos': (x1 + 20, y1 - total_h - 10), 'line': True}  # Offset Right Top
        ]
        
        chosen_candidate = None
        
        for cand in candidates:
            cx, cy = cand['pos']
            
            # Boundary check
            if cx < 0: cx = 0
            if cx + total_w > img_w: cx = img_w - total_w
            if cy < 0: cy = 0
            if cy + total_h > img_h: cy = img_h - total_h
            
            # Collision check
            current_rect = (cx, cy, total_w, total_h)
            is_overlapping = False
            for r in occupied_rects:
                if check_overlap(current_rect, r):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                chosen_candidate = cand
                chosen_candidate['final_pos'] = (int(cx), int(cy))
                break
        
        # If all candidates fail, default to the first one (overwrite happens, but rare)
        if chosen_candidate is None:
            chosen_candidate = candidates[0]
            cx, cy = candidates[0]['pos']
            # Clamp
            cx = max(0, min(cx, img_w - total_w))
            cy = max(0, min(cy, img_h - total_h))
            chosen_candidate['final_pos'] = (int(cx), int(cy))

        # --- Draw ---
        lx, ly = chosen_candidate['final_pos']
        
        # 1. Draw Leader Line if needed
        if chosen_candidate['line']:
            # Determine closest point on box to label
            box_center_x = x1 + (x2-x1)//2
            label_center_x = lx + total_w//2
            label_bottom = ly + total_h
            label_top = ly
            
            # Simple line to top-left of box or center
            pt1 = (int(label_center_x), int(label_bottom) if ly < y1 else int(label_top))
            pt2 = (int(x1), int(y1)) # anchor to top-left of object box
            
            cv2.line(image, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(image, pt2, 3, color, -1) # Dot at the anchor

        # 2. Draw Background
        cv2.rectangle(image, (lx, ly), (lx + total_w, ly + total_h), color, -1)
        
        # 3. Draw Text
        cv2.putText(image, text, (lx + 2, ly + text_h + 2), 
                    font, font_scale, (255, 255, 255), thickness)
        
        # 4. Draw Object Box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Register space
        occupied_rects.append((lx, ly, total_w, total_h))

    return image

def run_inference(image_path, detector, dist_model, device, focal_length):
    # 1. Load Image
    original_img_cv2 = cv2.imread(image_path)
    if original_img_cv2 is None:
        raise ValueError(f"Could not open image: {image_path}")
    
    # Convert to RGB for Tensor processing
    img_rgb = cv2.cvtColor(original_img_cv2, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb).to(device).unsqueeze(0)

    # 2. Stage 1: Detect Objects
    with torch.no_grad():
        detections = detector(img_tensor)[0]

    h_img, w_img, _ = original_img_cv2.shape
    
    # Filter by confidence
    threshold = 0.5
    keep_indices = detections['scores'] > threshold
    
    boxes = detections['boxes'][keep_indices].cpu().numpy()
    labels = detections['labels'][keep_indices].cpu().numpy()
    
    print(f"Found {len(boxes)} objects.")
    
    # Collect all valid detections first (DO NOT DRAW YET)
    valid_detections = []

    for i, (box, label_idx) in enumerate(zip(boxes, labels)):
        if label_idx not in COCO_TO_KITTI:
            continue 
            
        kitti_class = COCO_TO_KITTI[int(label_idx)]
        avg_h = AVG_HEIGHTS[kitti_class]

        # Extract Box Coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # --- PREPROCESSING (CROP) ---
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        x1_p = max(0, x1 - pad_w)
        y1_p = max(0, y1 - pad_h)
        x2_p = min(w_img, x2 + pad_w)
        y2_p = min(h_img, y2 + pad_h)
        
        crop = Image.fromarray(img_rgb[y1_p:y2_p, x1_p:x2_p])
        if crop.size[0] == 0 or crop.size[1] == 0: continue
        
        crop_tensor = preprocess_crop(crop).to(device)
        h_pixel = max(1, y2 - y1) 

        # --- PREPROCESSING (GEOMETRIC COORDS) ---
        norm_x = x1 / w_img
        norm_y = y1 / h_img
        norm_w = (x2 - x1) / w_img
        norm_h = (y2 - y1) / h_img
        
        box_tensor = torch.tensor([norm_x, norm_y, norm_w, norm_h], dtype=torch.float32).unsqueeze(0).to(device)

        # --- STAGE 2: PREDICT DISTANCE ---
        with torch.no_grad():
            cls_logits, delta_z = dist_model(crop_tensor, box_tensor)
            Z_pin = (focal_length * avg_h) / h_pixel
            Z_final = Z_pin + delta_z.item()
        
        # Store for batch drawing
        valid_detections.append({
            'box': [x1, y1, x2, y2],
            'text': f"#{i+1} {kitti_class} {Z_final:.1f}m",
            'color': CLASS_COLORS.get(kitti_class, (255, 255, 255)),
            'z': Z_final
        })

    # 3. Draw using Smart Labeling
    result_img = draw_smart_labels(original_img_cv2.copy(), valid_detections)
    
    return result_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--checkpoint', type=str, default='best_model.pth', help="Path to your trained .pth file")
    parser.add_argument('--focal', type=float, default=721.5, help="Camera focal length in pixels (Default: KITTI value)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    detector = get_object_detector(device)
    dist_model = load_distance_model(args.checkpoint, device)
    
    # Run
    result = run_inference(args.image, detector, dist_model, device, args.focal)
    
    # Save
    base_name = os.path.basename(args.image)
    file_id = os.path.splitext(base_name)[0]
    
    if not os.path.exists("image_predictions"):
        os.makedirs("image_predictions")
    
    save_path = f"image_predictions/prediction_result_{file_id}_adjusted.jpg"
    cv2.imwrite(save_path, result)
    print(f"Result saved to {save_path}")