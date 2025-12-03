import torch
import os
import cv2
import numpy as np
import argparse
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torch.nn.functional as F

from model import HybridDistanceModel

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

from dataset import KittiObjectDataset

# Initialize in 'train' mode to get the training statistics
ds = KittiObjectDataset(mode='train')

print(f"KITTI_CLASS_TO_IDX = {ds.class_to_idx}")
print(f"AVG_HEIGHTS = {ds.avg_heights}")
print("------------------------------------")

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
    'Car': 1.53,          # Now correctly includes Car!
    'Van': 2.21, 
    'Person_sitting': 1.27, 
    'Cyclist': 1.74, 
    'Truck': 3.25, 
    'Misc': 1.92, 
    'Tram': 3.53
}
'''
I will be using the COCO pre trained RCNN model to detect objcets to avoid make the process faster, and to be accurate. I will map the COCO classes to my KITTI classes 
My classes:

'''
# Update COCO Map to point to these keys
COCO_TO_KITTI = {
    1: 'Pedestrian',
    2: 'Cyclist',
    3: 'Car',      # maps to 'Car' 
    4: 'Cyclist',
    6: 'Truck',    # Bus -> Truck
    8: 'Truck'     # Truck -> Truck
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

    # 3. Process each detection
    results_img = original_img_cv2.copy()
    h_img, w_img, _ = original_img_cv2.shape # Get dimensions for normalization
    
    # Filter by confidence
    threshold = 0.5
    keep_indices = detections['scores'] > threshold
    
    boxes = detections['boxes'][keep_indices].cpu().numpy()
    labels = detections['labels'][keep_indices].cpu().numpy()
    
    print(f"Found {len(boxes)} objects.")

    # Enumerate to get an ID number
    for i, (box, label_idx) in enumerate(zip(boxes, labels)):
        if label_idx not in COCO_TO_KITTI:
            continue 
            
        # Use safe casting to int
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
        # Handle potential empty crops at edge of screen
        if crop.size[0] == 0 or crop.size[1] == 0: continue
        
        crop_tensor = preprocess_crop(crop).to(device)
        h_pixel = max(1, y2 - y1) 

        # --- PREPROCESSING (GEOMETRIC COORDS) ---
        # 1. Calculate Normalized Coordinates (0 to 1)
        # Note: We use the raw box (x1, y1), not the padded one, to match training
        norm_x = x1 / w_img
        norm_y = y1 / h_img
        norm_w = (x2 - x1) / w_img
        norm_h = (y2 - y1) / h_img
        
        # 2. Create Tensor (Batch=1, Features=4)
        box_tensor = torch.tensor([norm_x, norm_y, norm_w, norm_h], dtype=torch.float32).unsqueeze(0).to(device)

        # --- STAGE 2: PREDICT DISTANCE ---
        with torch.no_grad():
            # Pass BOTH image and coords
            cls_logits, delta_z = dist_model(crop_tensor, box_tensor)
            
            Z_pin = (focal_length * avg_h) / h_pixel
            Z_final = Z_pin + delta_z.item()
        
        print(f"ID #{i+1} [{kitti_class}]: {Z_final:.2f}m (Pinhole: {Z_pin:.2f}, Delta: {delta_z.item():.2f})")

        # --- IMPROVED VISUALIZATION (ALTERNATING) ---
        # 1. Get color and create label text with ID
        color = CLASS_COLORS.get(kitti_class, (255, 255, 255))
        label_text = f"#{i+1} {kitti_class} {Z_final:.1f}m"

        # 2. Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # 3. Determine Staggered Position
        label_x = x1
        
        # Logic: Even ID -> Try Top | Odd ID -> Try Bottom
        if i % 2 == 0:
            # Try placing ABOVE the box
            label_y = y1 - 5
        else:
            # Try placing BELOW the box
            label_y = y2 + text_height + 5

        # 4. Boundary Safety Checks (Clamp to Image Edges)
        if label_y - text_height < 0:
            label_y = y1 + text_height + 10
        elif label_y + baseline > h_img:
            label_y = y2 - 5

        # 5. Draw filled rectangle behind text for readability
        cv2.rectangle(results_img, 
                      (label_x, label_y - text_height - baseline), 
                      (label_x + text_width, label_y + baseline), 
                      color, -1) # -1 thickness means filled

        # 6. Draw the main bounding box
        cv2.rectangle(results_img, (x1, y1), (x2, y2), color, 2)
        
        # 7. Draw white text on top
        cv2.putText(results_img, label_text, (label_x, label_y), 
                    font, font_scale, (255, 255, 255), thickness)

    return results_img
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
        # 1. Get the base name (000053.png)
    base_name = os.path.basename(args.image)

    # 2. Split the base name from the extension to get '000053'
    file_id = os.path.splitext(base_name)[0]
    # Show/Save
    if not os.path.exists("image_predictions"):
        os.makedirs("image_predictions")
    save_path = f"image_predictions/prediction_result{file_id}_adjusted.jpg"
    cv2.imwrite(save_path, result)
    print(f"Result saved to {save_path}")
    
    # Optional: Display if running locally with GUI
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()