import torch
import os
import cv2
import numpy as np
import argparse
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torch.nn.functional as F

# Import your custom model definition
from model import HybridDistanceModel 

# --------------------------------------------------------------------------
# CONSTANTS & CONFIG
# --------------------------------------------------------------------------

# 1. MATCH THIS TO YOUR TRAINING DATASET CLASS ORDER
# This must match exactly how KittiObjectDataset assigns indices.
# Hard coded for now
#  {'Car': 0, 'Cyclist': 1, 'Misc': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Tram': 5, 'Truck': 6, 'Van': 7}
KITTI_CLASSES = [
    'Car', 'Cyclist', 'Misc', 'Pedestrian', 
    'Person_sitting', 'Tram', 'Truck', 'Van'
]

# index mapping
IDX_TO_CLASS = {i: name for i, name in enumerate(KITTI_CLASSES)}

# Averages from the real dataset, I had to hard code them because I forgot to save them when training the model.
# {'Car': 1.5219423839917103, 'Misc': 1.8312886597938152, 'Pedestrian': 1.762166064981948, 'Truck': 3.2293119266055053, 'Van': 2.200273037542662, 'Cyclist': 1.7258785942492014, 'Tram': 3.539541284403669, 'Person_sitting': 1.2773584905660378

# --------------------------------------------------------------------------
# CONSTANTS 
# --------------------------------------------------------------------------
# We still need the COCO mapping because the Region Proposer (R-CNN) 
# is external and pre-trained on COCO.
COCO_INTEREST_IDS = [1, 2, 3, 4, 6, 8] # Person, Bike, Car, Motor, Bus, Truck

# Placeholder globals (will be populated by load_custom_model)
IDX_TO_CLASS = {}
AVG_HEIGHTS = {}
CLASS_COLORS = {} # We can generate these dynamically or keep hardcoded defaults

def generate_colors(classes):
    """
    Generates random distinct colors for the classes loaded from the model.
    """
    colors = {}
    for cls_name in classes:
        # Generate a random color or map specific ones if you prefer
        # Simple hash-based color generation for consistency
        np.random.seed(sum(map(ord, cls_name)))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        colors[cls_name] = color
    return colors
def get_region_proposer(device):
    """
    Loads standard Faster R-CNN to find boxes only.
    """
    print("Loading Region Proposer (Faster R-CNN)...")
    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model

def load_custom_model(checkpoint_path, device, backbone_type='resnet18'):
    """
    Loads your trained HybridDistanceModel.
    Handles both standard checkpoints and 'patched' checkpoints with metadata.
    """
    print(f"Loading Custom Model from {checkpoint_path}...")
    
    # Initialize with same params as training
    model = HybridDistanceModel(
        num_classes=len(KITTI_CLASSES), 
        backbone=backbone_type,
        hidden_dim=512, 
        dropout=0.5
    ) 
    
    # Load the file
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ---------------------------------------------------------
    # 1. EXTRACT METADATA
    # ---------------------------------------------------------
    if 'class_to_idx' in checkpoint and 'avg_heights' in checkpoint:
        print("Metadata found in checkpoint. Updating globals...")
        # Update Global Variables
        global IDX_TO_CLASS, AVG_HEIGHTS
        IDX_TO_CLASS = {v: k for k, v in checkpoint['class_to_idx'].items()}
        AVG_HEIGHTS = checkpoint['avg_heights']
    
    # ---------------------------------------------------------
    # 2. PREPARE WEIGHTS 
    # ---------------------------------------------------------
    # A. If the weights are nested under 'state_dict' (Best Practice)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        
    # B. If the file is a flat mix of weights + metadata (Your current situation)
    else:
        state_dict = checkpoint.copy() # Make a copy so we don't break the original
        
        # REMOVE METADATA KEYS before loading
        # These keys confuse model.load_state_dict()
        keys_to_remove = ['class_to_idx', 'avg_heights', 'config', 'epoch', 'best_mae']
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]

    # 3. Load the Cleaned Weights
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed ({e}). Trying strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.eval()
    return model

def preprocess_crop(crop_img):
    """
    Standard ImageNet normalization (matches ResNet backbone).
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(crop_img).unsqueeze(0) 

def run_inference(image_path, proposer, my_model, device, focal_length):
    # 1. Load Image
    original_img_cv2 = cv2.imread(image_path)
    img_h, img_w, _ = original_img_cv2.shape
    img_rgb = cv2.cvtColor(original_img_cv2, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb).to(device).unsqueeze(0)

    # 2. Get Potential Boxes (Region Proposal)
    with torch.no_grad():
        proposals = proposer(img_tensor)[0]

    # Filter proposals (Confidence > 0.5 and is a Road Object)
    keep_indices = []
    for i, (score, label) in enumerate(zip(proposals['scores'], proposals['labels'])):
        if score > 0.5 and label.item() in COCO_INTEREST_IDS:
            keep_indices.append(i)
            
    boxes = proposals['boxes'][keep_indices].cpu().numpy()
    
    print(f"Region Proposer found {len(boxes)} regions of interest.")
    
    results = []

# 3. Run Model on every box
    print(f"{'Class':<12} | {'Pinhole (m)':<12} | {'Correction (m)':<15} | {'Final Z (m)':<12}")
    print("-" * 60)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # A. Crop
        crop = Image.fromarray(img_rgb[y1:y2, x1:x2])
        if crop.size[0] == 0 or crop.size[1] == 0: continue
        crop_tensor = preprocess_crop(crop).to(device)

        # B. Geometry
        norm_x = x1 / img_w
        norm_y = y1 / img_h
        norm_w = (x2 - x1) / img_w
        norm_h = (y2 - y1) / img_h
        box_tensor = torch.tensor([[norm_x, norm_y, norm_w, norm_h]], dtype=torch.float32).to(device)

        # C. INFERENCE
        with torch.no_grad():
            cls_logits, delta_z = my_model(crop_tensor, box_tensor)
            
            # --- CLASSIFICATION HEAD ---
            probs = F.softmax(cls_logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class_name = IDX_TO_CLASS[pred_idx]
            
            # --- REGRESSION HEAD ---
            # 1. Get Avg Height based on predicted class
            avg_h = AVG_HEIGHTS.get(pred_class_name, 1.5)
            
            # 2. Calculate Baseline (Pinhole)
            h_pixel = max(1, y2 - y1)
            Z_pin = (focal_length * avg_h) / h_pixel
            
            # 3. Add Residual Correction
            correction = delta_z.item()
            Z_final = Z_pin + correction

            # --- PRINT DEBUG INFO ---
            print(f"{pred_class_name:<12} | {Z_pin:<12.2f} | {correction:<15.2f} | {Z_final:<12.2f}")

        # Save result
        results.append({
            'box': [x1, y1, x2, y2],
            'class': pred_class_name,
            'z': Z_final,
            'conf': probs[0][pred_idx].item()
        })

    # 4. Draw Results
    for res in results:
        x1, y1, x2, y2 = res['box']
        label = res['class']
        dist = res['z']
        
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        # Draw Box
        cv2.rectangle(original_img_cv2, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label (Class + Distance)
        text = f"{label}: {dist:.2f}m"
        cv2.putText(original_img_cv2, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return original_img_cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Input image")
    parser.add_argument('--checkpoint', type=str, default='best_model_supermodel.pth')
    # Use 'resnet18' or 'custom' depending on what you trained with
    parser.add_argument('--backbone', type=str, default='resnet18') 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    region_proposer = get_region_proposer(device)
    my_model = load_custom_model(args.checkpoint, device, args.backbone)
    base_name = os.path.basename(args.image)

    file_id = os.path.splitext(base_name)[0]


    if not os.path.exists("image_predictions"):

        os.makedirs("image_predictions")
    # Run
    # Focal length approx 721 for KITTI
    result_img = run_inference(args.image, region_proposer, my_model, device, focal_length=721.5)
    
    cv2.imwrite(f"image_predictions/result_{file_id}_post.jpg", result_img)
    print(f"Saved to image_predictions/result_{file_id}_post.jpg")