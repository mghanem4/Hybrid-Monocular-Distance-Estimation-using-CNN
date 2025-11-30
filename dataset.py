import os
import pickle
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Callable

# --------------------------------------------------------------------------
# DATA MODULES
# --------------------------------------------------------------------------
from data_modules import (
    make_train_val_split_from_ids,
    load_or_build_split_ids,
    get_valid_image_ids,
    load_or_build,
    read_calib_files,
    read_label_files
)


# --------------------------------------------------------------------------
# CONSTANTS & PATHS
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
LABELS_DIR = (HERE / "training" / "label_2").as_posix()
CALIB_DIR  = (HERE / "data_object_calib" / "training" / "calib").as_posix()
IMAGE_DIR  = (HERE / "data_object_image_2" /"training" / "image_2").as_posix()
CACHE_DIR  = (HERE / "cache").as_posix() # Directory for pickles



# --------------------------------------------------------------------------
# DATASET CLASS
# --------------------------------------------------------------------------

class KittiObjectDataset(Dataset):
    def __init__(self,mode= 'train',image_dir=IMAGE_DIR,transform= False):
        """
        Args:
            split_ids (list): List of image IDs to include.
            label_data (dict): Dictionary {img_id: list_of_parsed_lines}
            calib_data (dict): Dictionary {img_id: P2_matrix}
            image_dir (str): Path to image directory.
            transform (callable): Optional transform.
        """
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)

        self.image_dir = image_dir
        
        self.calib_data = load_or_build(CALIB_DIR, read_calib_files, f"{CACHE_DIR}/calib_data.pkl")
        self.label_data = load_or_build(LABELS_DIR, read_label_files, f"{CACHE_DIR}/label_data.pkl")

        valid_ids = get_valid_image_ids(self.label_data, self.calib_data, IMAGE_DIR)
        split_data = load_or_build_split_ids(valid_ids, f"{CACHE_DIR}/split_ids.pkl")

        # select Splits
        if mode == 'test':
            self.split_ids = split_data['val_ids']        
        else: 
            self.split_ids = split_data['train_ids']

        self.objects = self._flatten_objects()
        self.transform= transform
        #Compute Averages AND Build Class Mapping
        # This function now does two jobs: calculates height stats and discovers classes
        self.avg_heights, self.class_to_idx = self.get_class_stats()
        
        print(f"[{mode}] Classes Found: {self.class_to_idx}")
        print(f"[{mode}] Average Heights: {self.avg_heights}")

    def _flatten_objects(self):
        """
        Iterates over split_ids and creates a flat list of objects.
        This intermediate format is necessary because the DataLoader index 'i'
        must map to a specific object, but our raw data is organized by image.
        """
        flat_objects = []

        for img_id in self.split_ids:
            # Skip if data missing
            if img_id not in self.label_data or img_id not in self.calib_data:
                continue

            img_path = Path(self.image_dir) / f"{img_id}.png"
            P2 = self.calib_data[img_id]
            focal_length = P2[0, 0]

            # Retrieve pre-parsed objects
            image_objects = self.label_data[img_id]

            for obj in image_objects:
                class_name = obj['class_name']
                if class_name in "DontCare":
                    continue

                # Construct the final training object
                # We inject the dynamic/calculated fields (focal_length, path, ids) here.
                # cls_idx = self.class_to_idx[class_name]

                flat_objects.append({
                    'img_id': img_id,
                    'img_path': str(img_path),
                    'box': obj['box'],
                    'class_name': class_name,
                    # 'class_idx':cls_idx,
                    'z_gt': obj['z_gt'],
                    'h3d': obj['h3d'],
                    'focal_length': focal_length
                })

        print(f"Dataset Split Loaded: {len(flat_objects)} objects from {len(self.split_ids)} images.")
        return flat_objects

    def get_class_stats(self):
        """
        Calculates average H3D per class and creates the class_to_idx mapping.
        """
        class_stats = {}
        unique_classes = set()

        for obj in self.objects:
            cls = obj['class_name']
            val = obj['h3d']
            unique_classes.add(cls)

            if cls not in class_stats:
                class_stats[cls] = {'total_h3d': 0.0, 'count': 0}
            
            class_stats[cls]['total_h3d'] += val
            class_stats[cls]['count'] += 1

        # 1. Calculate Averages
        averages = {}
        for cls, stats in class_stats.items():
            if stats['count'] > 0:
                averages[cls] = stats['total_h3d'] / stats['count']
            else:
                averages[cls] = 0.0
        
        # 2. Create Integer Mapping (Sorted for consistency)
        # e.g., {'Car': 0, 'Cyclist': 1, 'Pedestrian': 2}
        sorted_classes = sorted(list(unique_classes))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}

        return averages, class_to_idx

    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        obj = self.objects[idx]
        img_path = obj['img_path']
        img = cv2.imread(obj['img_path'])
        # 1. Load Image
        if img is None or img.size == 0:
            print(f"\n[WARNING] Corrupted/Missing image at: {img_path}")
            print(f"          Returning dummy black image to prevent crash.")
            
            # Create a black dummy image of standard KITTI size (Height, Width, Channels)
            img = np.zeros((375, 1242, 3), dtype=np.uint8)
        else:
            # Only convert color if the image is valid
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except cv2.error:
                # Double safety net
                print(f"\n[ERROR] cv2.cvtColor failed on: {img_path}")
                img = np.zeros((375, 1242, 3), dtype=np.uint8)


        x1, y1, x2, y2 = map(int, obj['box'])
        
        # add Padding 
        # Adds 10% context around the box
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        h_img, w_img = img.shape[:2]
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w_img, x2 + pad_w)
        y2 = min(h_img, y2 + pad_h)

        crop = img[y1:y2, x1:x2]
        
        # Handle edge case where crop is empty
        if crop.size == 0:
            crop = np.zeros((128, 128, 3), dtype=np.uint8)

        # 3. Resize
        crop = cv2.resize(crop, (128, 128))

        # Change from (128, 128, 3) -> (3, 128, 128)
        crop = np.transpose(crop, (2, 0, 1))

        # 4. Get the Average Height for this specific class 
        cls_name = obj['class_name']
        avg_h_for_class = self.avg_heights.get(cls_name, 1.5)

        # Calculate pixel height (used for the pinhole model)
        h_pixel = max(1, y2 - y1)

        if self.transform:
            crop = self.transform(crop)
        # Look up the integer index using the dictionary built in get_class_stats
        cls_idx = self.class_to_idx[cls_name]
        return {
            'image': crop,
            'bbox_height': np.float32(h_pixel),
            'class_idx': np.int64(cls_idx),
            'class_name': cls_name,
            'avg_height': np.float32(avg_h_for_class),
            'z_gt': np.float32(obj['z_gt']),
            'focal_length': np.float32(obj['focal_length'])
        }