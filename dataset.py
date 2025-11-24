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
    def __init__(self,mode= 'train',image_dir=IMAGE_DIR):
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
        
        self.calib_data = load_or_build(CALIB_DIR, read_calib_files, f"{CACHE_DIR}/calib_data.pkl")
        self.label_data = load_or_build(LABELS_DIR, read_label_files, f"{CACHE_DIR}/label_data.pkl")

        valid_ids = get_valid_image_ids(self.label_data, self.calib_data, IMAGE_DIR)
        split_data = load_or_build_split_ids(valid_ids, f"{CACHE_DIR}/split_ids.pkl")

        if mode == 'test':
            self.split_ids = split_data['val_ids']        
        if mode == 'train':
            self.split_ids = split_data['train_ids']
        
        self.image_dir = image_dir
        # Flatten the data structure: List of all valid objects in the split
        self.objects = self._flatten_objects()

        # Compute dynamic average heights
        self.avg_heights = self.get_class_averages()

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
                cls_type = obj['class_name']
                if cls_type in "DontCare":
                    continue

                # Construct the final training object
                # We inject the dynamic/calculated fields (focal_length, path, ids) here.
                flat_objects.append({
                    'img_id': img_id,
                    'img_path': str(img_path),
                    'box': obj['box'],
                    'class_name': cls_type,
                    'z_gt': obj['z_gt'],
                    'h3d': obj['h3d'],
                    'focal_length': focal_length
                })

        print(f"Dataset Split Loaded: {len(flat_objects)} objects from {len(self.split_ids)} images.")
        return flat_objects

    def get_class_averages(self):
        # Initialize a dictionary to store sum and count
        class_stats = {}
        for obj in self.objects:
            cls = obj['class_name']
            val = obj['h3d']
            # Initialize class in stats if not present
            if cls not in class_stats:
                class_stats[cls] = {'total_h3d': 0.0, 'count': 0}
            # Accumulate data
            class_stats[cls]['total_h3d'] += val
            class_stats[cls]['count'] += 1
        #calculate averages
        averages = {}
        for cls, stats in class_stats.items():
            if stats['count'] > 0:
                averages[cls] = stats['total_h3d'] / stats['count']
            else:
                averages[cls] = 0
        return averages

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj = self.objects[idx]

        # Load Image
        img = cv2.imread(obj['img_path'])
        if img is None:
            # Fallback for dummy data tests
            img = np.zeros((375, 1242, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x1, y1, x2, y2 = map(int, obj['box'])
        h_img, w_img = img.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((128, 128, 3), dtype=np.uint8)

        h_pixel = max(1, y2 - y1)

        if self.transform:
            crop = self.transform(crop)

        return {
            'image': crop,
            'bbox_height': h_pixel,
            'class_name': obj['class_name'],
            'z_gt': np.float32(obj['z_gt']),
            'focal_length': obj['focal_length']
        }
