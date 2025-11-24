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
# CONSTANTS & PATHS
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
LABELS_DIR = (HERE / "training" / "label_2").as_posix()
CALIB_DIR  = (HERE / "data_object_calib" / "training" / "calib").as_posix()
IMAGE_DIR  = (HERE / "data_object_image_2" /"training" / "image_2").as_posix()
CACHE_DIR  = (HERE / "cache").as_posix() # Directory for pickles

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# HELPER FUNCTIONS (Moved from config/read_info.py & data_prep.py)
# --------------------------------------------------------------------------

def save_pickle(data, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_P2_matrix(P2: np.ndarray) -> np.ndarray:
    """Convert KITTI P2 to a (3,4) camera matrix if needed."""
    P2 = np.asarray(P2)
    if P2.shape == (12,) or P2.shape == (1, 12):
        return P2.reshape(3, 4)
    if P2.shape == (3, 4):
        return P2
    raise ValueError(f"Unexpected P2 shape {P2.shape}")

def read_label_files(directory_path: str, verbose: bool = False) -> Tuple[int, Dict[str, List[Dict]]]:
    """
    Reads all label files and immediately parses them into structured dictionaries.
    Returns: (total_objects, {img_id: [ {class_name, box, h3d, z_gt}, ... ] })
    """
    objects: Dict[str, List[Dict]] = {}
    total_obj = 0
    for p in Path(directory_path).glob("*.txt"):
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except OSError as e:
            print(f"Error reading file {p.name}: {e}")
            continue

        file_objs = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            cls_type = parts[0]
            if cls_type == "DontCare": continue

            # Parse floats immediately
            try:
                # box: left, top, right, bottom (indices 4-7)
                box = [float(x) for x in parts[4:8]]
                h3d = float(parts[8])
                z_loc = float(parts[13])

                # Filter bad data
                if z_loc <= 0: continue

                file_objs.append({
                    'class_name': cls_type,
                    'box': box,
                    'h3d': h3d,
                    'z_gt': z_loc
                })
            except ValueError:
                continue

        key = p.stem
        total_obj += len(file_objs)
        objects[key] = file_objs
    return total_obj, objects

def read_calib_files(directory_path: str, verbose: bool = False) -> Tuple[int, Dict[str, np.ndarray]]:
    calib_p2: Dict[str, np.ndarray] = {}
    total_calib = 0
    for p in Path(directory_path).glob("*.txt"):
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except OSError:
            continue

        p2_rows = []
        for line in lines:
            calib = line.strip().split()
            if not calib: continue
            if calib[0].startswith("P2"):
                p2_rows.append(np.array(calib[1:], dtype=float))

        if not p2_rows: continue
        key = p.stem
        calib_p2[key] = ensure_P2_matrix(p2_rows[0])
        total_calib += 1
    return total_calib, calib_p2

def load_or_build(labels_dir: str, read_function: Callable, cache_path: str, verbose: bool = False, rebuild=False):
    if (not rebuild) and os.path.exists(cache_path):
        if verbose: print(f"Loading cached data from {cache_path}")
        return load_pickle(cache_path)['items']

    if verbose: print(f"Building data from {labels_dir}...")
    total_objects, objects = read_function(labels_dir, verbose=verbose)
    payload = {"total_items": total_objects, "items": objects}
    save_pickle(payload, cache_path)
    return payload['items']

def get_valid_image_ids(label_data: Dict, calib_data: Dict, image_dir: str) -> List[str]:
    label_ids = set(label_data.keys())
    calib_ids = set(calib_data.keys())
    image_ids = {p.stem for p in Path(image_dir).glob("*.png")}
    valid_ids = sorted(label_ids & calib_ids & image_ids)
    if not valid_ids:
        # Fallback for dry-run/dummy env if no intersection
        # (Assuming the user might have some partial data)
        if len(label_ids) > 0:
            print(f"Warning: No full intersection. Found {len(label_ids)} labels, {len(calib_ids)} calibs, {len(image_ids)} images.")
        else:
            raise RuntimeError("No valid image IDs found (intersection of labels, calib, images is empty).")
    return valid_ids

def make_train_val_split_from_ids(valid_ids: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    ids = list(valid_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n_total = len(ids)
    n_val = max(1, int(round(val_ratio * n_total)))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids

def load_or_build_split_ids(valid_ids: List[str], cache_path: str, val_ratio: float = 0.2, seed: int = 42):
    if os.path.exists(cache_path):
        return load_pickle(cache_path)
    train_ids, val_ids = make_train_val_split_from_ids(valid_ids, val_ratio=val_ratio, seed=seed)
    payload = {"train_ids": train_ids, "val_ids": val_ids}
    save_pickle(payload, cache_path)
    return payload

# --------------------------------------------------------------------------
# DATASET CLASS
# --------------------------------------------------------------------------

class KittiObjectDataset(Dataset):
    def __init__(self, split_ids, label_data: dict, calib_data: dict, image_dir=IMAGE_DIR, transform=None):
        """
        Args:
            split_ids (list): List of image IDs to include.
            label_data (dict): Dictionary {img_id: list_of_parsed_lines}
            calib_data (dict): Dictionary {img_id: P2_matrix}
            image_dir (str): Path to image directory.
            transform (callable): Optional transform.
        """
        self.split_ids = split_ids
        self.label_data = label_data
        self.calib_data = calib_data
        self.image_dir = image_dir
        self.transform = transform

        self.class_map = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

        # Flatten the data structure: List of all valid objects in the split
        self.objects = self._flatten_objects()

        # Compute dynamic average heights
        self.avg_heights = self._compute_avg_heights()

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
                if cls_type not in self.class_map:
                    continue

                # Construct the final training object
                # We inject the dynamic/calculated fields (focal_length, path, ids) here.
                flat_objects.append({
                    'img_id': img_id,
                    'img_path': str(img_path),
                    'box': obj['box'],
                    'class_name': cls_type,
                    'class_id': self.class_map[cls_type],
                    'z_gt': obj['z_gt'],
                    'h3d': obj['h3d'],
                    'focal_length': focal_length
                })

        print(f"Dataset Split Loaded: {len(flat_objects)} objects from {len(self.split_ids)} images.")
        return flat_objects

    def _compute_avg_heights(self):
        heights = {0: [], 1: [], 2: []}
        for obj in self.objects:
            heights[obj['class_id']].append(obj['h3d'])

        avg_heights = {}
        defaults = {0: 1.53, 1: 1.76, 2: 1.73}

        for cls_name, cls_id in self.class_map.items():
            if heights[cls_id]:
                avg_heights[cls_name] = np.mean(heights[cls_id])
            else:
                avg_heights[cls_name] = defaults[cls_id]

        return avg_heights

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
            'class_id': obj['class_id'],
            'z_gt': np.float32(obj['z_gt']),
            'focal_length': obj['focal_length']
        }
