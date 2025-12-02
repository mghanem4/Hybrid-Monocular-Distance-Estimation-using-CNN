import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import random
from loguru import logger

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
            logger.error(f"Error reading file {p.name}: {e}")
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
        if verbose: logger.info(f"Loading cached data from {cache_path}")
        return load_pickle(cache_path)['items']

    if verbose: logger.info(f"Building data from {labels_dir}...")
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
            logger.error(f"Warning: No full intersection. Found {len(label_ids)} labels, {len(calib_ids)} calibs, {len(image_ids)} images.")
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