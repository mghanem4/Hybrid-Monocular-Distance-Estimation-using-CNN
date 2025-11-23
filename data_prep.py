from config.read_info import fetchCalibrationData, fetchLabelData, load_pickle, save_pickle
from pathlib import Path
import os, pickle, random
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import pprint


HERE = Path(__file__).resolve().parent
# Adjust this to your repo layout
LABELS_DIR = (HERE / "training" / "label_2/").as_posix()
CALIB_DIR  = (HERE / "data_object_calib" / "training" / "calib/").as_posix()
IMAGE_DIR  = (HERE / "data_object_image_2" /"training" / "image_2/").as_posix()   # <- add this
CONFIG_PATH= (HERE / "config" ).as_posix()

def get_valid_image_ids(label_data: Dict[str, List[List[str]]], calib_data: Dict[str, np.ndarray],image_dir: str,) -> List[str]:
    """
    Return sorted list of image base IDs (e.g. '000001') that:
    - have a label file
    - have a calib file
    - have a corresponding image file in image_dir
    Otherwise we would end up with more images than labels + calib, which isn't right.
    """
    label_ids = set(label_data.keys())
    calib_ids = set(calib_data.keys())
    image_ids = {p.stem for p in Path(image_dir).glob("*.png")}  # KITTI uses .png
    # Get intersection (Images with calib + labels)
    valid_ids = sorted(label_ids & calib_ids & image_ids)
    if not valid_ids:
        raise RuntimeError("No valid image IDs found (intersection of labels, calib, images is empty).")
    return valid_ids

def make_train_val_split_from_ids(
    valid_ids: List[str],
    *,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Split a list of image base IDs (e.g. ['000001', '000002', ...]) 
    into train and val sets.
    Returns (train_ids, val_ids).
    """
    ids = list(valid_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)

    n_total = len(ids)
    n_val = max(1, int(round(val_ratio * n_total)))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids

def load_or_build_split_ids(
    valid_ids: List[str],
    cache_path: str,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Cache and load the split of image IDs.
    """
    if os.path.exists(cache_path):
        return load_pickle(cache_path)

    train_ids, val_ids = make_train_val_split_from_ids(
        valid_ids, val_ratio=val_ratio, seed=seed
    )
    payload = {"train_ids": train_ids, "val_ids": val_ids}
    save_pickle(payload, cache_path)
    return payload


if __name__ == "__main__":
    print("LABELS_DIR:", LABELS_DIR)
    print("CALIB_DIR :", CALIB_DIR)
    print("IMAGE_DIR :", IMAGE_DIR)
    print("CONFIG_PATH:", CONFIG_PATH)

    calib_data: dict = fetchCalibrationData(CALIB_DIR, f"{CONFIG_PATH}/calib_data.pkl")
    label_data: dict = fetchLabelData(LABELS_DIR, f"{CONFIG_PATH}/label_data.pkl")

    # 1) Compute valid image IDs
    valid_ids = get_valid_image_ids(label_data, calib_data, IMAGE_DIR)
    print(f"Total valid image IDs (labels ∩ calib ∩ images): {len(valid_ids)}")
    print("First few IDs:", valid_ids[10:])

    # 2) Build or load train/val split
    split_cache = f"{CONFIG_PATH}/split_ids.pkl"
    split = load_or_build_split_ids(valid_ids, split_cache, val_ratio=0.2, seed=42)
    print(f"Train IDs: {len(split['train_ids'])}, Val IDs: {len(split['val_ids'])}")
