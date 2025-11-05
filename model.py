from config.read_info import fetchCalibrationData, fetchLabelData, load_pickle, save_pickle
from pathlib import Path
import os, pickle, random
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np

HERE = Path(__file__).resolve().parent
# Adjust this to your repo layout
LABELS_DIR = (HERE / "training" / "label_2").as_posix()
CALIB_DIR  = (HERE / "data_object_calib" / "training" / "calib").as_posix()
CONFIG_PATH= (HERE / "config" ).as_posix()



def make_train_val_split(calib_data: Dict[str, np.ndarray],label_data: Dict[str, List[List[str]]],*,val_ratio: float = 0.2,seed: int = 42,keep_sorted: bool = True) -> Tuple[List[str], List[str]]:
    """
    Split on the intersection of keys to keep label/calib aligned.
    Returns (train_keys, val_keys).
    """
    keys = list(set(calib_data.keys()).intersection(label_data.keys()))
    if not keys:
        raise RuntimeError("No overlapping filenames between calib and label dicts.")
    if keep_sorted:
        keys.sort()
    rnd = random.Random(seed)
    rnd.shuffle(keys)  # reproducible shuffle
    n_total = len(keys)
    n_val = max(1, int(round(val_ratio * n_total)))
    val_keys = keys[:n_val]
    train_keys = keys[n_val:]
    return train_keys, val_keys

def load_or_build_split(calib_data: Dict[str, np.ndarray],label_data: Dict[str, List[List[str]]],cache_path: str = "split.pkl",*,val_ratio: float = 0.2,seed: int = 42
):
    if os.path.exists(cache_path):
        return load_pickle(cache_path)
    train_keys, val_keys = make_train_val_split(
        calib_data, label_data, val_ratio=val_ratio, seed=seed
    )
    payload = {"train_keys": train_keys, "val_keys": val_keys}
    save_pickle(payload, cache_path)
    return payload



if __name__ == "__main__":
    print(LABELS_DIR)
    print(CALIB_DIR)
    print(CONFIG_PATH)
    calibrationData:dict= fetchCalibrationData(CALIB_DIR,f'{CONFIG_PATH}/calib_data.pkl')
    first_key, first_value = next(iter(calibrationData.items()))
    print(f"Key: {first_key}, Value: {first_value}")
    labelData= fetchLabelData(LABELS_DIR,f'{CONFIG_PATH}/label_data.pkl')
    first_key, first_value = next(iter(labelData.items()))
    print(f"Key: {first_key}, Value: {first_value}")