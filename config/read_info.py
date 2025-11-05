import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from typing import Callable

def save_pickle(data, path: str) -> None:
    '''
    Generic Save pickle function
    @Params: 
    - data: object file to save
    - path: path to save pickle file in
    '''
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    '''
    Generic Save pickle function
    @Params: 
    - path: path to load pickle file
    '''
    with open(path, "rb") as f:   # <-- 'rb', not 'wb'
        return pickle.load(f)


def read_label_files(directory_path: str, verbose: bool = False) -> Tuple[int, Dict[str, List[List[str]]]]:
    """
    Reads all .txt label files in a directory.
    Returns (total_objects, objects_dict) where objects_dict maps filename -> list of parsed rows.
    """
    objects: Dict[str, List[List[str]]] = {}
    total_obj = 0
    for p in Path(directory_path).glob("*.txt"):
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except OSError as e:
            print(f"Error reading file {p.name}: {e}")
            continue

        file_objs: List[List[str]] = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "DontCare":
                # skip unwanted objects
                continue
            file_objs.append(parts)
        if verbose:
            print(f"{p.name}: {len(file_objs)} objects")

        total_obj += len(file_objs)
        objects[p.name] = file_objs  # store ALL objects for that file (not just the last one)
    return total_obj, objects


def read_calib_files(directory_path: str, verbose: bool = False) -> Tuple[int, Dict[str, List[List[str]]]]:
    """
    Reads all .txt label files in a directory.
    Returns (total_objects, objects_dict) where objects_dict maps filename -> list of parsed rows.
    Last file: 007480.txt\n
    Note: This only parses P2 (left color camera), discards the rest.
    """
    calib_p2: dict = {}
    total_calib = 0
    for p in Path(directory_path).glob("*.txt"):
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            # print(lines)
        except OSError as e:
            print(f"Error reading file {p.name}: {e}")
            continue
        file_calibs: List[List[str]] = []
        for line in lines:
            calib = line.strip().split()
            if 'P2' not in calib[0]:
                continue
                # skip unwanted objects
            if 'P2' in calib[0]:
                calib.pop(0)
            file_calibs.append(np.array(calib,dtype=float))

        if verbose:
            print(f"{p.name}: {len(file_calibs)} objects")
        total_calib += len(file_calibs)
        calib_p2[p.name] = file_calibs  # store ALL objects for that file0
    return total_calib, calib_p2


def load_or_build(labels_dir: str,read_function:Callable[[str, bool], Tuple[int,dict]],cache_path: str = "data.pkl", verbose: bool = False):
    """
    If cache_path exists, load and return it.
    Otherwise build from labels_dir, save, and return.
    """
    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading cached data from {cache_path}")
        return load_pickle(cache_path)

    if verbose:
        print("Building data from label files...")
    total_objects, objects = read_function(labels_dir, verbose=verbose)
    payload = {"total_items": total_objects, "items": objects}

    save_pickle(payload, cache_path)
    if verbose:
        print(f"Saved cache to {cache_path}")
    return payload

if __name__ == "__main__":
    calib_data = load_or_build('../data_object_calib/training/calib/',read_calib_files, cache_path="calib_data.pkl", verbose=False)
    label_data = load_or_build("../training/label_2/",read_label_files, cache_path="label_data.pkl", verbose=False)