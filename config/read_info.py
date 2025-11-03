import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

def read_files(directory_path: str, verbose: bool = False) -> Tuple[int, Dict[str, List[List[str]]]]:
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


def save_pickle(data, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as f:   # <-- 'rb', not 'wb'
        return pickle.load(f)


def load_or_build(labels_dir: str, cache_path: str = "corpus.pkl", verbose: bool = False):
    """
    If cache_path exists, load and return it.
    Otherwise build from labels_dir, save, and return.
    """
    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading cached corpus from {cache_path}")
        return load_pickle(cache_path)

    if verbose:
        print("Building corpus from label files...")
    total_objects, objects = read_files(labels_dir, verbose=verbose)
    payload = {"total_objects": total_objects, "objects": objects}

    save_pickle(payload, cache_path)
    if verbose:
        print(f"Saved cache to {cache_path}")
    return payload


if __name__ == "__main__":
    data = load_or_build("../training/label_2/", cache_path="corpus.pkl", verbose=True)
    # Access:
    # data["total_objects"], data["objects"]
    objects= data["objects"]
    for key, value in objects.items():
        print(f"Key: {key}, Value: {value}\n")
    print(data['total_objects'])