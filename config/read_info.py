import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from typing import Callable

HERE = Path(__file__).resolve().parent.parent
# Adjust this to your repo layout
LABELS_DIR = (HERE / "training" / "label_2").as_posix()
CALIB_DIR  = (HERE / "data_object_calib" / "training" / "calib").as_posix()
IMAGE_DIR  = (HERE / "data_object_image_2" /"training" / "image_2").as_posix()   # <- add this



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
        key = p.stem  # "000123" instead of "000123.txt"
        if verbose:
            print(f"{p.name}: {len(file_objs)} objects")

        total_obj += len(file_objs)
        objects[key] = file_objs  # store ALL objects for that file (not just the last one)
    return total_obj, objects


def read_calib_files(
    directory_path: str,
    verbose: bool = False
) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Reads all KITTI calibration .txt files.

    Returns:
    - total_calib: number of P2 matrices successfully parsed
    - calib_p2: dict mapping image_id (e.g. '000123') -> P2 (3x4 numpy array)

    Only parses P2 (left color camera), discards the rest.
    """
    calib_p2: Dict[str, np.ndarray] = {}
    total_calib = 0

    for p in Path(directory_path).glob("*.txt"):
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except OSError as e:
            print(f"Error reading file {p.name}: {e}")
            continue

        p2_rows = []

        for line in lines:
            calib = line.strip().split()
            if not calib:
                continue

            # match 'P2:' or 'P2'
            if calib[0].startswith("P2"):
                # drop first token, keep 12 floats
                vals = calib[1:]
                p2_rows.append(np.array(vals, dtype=float))

        if not p2_rows:
            if verbose:
                print(f"Warning: no P2 in {p.name}")
            continue

        if len(p2_rows) > 1 and verbose:
            print(f"Warning: multiple P2 rows in {p.name}, using first one")

        key = p.stem  # "000123"
        P2 = ensure_P2_matrix(p2_rows[0])  # single 3x4 matrix

        calib_p2[key] = P2
        total_calib += 1

        if verbose:
            print(f"{key}: stored P2 with shape {P2.shape}")

    return total_calib, calib_p2


def load_or_build(labels_dir: str,read_function:Callable[[str, bool], Tuple[int,dict]],cache_path: str = "data.pkl", verbose: bool = False, rebuild=False):
    """
    If cache_path exists, load and return it.
    Otherwise build from labels_dir, save, and return.
    """

    if (not rebuild) and os.path.exists(cache_path):
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

def fetchCalibrationData(datapath, pkl_path) -> dict:
    calib_data = load_or_build(datapath,read_calib_files, cache_path=pkl_path, verbose=False)
    return calib_data['items']

def fetchLabelData(datapath,pkl_path) -> dict :
    label_data = load_or_build(datapath,read_label_files, cache_path=pkl_path, verbose=False)
    return label_data['items']


def ensure_P2_matrix(P2: np.ndarray) -> np.ndarray:
    """
    Convert KITTI P2 to a (3,4) camera matrix if needed.

    Expected input:
      - shape (12,)
      - or shape (1, 12)
      - or shape (3, 4)
    """
    P2 = np.asarray(P2)

    if P2.shape == (12,):
        return P2.reshape(3, 4)
    if P2.shape == (1, 12):
        return P2.reshape(3, 4)
    if P2.shape == (3, 4):
        return P2

    raise ValueError(f"Unexpected P2 shape {P2.shape}; expected 12, (1,12), or (3,4).")


# me testing
if __name__ == "__main__":
    calib_data :dict = fetchCalibrationData(CALIB_DIR, f"calib_data.pkl")
    print("LABELS_DIR:", LABELS_DIR)
    print("CALIB_DIR :", CALIB_DIR)
    print("IMAGE_DIR :", IMAGE_DIR)

    print("CALIB_DIR exists? ", Path(CALIB_DIR).exists())
    print("LABELS_DIR exists?", Path(LABELS_DIR).exists())

    print("Num calib txt:", len(list(Path(CALIB_DIR).glob("*.txt"))))
    print("Num label txt:", len(list(Path(LABELS_DIR).glob("*.txt"))))


    i=0
    print("Calib Data")
    for k,v in calib_data.items():
        print(f"Key:{k}\nValue: {v}")
        if i > 10: break
        i+=1
    print("Label Data")
    label_data = fetchLabelData(LABELS_DIR, f"label_data.pkl")
    i=0
    for k,v in label_data.items():
        print(f"Key:{k}\nValue: {v}")
        if i > 10: break
        i+=1