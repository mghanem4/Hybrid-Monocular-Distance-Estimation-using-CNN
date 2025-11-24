# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
from dataset import (
    KittiObjectDataset,
    load_or_build,
    read_label_files,
    read_calib_files,
    get_valid_image_ids,
    load_or_build_split_ids,
    LABELS_DIR, CALIB_DIR, IMAGE_DIR, CACHE_DIR
)

def _get_unique_cls(label_dict:dict):
    print("here")
    classes= set()
    for _id, obj in label_dict.items():
        for type in obj:
            print(type)
            cls =type['class_name']
            classes.add(cls)
    return classes

def main():
    print("Setting up data...")
    print(f"Label directory: {LABELS_DIR}")
    print(f"Calib directory: {CALIB_DIR}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    # 1. Load Raw Data (Label & Calib Dictionaries)
    try:
        calib_dict = load_or_build(CALIB_DIR, read_calib_files, f"{CACHE_DIR}/calib_data.pkl")
        label_dict = load_or_build(LABELS_DIR, read_label_files, f"{CACHE_DIR}/label_data.pkl")
        print("here")
        classes =_get_unique_cls(label_dict)

        print(classes)


        # 2. Compute Splits
        valid_ids = get_valid_image_ids(label_dict, calib_dict, IMAGE_DIR)
        split_data = load_or_build_split_ids(valid_ids, f"{CACHE_DIR}/split_ids.pkl")
        train_ids = split_data['train_ids']
        val_ids = split_data['val_ids']


    except Exception as e:
        print(f"Warning: Data loading failed ({e}). using fallback dummy logic if needed or crashing.")
        train_ids, val_ids = [], []
        calib_dict, label_dict = {}, {}

        # i will debug the rest later
        
if __name__ == "__main__":
    main()