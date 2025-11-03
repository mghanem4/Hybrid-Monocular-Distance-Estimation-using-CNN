import cv2
import numpy as np
import os

def pad_and_clip_box(x1, y1, x2, y2, img_w, img_h, pad_ratio=0.1):
    w = x2 - x1
    h = y2 - y1
    pad_x = int(pad_ratio * w)
    pad_y = int(pad_ratio * h)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(img_w - 1, x2 + pad_x)
    y2p = min(img_h - 1, y2 + pad_y)
    return x1p, y1p, x2p, y2p

def crop_from_kitti(img_path, label_line:list, out_size=(128, 128)):
    # Parse the label line
    cropped_objects=[]
    cropped_objects_Z=[]
    for obj in label:
        obj = obj.strip().split()
        if(obj[0]=='DontCare'):
            continue
        print(obj[4:8])
        print(obj)
    # KITTI bbox floats
        x1, y1, x2, y2 = map(float, obj[4:8])
    # Distance target (forward range)
        Z = float(obj[13])  # camera Z in meters
        img = cv2.imread(img_path)[:, :, ::-1]  # to RGB
        H, W = img.shape[:2]
    # Pad and clip
        x1i, y1i, x2i, y2i = pad_and_clip_box(int(x1), int(y1), int(x2), int(y2), W, H, pad_ratio=0.1)
    # Crop and resize
        crop = img[y1i:y2i, x1i:x2i]
        crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)
        cropped_objects.append(crop)
        cropped_objects_Z.append(Z)
    return cropped_objects, cropped_objects_Z



def read_files(directory_path):
    """Reads and prints the content of all .txt files in a given directory using os module."""
    objects=[]
    for filename in os.listdir(directory_path):
        if str(filename).endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    label = file.read()
                    label= label.strip().split('\n')
                    for obj in label:
                        objects.append(obj)
                        print(obj)
            except IOError as e:
                print(f"Error reading file {filename}: {e}")

# Example usage:
# Replace 'your_directory_path' with the actual path to your directory
# read_all_txt_files_os('your_directory_path')
if __name__ == '__main__':
    img_index='000003'
    image_path=f'data_object_image_2/training/image_2/{img_index}.png'
    label_line_path=f'training/label_2/{img_index}.txt'
    with open(label_line_path, "r") as f:
        label = f.read()
        # print('Line: ',label)
    label= label.strip().split('\n')
    # for obj in label:
    #     print('Object:',obj)
    objects_cropped, objects_Z=crop_from_kitti(image_path,label)
    for i, img in enumerate(objects_cropped):
        print('Z', objects_Z[i])            
        cv2.imshow("Image Window", img)
        cv2.waitKey(0)              # keep window alive until a keypress
        cv2.destroyAllWindows()     # properly tear down Qt windows