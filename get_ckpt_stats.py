import torch
from dataset import KittiObjectDataset

# 1. Load the Dataset to calculate stats
print("Loading dataset stats...")
ds = KittiObjectDataset(mode='train') # This runs get_class_stats()

# 2. Load your existing model file
path = 'resnet18_model/best_model.pth'
checkpoint = torch.load(path)

# 3. Inject the data
checkpoint['class_to_idx'] = ds.class_to_idx
checkpoint['avg_heights'] = ds.avg_heights

# 4. Save it back
torch.save(checkpoint, path)
print(f"Updated {path} with metadata!")
print(f"Classes: {ds.class_to_idx}")
print(f"Heights: {ds.avg_heights}")