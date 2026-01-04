import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths
TRAIN_DIR = "data/face/train"
VAL_DIR   = "data/face/val"
TEST_DIR  = "data/face/test"

# Image transformations (DO NOT CHANGE)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Datasets
class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("Empty dataset")


def make_imagefolder_or_empty(directory, transform):
    # Resolve dataset path relative to this script if not absolute
    p = Path(directory)
    if not p.is_absolute():
        p = (Path(__file__).parent / p).resolve()
    if not p.exists():
        print(f"Warning: dataset directory not found: {p}")
        return EmptyDataset()

    # check for class subfolders
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"Warning: no class subfolders found in {p} (expected class folders).")
        return EmptyDataset()

    return datasets.ImageFolder(str(p), transform=transform)


train_dataset = make_imagefolder_or_empty(TRAIN_DIR, transform)
val_dataset   = make_imagefolder_or_empty(VAL_DIR, transform)
test_dataset  = make_imagefolder_or_empty(TEST_DIR, transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Sanity check
if __name__ == "__main__":
    if len(train_dataset) > 0:
        images, labels = next(iter(train_loader))
        print("Train batch image shape:", images.shape)
        print("Train batch label shape:", labels.shape)
    else:
        print("Train dataset is empty.")

    if hasattr(train_dataset, 'classes'):
        print("Number of classes:", len(train_dataset.classes))
        print("Class names:", train_dataset.classes)
    else:
        print("Train dataset has no classes.")
