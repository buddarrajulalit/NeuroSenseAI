import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from face_model import get_face_model
from pathlib import Path

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = "data/face/train"
MODEL_PATH = "models/face_cnn/model.pt"
SAVE_PATH = "embeddings/face.npy"

# Transform (DO NOT CHANGE)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset & Loader
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load model state dict and handle different saved architectures
sd_path = Path(MODEL_PATH)
if not sd_path.exists():
    raise SystemExit(f"Model file not found: {MODEL_PATH}. Run training first.")

state = torch.load(str(sd_path), map_location=device)

# detect keys to decide which architecture was saved
keys = list(state.keys())
is_resnet = any(k.startswith("conv1") or k.startswith("layer1") or k.startswith("fc.") for k in keys)
is_small = any(k.startswith("0.") and "weight" in k for k in keys)

if is_resnet:
    num_classes = state.get("fc.weight").shape[0] if "fc.weight" in state else 7
    model = get_face_model(num_classes=num_classes, small=False)
    model.load_state_dict(state)
    # remove final classification layer to get 512-d features
    model.fc = nn.Identity()
elif is_small:
    # infer num_classes from small model's final linear (module 9)
    # small model keys include '9.weight' with shape [num_classes, 128]
    if "9.weight" in state:
        num_classes = state["9.weight"].shape[0]
    else:
        num_classes = 7
    model = get_face_model(num_classes=num_classes, small=True)
    model.load_state_dict(state)
    # remove final classification layer (last sequential element) to get 128-d features
    if isinstance(model, nn.Sequential):
        model[-1] = nn.Identity()
else:
    raise SystemExit("Unrecognized model state dict format")

model = model.to(device)
model.eval()

# Extract embeddings
embeddings = []

with torch.no_grad():
    for images, _ in loader:
        images = images.to(device)
        feats = model(images)  # [B, 512]
        embeddings.append(feats.cpu().numpy())

embeddings = np.vstack(embeddings)

# Save
np.save(SAVE_PATH, embeddings)

print("Face embeddings saved.")
print("Embedding shape:", embeddings.shape)
