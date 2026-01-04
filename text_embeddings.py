import torch
import numpy as np
from torch.utils.data import DataLoader

from text_dataloader import TextEmotionDataset
from text_model import TextBERTClassifier

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Paths
# ======================
MODEL_PATH = "models/text_bert/model.pt"
SAVE_PATH = "embeddings/text.npy"

# ======================
# Dataset & Loader
# ======================
dataset = TextEmotionDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# ======================
# Load model
# ======================
model = TextBERTClassifier(num_classes=7).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ======================
# Extract embeddings
# ======================
all_embeddings = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        _, embeddings = model(input_ids, attention_mask)  # [B, 768]
        all_embeddings.append(embeddings.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)

# ======================
# Save
# ======================
np.save(SAVE_PATH, all_embeddings)

print("Text embeddings saved.")
print("Embedding shape:", all_embeddings.shape)
