import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

from speech_dataloader import SpeechDataset
from speech_model import SpeechCNNBiLSTM

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Paths
# ======================
DATA_DIR = "data/speech/train"
MODEL_PATH = "models/speech_cnn_lstm/model.pt"
SAVE_PATH = "embeddings/speech.npy"

# ======================
# Dataset & Loader
# ======================
dataset = SpeechDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# ======================
# Load model
# ======================
model = SpeechCNNBiLSTM(num_classes=7, embedding_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ======================
# Extract embeddings
# ======================
all_embeddings = []

with torch.no_grad():
    for mfccs, _ in loader:
        mfccs = mfccs.to(device)
        _, embeds = model(mfccs)   # embeds → [B, 128]
        all_embeddings.append(embeds.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)

# ======================
# Save embeddings
# ======================
np.save(SAVE_PATH, all_embeddings)

print("Speech embeddings saved.")
print("Embedding shape:", all_embeddings.shape)
