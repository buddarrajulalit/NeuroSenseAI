import torch
import torch.nn.functional as F
import numpy as np
from final_model import NeuroSenseAI, build_adjacency

# Load prepared embeddings
face = torch.tensor(np.load("embeddings/face_norm.npy"), dtype=torch.float32)
speech = torch.tensor(np.load("embeddings/speech_norm.npy"), dtype=torch.float32)
text = torch.tensor(np.load("embeddings/text_proj.npy"), dtype=torch.float32)

# Align sizes
N = min(len(face), len(speech), len(text))
face, speech, text = face[:N], speech[:N], text[:N]

# Build adjacency
with torch.no_grad():
    fused_temp = (face + speech + text) / 3.0
    adj = build_adjacency(fused_temp)

# Load model
model = NeuroSenseAI(num_classes=7)
model.eval()

# Forward
with torch.no_grad():
    logits = model(face, speech, text, adj)
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

print("Prediction shape:", preds.shape)
print("Sample predictions:", preds[:10])
