from fastapi import FastAPI
import torch
import numpy as np
from final_model import NeuroSenseAI, build_adjacency

app = FastAPI(title="NeuroSenseAI API")

# Load embeddings
face = torch.tensor(np.load("embeddings/face_norm.npy"), dtype=torch.float32)
speech = torch.tensor(np.load("embeddings/speech_norm.npy"), dtype=torch.float32)
text = torch.tensor(np.load("embeddings/text_proj.npy"), dtype=torch.float32)

N = min(len(face), len(speech), len(text))
face, speech, text = face[:N], speech[:N], text[:N]

# Build adjacency
adj = build_adjacency((face + speech + text) / 3.0)

# Load model
model = NeuroSenseAI(num_classes=7)
model.eval()

EMOTIONS = [
    "Angry", "Fear", "Happy",
    "Sad", "Neutral", "Surprised", "Calm"
]

@app.get("/")
def root():
    return {"message": "NeuroSenseAI backend running"}

@app.get("/predict")
def predict(sample_id: int = 0):
    with torch.no_grad():
        logits = model(
            face[sample_id:sample_id+1],
            speech[sample_id:sample_id+1],
            text[sample_id:sample_id+1],
            adj
        )
        pred = torch.argmax(logits, dim=1).item()

    return {
        "sample_id": sample_id,
        "emotion": EMOTIONS[pred]
    }
