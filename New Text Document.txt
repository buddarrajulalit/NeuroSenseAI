import numpy as np
import torch
import torch.nn as nn

# ======================
# Load prepared embeddings
# ======================
face = np.load("embeddings/face_norm.npy")
speech = np.load("embeddings/speech_norm.npy")
text = np.load("embeddings/text_proj.npy")

# Align sample size
N = min(len(face), len(speech), len(text))
face = face[:N]
speech = speech[:N]
text = text[:N]

# Convert to tensors
face = torch.tensor(face, dtype=torch.float32)
speech = torch.tensor(speech, dtype=torch.float32)
text = torch.tensor(text, dtype=torch.float32)

print("Aligned samples:", N)

# ======================
# Attention Fusion Model
# ======================
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, face, speech, text):
        # Compute attention scores
        a_f = self.attn(face)
        a_s = self.attn(speech)
        a_t = self.attn(text)

        scores = torch.cat([a_f, a_s, a_t], dim=1)
        weights = torch.softmax(scores, dim=1)

        fused = (
            weights[:, 0:1] * face +
            weights[:, 1:2] * speech +
            weights[:, 2:3] * text
        )

        return fused, weights


# ======================
# Run fusion
# ======================
model = AttentionFusion()
fused_embeddings, attn_weights = model(face, speech, text)

print("Fused embedding shape:", fused_embeddings.shape)
print("Attention weights shape:", attn_weights.shape)
print("Sample attention weights:", attn_weights[0])
