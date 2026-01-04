import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ======================
# Load prepared embeddings
# ======================
face = torch.tensor(np.load("embeddings/face_norm.npy"), dtype=torch.float32)
speech = torch.tensor(np.load("embeddings/speech_norm.npy"), dtype=torch.float32)
text = torch.tensor(np.load("embeddings/text_proj.npy"), dtype=torch.float32)

# Align sizes
N = min(len(face), len(speech), len(text))
face, speech, text = face[:N], speech[:N], text[:N]

# ======================
# Attention Fusion
# ======================
class AttentionFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, f, s, t):
        a_f = self.attn(f)
        a_s = self.attn(s)
        a_t = self.attn(t)

        scores = torch.cat([a_f, a_s, a_t], dim=1)
        weights = torch.softmax(scores, dim=1)

        fused = (
            weights[:, 0:1] * f +
            weights[:, 1:2] * s +
            weights[:, 2:3] * t
        )
        return fused

# ======================
# GCN Layer
# ======================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D = torch.diag(deg_inv_sqrt)
        adj_norm = D @ adj @ D
        return self.linear(adj_norm @ x)

# ======================
# Final Model
# ======================
class NeuroSenseAI(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.fusion = AttentionFusion()
        self.gcn1 = GCNLayer(128, 128)
        self.gcn2 = GCNLayer(128, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, face, speech, text, adj):
        x = self.fusion(face, speech, text)
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        logits = self.classifier(x)
        return logits

# ======================
# Build adjacency
# ======================
def build_adjacency(x, threshold=0.8):
    x_norm = F.normalize(x, dim=1)
    sim = torch.mm(x_norm, x_norm.t())
    adj = (sim > threshold).float()
    adj.fill_diagonal_(1.0)
    return adj

# ======================
# Run end-to-end
# ======================
model = NeuroSenseAI(num_classes=7)

with torch.no_grad():
    fused_temp = (face + speech + text) / 3.0
    adj = build_adjacency(fused_temp)
    outputs = model(face, speech, text, adj)

print("Final output shape:", outputs.shape)
