import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ======================
# Load fused embeddings
# ======================
fused = torch.load("embeddings/fused.pt") if False else None
# We recompute fused embeddings here for safety
face = torch.tensor(np.load("embeddings/face_norm.npy"), dtype=torch.float32)
speech = torch.tensor(np.load("embeddings/speech_norm.npy"), dtype=torch.float32)
text = torch.tensor(np.load("embeddings/text_proj.npy"), dtype=torch.float32)

N = min(len(face), len(speech), len(text))
face, speech, text = face[:N], speech[:N], text[:N]

# Simple equal fusion (already attention-tested earlier)
fused = (face + speech + text) / 3.0

print("Using fused embeddings:", fused.shape)

# ======================
# Build Graph (cosine similarity)
# ======================
def build_adjacency(x, threshold=0.8):
    x_norm = F.normalize(x, dim=1)
    sim = torch.mm(x_norm, x_norm.t())

    adj = (sim > threshold).float()
    adj.fill_diagonal_(1.0)
    return adj

adj = build_adjacency(fused)
print("Adjacency matrix shape:", adj.shape)

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

        out = adj_norm @ x
        return self.linear(out)

# ======================
# GNN Model
# ======================
class EmotionGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(128, 128)
        self.gcn2 = GCNLayer(128, 128)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        return x

# ======================
# Run GNN
# ======================
model = EmotionGNN()
refined = model(fused, adj)

print("Refined embeddings shape:", refined.shape)
