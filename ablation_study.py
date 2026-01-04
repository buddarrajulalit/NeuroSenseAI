import torch
import torch.nn.functional as F
import numpy as np
from final_model import NeuroSenseAI, build_adjacency

# Load embeddings
face = torch.tensor(np.load("embeddings/face_norm.npy"), dtype=torch.float32)
speech = torch.tensor(np.load("embeddings/speech_norm.npy"), dtype=torch.float32)
text = torch.tensor(np.load("embeddings/text_proj.npy"), dtype=torch.float32)

N = min(len(face), len(speech), len(text))
face, speech, text = face[:N], speech[:N], text[:N]

# ========== BASELINE ==========
baseline_fused = (face + speech + text) / 3.0

# ========== ATTENTION ONLY ==========
from final_model import AttentionFusion
attn = AttentionFusion()
attn_fused = attn(face, speech, text)

# ========== GNN ONLY ==========
adj = build_adjacency(baseline_fused)
from final_model import GCNLayer
gcn = GCNLayer(128, 128)
gnn_out = gcn(baseline_fused, adj)

# ========== FULL MODEL ==========
model = NeuroSenseAI(num_classes=7)
full_out = model(face, speech, text, adj)

print("Ablation outputs:")
print("Baseline:", baseline_fused.shape)
print("Attention only:", attn_fused.shape)
print("GNN only:", gnn_out.shape)
print("Full model:", full_out.shape)
