import numpy as np
from sklearn.preprocessing import StandardScaler

# ======================
# Load embeddings
# ======================
face = np.load("embeddings/face.npy")
speech = np.load("embeddings/speech.npy")
text = np.load("embeddings/text.npy")

print("Face shape:", face.shape)
print("Speech shape:", speech.shape)
print("Text shape:", text.shape)

# ======================
# Normalize (VERY IMPORTANT)
# ======================
face = StandardScaler().fit_transform(face)
speech = StandardScaler().fit_transform(speech)
text = StandardScaler().fit_transform(text)

# ======================
# Project text: 768 → 128
# ======================
W_text = np.random.randn(text.shape[1], 128) * 0.01
text_proj = text @ W_text

# ======================
# Save prepared embeddings
# ======================
np.save("embeddings/face_norm.npy", face)
np.save("embeddings/speech_norm.npy", speech)
np.save("embeddings/text_proj.npy", text_proj)

print("Prepared embeddings saved.")
print("Text projected shape:", text_proj.shape)
