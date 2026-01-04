import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from speech_dataloader import SpeechDataset
from speech_model import SpeechCNNBiLSTM

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Paths
# ======================
TRAIN_DIR = "data/speech/train"
MODEL_DIR = "models/speech_cnn_lstm"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# Dataset & Loader
# ======================
train_dataset = SpeechDataset(TRAIN_DIR)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True
)

print("Total training samples:", len(train_dataset))

# ======================
# Model
# ======================
model = SpeechCNNBiLSTM(
    num_classes=7,
    embedding_dim=128
).to(device)

# ======================
# Loss & Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ======================
# Training settings
# ======================
EPOCHS = 10

# ======================
# Training loop
# ======================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for mfccs, labels in train_loader:
        mfccs = mfccs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(mfccs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Train Acc: {acc:.2f}%"
    )

# ======================
# SAVE MODEL (FREEZE STEP)
# ======================
torch.save(model.state_dict(), MODEL_PATH)
print("Speech model saved at:", MODEL_PATH)
print("Training completed and model frozen.")
