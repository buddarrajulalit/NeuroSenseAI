import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW   # ✅ FIXED IMPORT

from text_dataloader import TextEmotionDataset
from text_model import TextBERTClassifier

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Dataset & Loader
# ======================
dataset = TextEmotionDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ======================
# Model
# ======================
model = TextBERTClassifier(num_classes=7).to(device)

# ======================
# Loss & Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# ======================
# Training
# ======================
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs, _ = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Train Acc: {acc:.2f}%"
    )

print("Training completed.")
