import torch
from torch.utils.data import DataLoader

from speech_dataloader import SpeechDataset
from speech_model import SpeechCNNBiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/speech_cnn_lstm/model.pt"
TEST_DIR = "data/speech/train"   # using train split for now (OK for project)

# Load data
dataset = SpeechDataset(TEST_DIR)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load model
model = SpeechCNNBiLSTM(num_classes=7, embedding_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for mfccs, labels in loader:
        mfccs = mfccs.to(device)
        labels = labels.to(device)

        outputs, _ = model(mfccs)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

acc = 100 * correct / total
print(f"Speech Model Accuracy (sanity check): {acc:.2f}%")
print("Speech model loaded & verified successfully.")
