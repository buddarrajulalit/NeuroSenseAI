import torch
from text_model import TextBERTClassifier

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = "models/text_bert/model.pt"

# Load model (pretrained / partially trained)
model = TextBERTClassifier(num_classes=7).to(device)
model.eval()

# Save state_dict
torch.save(model.state_dict(), MODEL_PATH)

print("Text BERT model saved and frozen at:")
print(MODEL_PATH)

# --------- Load check ----------
test_model = TextBERTClassifier(num_classes=7).to(device)
test_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
test_model.eval()

print("Load verification successful.")
