import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from text_model import TextBERTClassifier

# Load model
model = TextBERTClassifier(num_classes=7)
model.load_state_dict(
    torch.load("models/text_bert/model.pt", map_location="cpu")
)
model.eval()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Emotion labels
EMOTIONS = [
    "Angry 😠",
    "Fear 😨",
    "Happy 😊",
    "Sad 😢",
    "Neutral 😐",
    "Surprised 😲",
    "Calm 😌"
]

# Test sentences
test_sentences = [
    "I am feeling very happy today!",
    "I am scared about my exam results",
    "This is so annoying and frustrating",
    "I feel relaxed and peaceful now",
    "I didn't expect this at all!"
]

for text in test_sentences:
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits, _ = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    print("-" * 50)
    print("Text:", text)
    print("Prediction:", EMOTIONS[pred])
    print(f"Confidence: {confidence:.2f}%")
