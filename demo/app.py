import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer
from text_model import TextBERTClassifier
import torch.nn.functional as F


# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="NeuroSenseAI Demo",
    layout="centered"
)

st.title("🧠 NeuroSenseAI")
st.subheader("Multimodal Emotion Intelligence System")

# ----------------------
# Load model & tokenizer
# ----------------------
@st.cache_resource
def load_model():
    model = TextBERTClassifier(num_classes=7)
    model.load_state_dict(
        torch.load("models/text_bert/model.pt", map_location="cpu")
    )
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

# ----------------------
# Emotion labels
# ----------------------
EMOTIONS = [
    "Angry 😠",
    "Fear 😨",
    "Happy 😊",
    "Sad 😢",
    "Neutral 😐",
    "Surprised 😲",
    "Calm 😌"
]

# ----------------------
# User input
# ----------------------
text = st.text_area(
    "Enter text:",
    "I am feeling really excited today!"
)

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits, embedding = model(
                encoding["input_ids"],
                encoding["attention_mask"]
            )

            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100

        st.success(f"**Predicted Emotion:** {EMOTIONS[pred]}")
        st.write(f"**Confidence:** {confidence:.2f}%")
