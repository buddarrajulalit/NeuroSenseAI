import torch
import torch.nn as nn
from transformers import BertModel

class TextBERTClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        x = self.dropout(cls_embedding)
        logits = self.classifier(x)  # [B, 7]

        return logits, cls_embedding


# ======================
# Sanity Check
# ======================
if __name__ == "__main__":
    model = TextBERTClassifier()

    dummy_input_ids = torch.randint(0, 30522, (2, 128))
    dummy_attention = torch.ones((2, 128), dtype=torch.long)

    out, emb = model(dummy_input_ids, dummy_attention)

    print("Logits shape:", out.shape)
    print("Embedding shape:", emb.shape)
