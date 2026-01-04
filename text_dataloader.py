import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import ast

# ======================
# Constants
# ======================
DATA_PATH = "data/text/data/train.tsv"
MAX_LEN = 128

# GoEmotions → 7-class mapping
LABEL_MAP = {
    0: 0,    # anger
    1: 1,    # fear
    2: 2,    # joy -> happy
    3: 3,    # sadness
    27: 4,   # neutral
    8: 5,    # surprise
    15: 6    # calm
}

# ======================
# Helper: normalize labels
# ======================
def parse_labels(raw):
    # Case 1: already list
    if isinstance(raw, list):
        return raw

    # Case 2: string
    if isinstance(raw, str):
        raw = raw.strip()

        # "[27, 15]"
        if raw.startswith("["):
            try:
                return ast.literal_eval(raw)
            except:
                return []

        # "8,20"
        if "," in raw:
            try:
                return [int(x) for x in raw.split(",")]
            except:
                return []

        # "27"
        try:
            return [int(raw)]
        except:
            return []

    # Case 3: int
    if isinstance(raw, int):
        return [raw]

    return []

# ======================
# Dataset
# ======================
class TextEmotionDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(
            DATA_PATH,
            sep="\t",
            header=None,
            names=["text", "labels", "id"]
        )

        self.samples = []

        for _, row in df.iterrows():
            labels = parse_labels(row["labels"])

            for l in labels:
                if l in LABEL_MAP:
                    self.samples.append(
                        (row["text"], LABEL_MAP[l])
                    )
                    break

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print("Usable samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# ======================
# Sanity Check
# ======================
if __name__ == "__main__":
    dataset = TextEmotionDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Attention mask shape:", batch["attention_mask"].shape)
    print("Labels:", batch["label"])
