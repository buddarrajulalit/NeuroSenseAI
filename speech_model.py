import torch
import torch.nn as nn

class SpeechCNNBiLSTM(nn.Module):
    def __init__(self, num_classes=7, embedding_dim=128):
        super().__init__()

        # CNN part (input: 1 x 40 x 300)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),          # -> 16 x 20 x 150

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)           # -> 32 x 10 x 75
        )

        # BiLSTM part
        self.lstm = nn.LSTM(
            input_size=32 * 10,       # features per time step
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Embedding layer
        self.embedding = nn.Linear(128 * 2, embedding_dim)

        # Classification layer
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: [B, 40, 300]
        x = x.unsqueeze(1)           # [B, 1, 40, 300]
        x = self.cnn(x)              # [B, 32, 10, 75]

        # Prepare for LSTM
        x = x.permute(0, 3, 1, 2)    # [B, 75, 32, 10]
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        # -> [B, 75, 320]

        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]   # last time step

        embed = self.embedding(last_out)  # [B, 128]
        logits = self.classifier(embed)   # [B, 7]

        return logits, embed


# Sanity check
if __name__ == "__main__":
    model = SpeechCNNBiLSTM()
    print(model)

    dummy = torch.randn(2, 40, 300)
    out, emb = model(dummy)

    print("Logits shape:", out.shape)
    print("Embedding shape:", emb.shape)
