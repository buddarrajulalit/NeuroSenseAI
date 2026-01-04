import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Constants (LOCK THESE)
TARGET_SR = 22050
N_MFCC = 40
MAX_LEN = 300   # time frames

EMOTION_LABELS = {
    "angry": 0,
    "calm": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6
}

class SpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.root_dir = root_dir

        for emotion in os.listdir(root_dir):
            emotion_path = os.path.join(root_dir, emotion)
            if emotion not in EMOTION_LABELS:
                continue

            for file in os.listdir(emotion_path):
                if file.endswith(".wav"):
                    self.samples.append(
                        (os.path.join(emotion_path, file),
                         EMOTION_LABELS[emotion])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        signal, _ = librosa.load(path, sr=TARGET_SR, mono=True)

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=TARGET_SR,
            n_mfcc=N_MFCC,
            n_fft=1024,
            hop_length=512
        )

        # Pad / Truncate
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
        else:
            mfcc = mfcc[:, :MAX_LEN]

        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        return mfcc, label


if __name__ == "__main__":
    dataset = SpeechDataset("data/speech/train")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    x, y = next(iter(loader))
    print("MFCC batch shape:", x.shape)
    print("Label batch shape:", y.shape)
