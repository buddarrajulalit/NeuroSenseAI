import librosa
import numpy as np
import os

TARGET_SR = 22050
N_MFCC = 40

# Find one audio file
SAMPLE_AUDIO = None
for root, _, files in os.walk("data/speech/train"):
    for f in files:
        if f.endswith(".wav"):
            SAMPLE_AUDIO = os.path.join(root, f)
            break
    if SAMPLE_AUDIO:
        break

if SAMPLE_AUDIO is None:
    raise RuntimeError("No .wav files found")

# Load audio
signal, sr = librosa.load(SAMPLE_AUDIO, sr=TARGET_SR, mono=True)

# Extract MFCC
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=TARGET_SR,
    n_mfcc=N_MFCC,
    n_fft=1024,
    hop_length=512
)

print("Audio file:", SAMPLE_AUDIO)
print("Waveform shape:", signal.shape)
print("MFCC shape:", mfcc.shape)
