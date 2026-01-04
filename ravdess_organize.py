import os
import shutil

BASE_DIR = "data/speech/train"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "08": "surprised"
}

# Create emotion folders
for emotion in emotion_map.values():
    os.makedirs(os.path.join(BASE_DIR, emotion), exist_ok=True)

# Traverse actor folders
for actor in os.listdir(BASE_DIR):
    actor_path = os.path.join(BASE_DIR, actor)

    if not actor.startswith("Actor_"):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        parts = file.split("-")
        emotion_code = parts[2]

        if emotion_code == "07":  # ignore disgust
            continue

        emotion = emotion_map.get(emotion_code)
        if emotion is None:
            continue

        src = os.path.join(actor_path, file)
        dst = os.path.join(BASE_DIR, emotion, file)

        shutil.copy(src, dst)

print("RAVDESS audio organized by emotion.")
