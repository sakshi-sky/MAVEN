import os
import csv
import re

DATA_ROOT = "data/raw/IEMOCAP_full_release"
OUTPUT_CSV = "data/iemocap_labels.csv"

emotion_map = []

pattern = re.compile(r"\[.*?\]\s+(\S+)\s+(\w+)")

for session in os.listdir(DATA_ROOT):
    if not session.startswith("Session"):
        continue

    emo_path = os.path.join(
        DATA_ROOT,
        session,
        "dialog",
        "EmoEvaluation"
    )

    if not os.path.exists(emo_path):
        continue

    for file in os.listdir(emo_path):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(emo_path, file)

        with open(file_path, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    utterance_id = match.group(1)
                    emotion = match.group(2)

                    emotion_map.append((utterance_id, emotion))


# Write CSV
os.makedirs("data", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["utterance_id", "emotion"])

    for uid, emo in emotion_map:
        writer.writerow([uid, emo])

print(f"Saved {len(emotion_map)} labels to {OUTPUT_CSV}")
