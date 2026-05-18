import os
import pandas as pd
import numpy as np
import librosa

# --- CONFIGURE THESE TWO PATHS ---
LABELS_CSV   = "data/iemocap_labels.csv"
AUDIO_ROOT   = "D:/IEMOCAP_full_release"   # folder containing Session1, Session2, etc.
OUTPUT_CSV   = "data/iemocap_features.csv"
# ----------------------------------

def extract_features(wav_path, sr=16000, n_mfcc=40):
    try:
        y, sr = librosa.load(wav_path, sr=sr)
        mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta  = librosa.feature.delta(mfcc)
        chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr         = librosa.feature.zero_crossing_rate(y)
        rms         = librosa.feature.rms(y=y)

        features = np.concatenate([
            np.mean(mfcc, axis=1),       # 40 features
            np.std(mfcc, axis=1),        # 40
            np.mean(mfcc_delta, axis=1), # 40
            np.mean(chroma, axis=1),     # 12
            [np.mean(zcr)],              # 1
            [np.mean(rms)],              # 1
        ])
        return features
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

df = pd.read_csv(LABELS_CSV)

# Build a map of utterance_id -> wav path by walking the audio directory
wav_map = {}
for root, dirs, files in os.walk(AUDIO_ROOT):
    for f in files:
        if f.endswith(".wav"):
            uid = os.path.splitext(f)[0]
            wav_map[uid] = os.path.join(root, f)

print(f"Found {len(wav_map)} wav files")
print(f"Sample utterance_id: {df['utterance_id'].iloc[0]}")

rows = []
not_found = 0
for _, row in df.iterrows():
    uid = row['utterance_id']
    wav_path = wav_map.get(uid)
    if wav_path is None:
        not_found += 1
        continue
    feats = extract_features(wav_path)
    if feats is not None:
        entry = {'utterance_id': uid, 'emotion': row['emotion']}
        for i, v in enumerate(feats): 
            entry[f'feat_{i}'] = v
        rows.append(entry)

print(f"Extracted features for {len(rows)} utterances ({not_found} not found)")
out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")