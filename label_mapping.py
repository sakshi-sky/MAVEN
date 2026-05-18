# label_mapping.py

# Map raw IEMOCAP labels to 4 classes
RAW_TO_4CLASS = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",   # merge excited into happy
    "sad": "sad",
    "neu": "neutral"
}

# Final numeric encoding
EMOTION_TO_INDEX = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3
}

INDEX_TO_EMOTION = {v: k for k, v in EMOTION_TO_INDEX.items()}
