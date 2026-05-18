from label_mapping import RAW_TO_4CLASS, EMOTION_TO_INDEX

import os
import csv
from torch.utils.data import Dataset
from PIL import Image


class IEMOCAPVideoDataset(Dataset):
    def __init__(self, root_dir, label_csv):
        self.root_dir = root_dir
        self.video_dirs = []
        self.labels_dict = {}

        # Load CSV mapping
        with open(label_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                video_id, emotion = row
                self.labels_dict[video_id] = emotion

        # Collect valid video folders
        for root, dirs, files in os.walk(root_dir):
            if len(files) > 0:
                video_id = os.path.basename(root)

                # find all utterances belonging to this video
                matching_keys = [
                    k for k in self.labels_dict.keys()
                    if k.startswith(video_id)
                ]

                if len(matching_keys) == 0:
                    continue

                # take first match (simple strategy for now)
                raw_label = self.labels_dict[matching_keys[0]]

                if raw_label in RAW_TO_4CLASS:
                    self.video_dirs.append(root)

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        video_id = os.path.basename(video_dir)

        # Load frames
        frames = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith(".jpg")
        ])

        frames=frames[:15]      #limiting number of frames

        images = []
        for frame_path in frames:
            image = Image.open(frame_path).convert("RGB")
            images.append(image)

        # label mapping
        matching_keys = [
            k for k in self.labels_dict.keys()
            if k.startswith(video_id)
        ]

        if len(matching_keys) == 0:
            raise ValueError(f"No label found for {video_id}")

        raw_label = self.labels_dict[matching_keys[0]]

        emotion_name = RAW_TO_4CLASS[raw_label]
        label_index = EMOTION_TO_INDEX[emotion_name]

        return images, label_index

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    dataset = IEMOCAPVideoDataset(
        os.path.join(BASE_DIR, "data/frames"),
        os.path.join(BASE_DIR, "data/iemocap_labels.csv")
    )

    print("Number of videos:", len(dataset))

    if len(dataset) > 0:
        frames, label = dataset[0]
        print("Frames in first video:", len(frames))
        print("Label index:", label)
    else:
        print("Dataset is empty!")
    print("Frames in first video:", len(frames))
    print("Label index:", label)
