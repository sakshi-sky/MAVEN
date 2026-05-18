import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        self.label_map = {
            "angry": 0,
            "happy": 1,
            "sad": 2,
            "neutral": 3
        }

        # 🔥 Transform: VERY IMPORTANT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT expects this
            transforms.ToTensor()
        ])

        for folder in os.listdir(root_dir):
            if folder not in self.label_map:
                continue

            folder_path = os.path.join(root_dir, folder)

            for file in os.listdir(folder_path):
                if file.endswith(".jpg"):
                    self.samples.append((
                        os.path.join(folder_path, file),
                        self.label_map[folder]
                    ))
            self.samples = self.samples[:2000]  # 🔥 limit dataset size for faster training
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)  # 🔥 convert to tensor

        return image, label