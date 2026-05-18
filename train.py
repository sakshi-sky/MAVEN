import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from collections import Counter

from dataset import IEMOCAPVideoDataset
from temporal_model import EmotionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 7


# ---------------------------------------------------
# Collate Function (variable-length videos)
# ---------------------------------------------------
def collate_fn(batch):
    videos = []
    labels = []

    for frames, label in batch:
        videos.append(frames)
        labels.append(label)

    return videos, torch.tensor(labels)


# ---------------------------------------------------
# Load datasets (Session split)
# ---------------------------------------------------
train_dataset1 = IEMOCAPVideoDataset(
    "data/frames/Session1",
    "data/iemocap_labels.csv"
)

train_dataset2 = IEMOCAPVideoDataset(
    "data/frames/Session2",
    "data/iemocap_labels.csv"
)

train_dataset3 = IEMOCAPVideoDataset(
    "data/frames/Session3",
    "data/iemocap_labels.csv"
)

val_dataset = IEMOCAPVideoDataset(
    "data/frames/Session4",
    "data/iemocap_labels.csv"
)

# Combine datasets
train_dataset = ConcatDataset([
    train_dataset1,
    train_dataset2,
    train_dataset3
])


# ---------------------------------------------------
# 🔍 Label Distribution
# ---------------------------------------------------
all_labels = []
for i in range(len(train_dataset)):
    _, label = train_dataset[i]
    all_labels.append(label)

label_counts = Counter(all_labels)
total = sum(label_counts.values())

print("Label Distribution:", label_counts)

# 🔥 Auto class weights
class_weights = torch.tensor([
    total / label_counts[i] for i in range(4)
]).to(DEVICE)

print("Class Weights:", class_weights)


# ---------------------------------------------------
# DataLoaders
# ---------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ---------------------------------------------------
# Model
# ---------------------------------------------------
model = EmotionModel(num_classes=4).to(DEVICE)

# 🔥 Freeze ViT initially
for param in model.encoder.vit.parameters():
    param.requires_grad = False


# ---------------------------------------------------
# Loss & Optimizer
# ---------------------------------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)


# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
for epoch in range(EPOCHS):

    # 🔓 Unfreeze after 3 epochs
    if epoch == 3:
        print("🔓 Unfreezing ViT...")

        for param in model.encoder.vit.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5
        )

    model.train()
    total_loss = 0

    for videos, labels in train_loader:
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(videos)
        loss = criterion(outputs, labels)

        loss.backward()

        # 🔥 Gradient clipping (VERY IMPORTANT)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

    # ---------------- Validation ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            labels = labels.to(DEVICE)

            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1)

            # 🔍 Debug predictions
            print("Pred:", preds.item(), "Label:", labels.item())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {100 * correct/total:.2f}%")