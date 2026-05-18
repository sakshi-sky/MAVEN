import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fer_dataset import FERDataset
from vit_encoder import ViTEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5


class FERModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = ViTEncoder()
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, images):
        features = self.encoder(images)  # (B, 768)
        return self.classifier(features)


def main():
    print("🚀 Loading FER dataset...")

    train_dataset = FERDataset("data/fer/train")

    # 🔥 LIMIT DATA FOR SPEED (REMOVE LATER)
    train_dataset.samples = train_dataset.samples[:2000]

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # ✅ Windows safe
    )

    print("✅ Dataset loaded:", len(train_dataset))

    model = FERModel(num_classes=4).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ---------------- TRAINING ----------------
    for epoch in range(EPOCHS):
        print(f"\n🚀 Starting Epoch {epoch+1}")

        model.train()
        total_loss = 0

        for i, (images, labels) in enumerate(train_loader):

            if i % 50 == 0:
                print(f"Batch {i}")

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # ---------------- SAVE MODEL ----------------
    torch.save(model.encoder.vit.state_dict(), "vit_fer_pretrained.pth")
    print("✅ Saved ViT pretrained weights!")


# 🔥 CRITICAL FOR WINDOWS
if __name__ == "__main__":
    main()