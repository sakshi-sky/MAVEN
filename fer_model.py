import torch.nn as nn
from vit_encoder import ViTEncoder

class FERModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ViTEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

    def forward(self, images):
        emb = self.encoder(images)  # (B, 768)
        return self.classifier(emb)