import torch
import torch.nn as nn
from vit_encoder import ViTEncoder


class EmotionModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.encoder = ViTEncoder()

        # 🔥 LOAD FER PRETRAINED WEIGHTS HERE
        self.encoder.vit.load_state_dict(
            torch.load("vit_fer_pretrained.pth")
        )

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, videos):
        batch_outputs = []

        for frames in videos:
            embeddings = []

            # 🔥 Limit frames (VERY IMPORTANT)
            frames = frames[:10]

            for frame in frames:
                emb = self.encoder([frame])  # (1, 768)
                embeddings.append(emb.squeeze(0))

            embeddings = torch.stack(embeddings)  # (T, 768)
            embeddings = embeddings.unsqueeze(0)  # (1, T, 768)

            lstm_out, _ = self.lstm(embeddings)

            # 🔥 BETTER: mean pooling instead of last step
            video_emb = torch.mean(lstm_out, dim=1)

            video_emb = self.dropout(video_emb)

            out = self.classifier(video_emb)  # (1, num_classes)

            batch_outputs.append(out)

        return torch.cat(batch_outputs, dim=0)