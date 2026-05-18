import torch
from transformers import ViTModel, AutoImageProcessor

class ViTEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        )

    def forward(self, images):
        if isinstance(images, torch.Tensor):
            # Already tensors → skip processor
            outputs = self.vit(pixel_values=images)
        else:
            inputs = self.processor(images, return_tensors="pt")
            outputs = self.vit(**inputs)

        return outputs.last_hidden_state[:, 0] # CLS token embedding

if __name__ == "__main__":
    from PIL import Image
    import requests

    print("Testing ViTEncoder...")

    model = ViTEncoder()
    model.eval()

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    with torch.no_grad():
        embedding = model([image])

    print("Output shape:", embedding.shape)
