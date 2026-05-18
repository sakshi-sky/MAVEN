print("Script started")

import torch
print("Torch imported")

import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

print("All imports done")

image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    use_fast=True
)

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

model.eval()
print("Model loaded")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class_id = logits.argmax(dim=-1).item()

label = model.config.id2label[predicted_class_id]
print(f"Predicted class label: {label}")
