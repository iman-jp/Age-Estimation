import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torchvision import transforms
from PIL import Image

from model import build_age_model


def predict_age(checkpoint_path, image_path):
    model = build_age_model()
    model.model.to("cuda")

    checkpoint = torch.load(checkpoint_path)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  (trained to epoch {checkpoint['epoch']}, best val loss: {checkpoint['best_val_loss']:.4f})")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to("cuda")

    model.model.eval()
    with torch.no_grad():
        output = model.model(image_tensor)
        predicted_age = output[1] if isinstance(output, tuple) else output

    predicted_age = predicted_age.item()
    print(f"\nImage: {image_path}")
    print(f"Predicted age: {predicted_age:.1f} years")

    return predicted_age


if __name__ == "__main__":
    checkpoint_path = "checkpoints/bs64_ep40_lr0.001_classWeightedHybrid_capMultiplier10.0_bucketThreshold65_bucketSize10_20260903_224611_softplus.pt" 
    image_path = "/mnt/c/Users/User/Downloads/WIN_20260904_12_36_41_Pro.jpg"

    predict_age(checkpoint_path, image_path)