import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torch.nn as nn


def build_age_model():
    model = YOLO("yolo26n-cls.pt")
    model.model.model[10].linear = nn.Linear(in_features=1280, out_features=1, bias=True)
    return model

if __name__ == "__main__":

    model = build_age_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open("/home/omid/Age-Estimation/data/train/47_0_0_20170109012806452.jpg").convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    model.model.to("cuda")
    image_tensor = image_tensor.to("cuda")
    model.model.eval()

    with torch.no_grad():
        output = model.model(image_tensor)

    predicted_age = output[1]
    print(predicted_age)
    print(predicted_age.shape)

    true_age = torch.tensor([[47.0]], device="cuda")
    loss_fn = nn.L1Loss()
    loss = loss_fn(predicted_age, true_age)
    print(loss)