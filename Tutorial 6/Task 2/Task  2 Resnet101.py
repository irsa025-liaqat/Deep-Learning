import os
import requests
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Download ImageNet class labels if not already present
labels_path = "imagenet_classes.txt"
if not os.path.exists(labels_path):
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    with open(labels_path, "w") as f:
        f.write(response.text)
    print("imagenet_classes.txt downloaded successfully.")

with open(labels_path) as f:
    labels = [line.strip() for line in f.readlines()]


# Function to load and preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)


# Load ResNet101 model
resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
resnet101.eval()


# Prediction function
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)

        results = []
        for i in range(5):
            results.append((labels[top5_catid[i]], top5_prob[i].item()))
    return results


# Image display helper
def show_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Run prediction
image_path = "/content/WhatsApp Image 2026-05-02 at 8.08.58 PM.jpeg"  # Update path as needed

image_tensor = preprocess_image(image_path)

print("ResNet101 Predictions:")
for label, prob in predict(resnet101, image_tensor):
    print(f"{label}: {prob * 100:.2f}%")

show_image(image_path)
