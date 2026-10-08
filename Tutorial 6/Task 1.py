
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
     

# Function to load and preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

        # Normalize using ImageNet stats
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)

    # Add batch dimension → (1, 3, 224, 224)
    image = image.unsqueeze(0)

    return image
     
# Load models
vgg16 = models.vgg16(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

# Set to evaluation mode
vgg16.eval()
resnet50.eval()


# Load ImageNet labels
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Prediction function
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)

        # Convert to probabilities
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        # Top 5 predictions
        top5_prob, top5_catid = torch.topk(probs, 5)

        results = []
        for i in range(5):
            results.append((labels[top5_catid[i]], top5_prob[i].item()))

    return results
     

import requests

# URL for ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Download the file
response = requests.get(url)
with open("imagenet_classes.txt", "w") as f:
    f.write(response.text)

print("imagenet_classes.txt downloaded successfully.")
     


image_path = "/content/WhatsApp Image 2026-05-02 at 8.08.58 PM.jpeg"  # change this

image = preprocess_image(image_path)
     



print("VGG16 Predictions:")
for label, prob in predict(vgg16, image):
    print(f"{label}: {prob:.4f}")

print("\nResNet50 Predictions:")
for label, prob in predict(resnet50, image):
    print(f"{label}: {prob:.4f}")

def show_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

show_image(image_path)
     
