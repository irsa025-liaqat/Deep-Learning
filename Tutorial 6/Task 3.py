!pip install roboflow


from roboflow import Roboflow

rf = Roboflow(api_key="2yhE0XoVP8VjERSiNyya")
project = rf.workspace("mubashar-workspace-nd2nw").project("wildfire-vxrin-sende")
version = project.version(1)

dataset = version.download("yolov11")   # your format
import os
import shutil

base_path = dataset.location

# Create classification folders
os.makedirs("data/train/fire", exist_ok=True)
os.makedirs("data/train/no_fire", exist_ok=True)

# Function to convert YOLO → classification
def convert_split(split):
    img_dir = os.path.join(base_path, split, "images")
    label_dir = os.path.join(base_path, split, "labels")

    # Clear directories before copying to avoid duplicate files if cell is run multiple times
    shutil.rmtree("data/train/fire", ignore_errors=True)
    shutil.rmtree("data/train/no_fire", ignore_errors=True)
    os.makedirs("data/train/fire", exist_ok=True)
    os.makedirs("data/train/no_fire", exist_ok=True)

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        # Assuming label files have the same name as images but with .txt extension
        # and are in the 'labels' directory
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

        # If label file exists and NOT empty → fire
        # Otherwise → no_fire
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            shutil.copy(img_path, f"data/train/fire/{img_name}")
        else:
            shutil.copy(img_path, f"data/train/no_fire/{img_name}")

# Convert train split
convert_split("train")

# --- Added fix: Remove empty class directories ---
# Check and remove empty class directories to prevent ImageFolder errors
if not os.listdir("data/train/fire"): # If 'fire' directory is empty
    os.rmdir("data/train/fire")
    print("Removed empty 'data/train/fire' directory. No 'fire' instances found for training.")
if not os.listdir("data/train/no_fire"): # If 'no_fire' directory is empty
    os.rmdir("data/train/no_fire")
    print("Removed empty 'data/train/no_fire' directory. No 'no_fire' instances found for training.")
# --- End added fix ---

print("Conversion Done ")
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}

train_data = datasets.ImageFolder("data/train", transform=data_transforms['train'])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

print("Classes:", train_data.classes)
from torchvision import models

model = models.resnet50(pretrained=True)

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = len(train_data.classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    import os

# Get a list of files in the 'fire' directory
fire_images = os.listdir("data/train/fire")

# Check if there are any images
if fire_images:
    # Take the first image found as a sample
    sample_image_name = fire_images[0]
    sample_image_path = os.path.join("data/train/fire", sample_image_name)
    print(f"Using sample image: {sample_image_path}")
else:
    sample_image_path = None
    print("No images found in data/train/fire to use as a sample.")

from PIL import Image
import matplotlib.pyplot as plt

def show_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def predict_fire(image_path):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return train_data.classes[pred.item()]

# Example
if sample_image_path:
    print(predict_fire(sample_image_path))
    show_image(sample_image_path)
else:
    print("Cannot predict: No sample image path available.")
