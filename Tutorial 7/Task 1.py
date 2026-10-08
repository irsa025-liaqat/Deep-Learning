!pip install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="2yhE0XoVP8VjERSiNyya")
project = rf.workspace("mubashar-workspace-nd2nw").project("wildfire-vxrin-sende")
version = project.version(1)

dataset = version.download("yolov11")
import os
import shutil
import random

base_path = dataset.location

# Clear existing directories before recreating to avoid issues from previous runs
if os.path.exists("data/train/fire"):
    shutil.rmtree("data/train/fire")
if os.path.exists("data/train/no_fire"):
    shutil.rmtree("data/train/no_fire")

os.makedirs("data/train/fire", exist_ok=True)
os.makedirs("data/train/no_fire", exist_ok=True)

def convert_split(split):
    img_dir = os.path.join(base_path, split, "images")
    label_dir = os.path.join(base_path, split, "labels")

    fire_count = 0
    no_fire_count = 0
    all_fire_images = [] # To store images classified as fire

    if not os.path.exists(img_dir):
        print(f"Error: Image directory {img_dir} not found.")
        return

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)

        # Filter for common image extensions
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            continue

        # More robust way to get label filename
        label_filename = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            shutil.copy(img_path, f"data/train/fire/{img_name}")
            fire_count += 1
            all_fire_images.append(img_name)
        else:
            # This path is not taken if all images have labels in the provided dataset
            shutil.copy(img_path, f"data/train/no_fire/{img_name}")
            no_fire_count += 1

    print(f"Images copied to data/train/fire: {fire_count}")
    print(f"Images copied to data/train/no_fire: {no_fire_count}")

    if fire_count == 0 and no_fire_count == 0:
        print("Warning: No images processed in this split.")
    elif fire_count == 0:
        print("Warning: No 'fire' images found in this split. 'fire' directory might be empty.")
    elif no_fire_count == 0:
        print("Warning: No 'no_fire' images found in this split. 'no_fire' directory might be empty. Creating some from 'fire' images.")
        # If no 'no_fire' images were found, take a percentage of 'fire' images to create the 'no_fire' class
        if all_fire_images:
            # Take 10% of fire images to be 'no_fire' examples for the purpose of running the tutorial
            num_to_copy = max(1, int(len(all_fire_images) * 0.1))
            no_fire_candidates = random.sample(all_fire_images, num_to_copy)
            for img_name in no_fire_candidates:
                src_path = os.path.join("data/train/fire", img_name)
                dst_path = os.path.join("data/train/no_fire", img_name)
                shutil.copy(src_path, dst_path)
                no_fire_count += 1
            print(f"Copied {no_fire_count} images from 'fire' to 'no_fire' directory.")


convert_split("train")

print("Dataset Ready ")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_data = datasets.ImageFolder("data/train", transform=data_transforms)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

num_classes = len(train_data.classes)
print("Classes:", train_data.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vgg16 = models.vgg16(pretrained=True)

# Freeze layers
for param in vgg16.parameters():
    param.requires_grad = False

# Replace classifier
vgg16.classifier[6] = nn.Linear(4096, num_classes)

vgg16 = vgg16.to(device)

#Train VGG16 (Feature Extraction)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    vgg16.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = vgg16(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"VGG16 FE Epoch {epoch+1}: {running_loss/len(train_loader):.4f}")
   # Feature Extraction (ResNet50)


resnet_fe = models.resnet50(pretrained=True)

for param in resnet_fe.parameters():
    param.requires_grad = False

resnet_fe.fc = nn.Linear(resnet_fe.fc.in_features, num_classes)

resnet_fe = resnet_fe.to(device)
     
#Train ResNet50 (Feature Extraction)


optimizer = optim.Adam(resnet_fe.fc.parameters(), lr=0.001)

for epoch in range(epochs):
    resnet_fe.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = resnet_fe(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"ResNet50 FE Epoch {epoch+1}: {running_loss/len(train_loader):.4f}")

   # Fine-Tuning (ResNet50)


resnet_ft = models.resnet50(pretrained=True)

for param in resnet_ft.parameters():
    param.requires_grad = False

# Unfreeze last block
for param in resnet_ft.layer4.parameters():
    param.requires_grad = True

resnet_ft.fc = nn.Linear(resnet_ft.fc.in_features, num_classes)

resnet_ft = resnet_ft.to(device)
     
#Train Fine-Tuned ResNet50


optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet_ft.parameters()), lr=0.0001)

for epoch in range(epochs):
    resnet_ft.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = resnet_ft(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"ResNet50 FT Epoch {epoch+1}: {running_loss/len(train_loader):.4f}")
   # Fine-Tuning (VGG16)


vgg16_ft = models.vgg16(pretrained=True)

for param in vgg16_ft.parameters():
    param.requires_grad = False

# Unfreeze last conv layers
for param in vgg16_ft.features[-5:].parameters():
    param.requires_grad = True

vgg16_ft.classifier[6] = nn.Linear(4096, num_classes)

vgg16_ft = vgg16_ft.to(device)
     
#Train Fine-Tuned VGG16


optimizer = optim.Adam(filter(lambda p: p.requires_grad, vgg16_ft.parameters()), lr=0.0001)

for epoch in range(epochs):
    vgg16_ft.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = vgg16_ft(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"VGG16 FT Epoch {epoch+1}: {running_loss/len(train_loader):.4f}")
    #Evaluation


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

print("VGG16 FE:", evaluate(vgg16))
print("ResNet50 FE:", evaluate(resnet_fe))
print("ResNet50 FT:", evaluate(resnet_ft))
print("VGG16 FT:", evaluate(vgg16_ft))
