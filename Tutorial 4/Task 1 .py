## TASK 1
######################### IN Py torch ############################
import os
from PIL import Image
import torchvision.transforms as transforms

output_folder = "augmented_images_pytorch"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

transform = transforms.Compose([
    transforms.RandomRotation(50),            
    transforms.RandomAffine(degrees=2, shear=30),  
    transforms.RandomResizedCrop(size=256, scale=(0.6, 1.8)),  
    transforms.RandomHorizontalFlip(),        
    transforms.ColorJitter(brightness=0.8),  
])

image_path = "/content/WhatsApp Image 2026-05-02 at 8.08.58 PM.jpeg"   #  I replace the image from my own laptop
image = Image.open(image_path).convert("RGB")

for i in range(40):
    augmented_image = transform(image)
    save_path = os.path.join(output_folder, f"aug_{i+1}.jpeg")
    augmented_image.save(save_path)

print("40 augmented images have been saved in:", output_folder)
