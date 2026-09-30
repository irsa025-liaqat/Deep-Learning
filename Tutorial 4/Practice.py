
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

output_folder = "augmented_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

datagen = ImageDataGenerator(
    rotation_range=40,        
    shear_range=0.2,          
    zoom_range=0.2,           
    horizontal_flip=True,     
    brightness_range=[0.5, 1.5],  
    fill_mode='nearest'
)

image_path = "/content/WhatsApp Image 2026-05-02 at 8.08.58 PM.jpeg"   # I take the picture of my brother from my own laptop
img = load_img(image_path)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(
    x,
    batch_size=1,
    save_to_dir=output_folder,
    save_prefix='aug',
    save_format='jpeg'
):
    i += 1
    if i >= 40:   # Generate 40 augmented images
        break

print("40 augmented images have been saved in:", output_folder)
