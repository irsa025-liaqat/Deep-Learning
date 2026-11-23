import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Create synthetic images and masks
def create_sample_data(img_size=128, num_samples=100):
    images = []
    masks = []
    for _ in range(num_samples):
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        mask = np.zeros((img_size, img_size, 1), dtype=np.uint8)
        # Add a random rectangle
        x1, y1, x2, y2 = np.random.randint(0, img_size // 2, 4)
        x2, y2 = x2 + img_size // 2, y2 + img_size // 2
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (1), -1)
        images.append(img)
        masks.append(mask)
    return np.array(images) / 255.0, np.array(masks)  # Normalize to [0, 1]

# Generate synthetic data
images, masks = create_sample_data()
def unet(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return Model(inputs, outputs)

# Build the model
model = unet()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Split data into training and validation sets
train_images, val_images = images[:80], images[80:]
train_masks, val_masks = masks[:80], masks[80:]

# Train the U-Net model
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    batch_size=8,
    epochs=20
)
# Evaluate the model on validation data
loss, accuracy = model.evaluate(val_images, val_masks)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
# Predict on a few validation samples
pred_masks = model.predict(val_images)

# Plot the results
def plot_sample(image, true_mask, pred_mask):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(true_mask.squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow((pred_mask.squeeze() > 0.5).astype(np.uint8), cmap='gray')

# Show a sample result
for i in range(3):  # Show 3 examples
    plot_sample(val_images[i], val_masks[i], pred_masks[i])
    plt.show()
