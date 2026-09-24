# Import necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting and visualization
from tensorflow.keras.datasets import mnist  # To import the MNIST dataset
from tensorflow.keras.models import Sequential  # To create a sequential neural network model
from tensorflow.keras.layers import Dense, Flatten  # For dense (fully connected) and flatten layers
from tensorflow.keras.utils import to_categorical  # For one-hot encoding of labels
from tensorflow.keras.optimizers import Adam  # For the Adam optimizer
# Step 1: Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # Load the dataset

# Normalize the pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0  # Scale training data
X_test = X_test.astype('float32') / 255.0  # Scale test data

# One-hot encode the labels (0-9) for categorical classification
y_train = to_categorical(y_train, 10)  # Encode training labels
y_test = to_categorical(y_test, 10)  # Encode test labels
# Step 2: Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to a 1D vector of 784 features
    Dense(128, activation='relu'),   # Hidden layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'),    # Hidden layer with 64 neurons and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit) and softmax activation
])

# Print the model summary to understand the architecture
model.summary()
# Step 3: Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")  # Print the test accuracy
# Step 6: Visualize training and validation loss and accuracy
plt.figure(figsize=(12, 5))  # Create a figure with a specific size

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)  # Create a subplot for accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')  # Plot training accuracy
plt.plot(history.history['val_accuracy'], label='Val Accuracy')  # Plot validation accuracy
plt.title('Model Accuracy')  # Set the title
plt.xlabel('Epochs')  # Set x-axis label
plt.ylabel('Accuracy')  # Set y-axis label
plt.legend()  # Show legend
plt.grid()  # Add grid lines

# Plot training & validation loss values
plt.subplot(1, 2, 2)  # Create a subplot for loss
plt.plot(history.history['loss'], label='Train Loss')  # Plot training loss
plt.plot(history.history['val_loss'], label='Val Loss')  # Plot validation loss
plt.title('Model Loss')  # Set the title
plt.xlabel('Epochs')  # Set x-axis label
plt.ylabel('Loss')  # Set y-axis label
plt.legend()  # Show legend
plt.grid()  # Add grid lines

plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()  # Display the plots
# Step 7: Make predictions on the test set
predictions = model.predict(X_test)

# Display the first test image and its predicted label
plt.figure(figsize=(5, 5))  # Create a figure for the image
plt.imshow(X_test[0], cmap='gray')  # Show the first test image in grayscale
plt.title(f"True Label: {np.argmax(y_test[0])}, Predicted: {np.argmax(predictions[0])}")  # Display true and predicted labels
plt.axis('off')  # Hide the axis
plt.show()  # Display the image

# Display a grid of images with true and predicted labels
num_images = 9  # Number of images to display
plt.figure(figsize=(10, 10))  # Create a figure for the grid of images
for i in range(num_images):
    plt.subplot(3, 3, i + 1)  # Create a 3x3 grid of subplots
    plt.imshow(X_test[i], cmap='gray')  # Show each test image in grayscale
    plt.title(f"True: {np.argmax(y_test[i])}, Predicted: {np.argmax(predictions[i])}")  # Display true and predicted labels
    plt.axis('off')  # Hide the axis
plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()  # Display the grid of images



