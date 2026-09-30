import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values (0–255 → 0–1)
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
model = models.Sequential()

# Convolution + Pooling layers
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

# Flatten + Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))  # 10 classes
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels,
                    epochs=10,
                    validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
def plot_image(i, predictions_array, true_label, img):
    true_label = true_label[i][0]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i])

    predicted_label = np.argmax(predictions_array)
    
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
                                color=color)

plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1,5,i+1)
    plot_image(i, predictions[i], test_labels, test_images)
plt.show()

#################RESULTS################################
 Epoch 1/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 79s 50ms/step - accuracy: 0.4426 - loss: 1.5279 - val_accuracy: 0.5666 - val_loss: 1.2190
Epoch 2/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 82s 52ms/step - accuracy: 0.5914 - loss: 1.1550 - val_accuracy: 0.6171 - val_loss: 1.0929
Epoch 3/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 81s 52ms/step - accuracy: 0.6477 - loss: 1.0062 - val_accuracy: 0.6354 - val_loss: 1.0522
Epoch 4/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 82s 52ms/step - accuracy: 0.6796 - loss: 0.9134 - val_accuracy: 0.6791 - val_loss: 0.9220
Epoch 5/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 85s 54ms/step - accuracy: 0.7069 - loss: 0.8351 - val_accuracy: 0.6904 - val_loss: 0.8894
Epoch 6/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 85s 54ms/step - accuracy: 0.7273 - loss: 0.7803 - val_accuracy: 0.6994 - val_loss: 0.8719
Epoch 7/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 140s 53ms/step - accuracy: 0.7429 - loss: 0.7343 - val_accuracy: 0.6985 - val_loss: 0.8823
Epoch 8/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 75s 48ms/step - accuracy: 0.7563 - loss: 0.6941 - val_accuracy: 0.6982 - val_loss: 0.8884
Epoch 9/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 73s 47ms/step - accuracy: 0.7688 - loss: 0.6541 - val_accuracy: 0.7048 - val_loss: 0.8964
Epoch 10/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 81s 46ms/step - accuracy: 0.7831 - loss: 0.6181 - val_accuracy: 0.7020 - val_loss: 0.9161
313/313 - 4s - 13ms/step - accuracy: 0.7020 - loss: 0.9161

Test accuracy: 0.7020000219345093
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step

