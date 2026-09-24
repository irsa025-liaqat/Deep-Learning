import keras
from keras import layers
import numpy as np

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize image data to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode the labels
y_train_cat = keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=10)


optimizers = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': keras.optimizers.Adam(learning_rate=0.001)
}

optimizer_results = {}

for opt_name, opt in optimizers.items():
    print(f"\nTraining with optimizer: {opt_name}")
    
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train_cat,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    optimizer_results[opt_name] = test_acc
    print(f"Test Accuracy: {test_acc:.4f}")
