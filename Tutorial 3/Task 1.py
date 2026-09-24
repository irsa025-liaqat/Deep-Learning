import tensorflow.keras as keras
from tensorflow.keras import layers

architectures = [
    [128, 64],
    [256, 128],
    [256, 128, 64],
    [512, 256, 128]
]

activations = ['relu', 'tanh', 'sigmoid']
results = {}

for arch in architectures:
    for act in activations:
        print(f"\nTraining model with layers {arch} and activation {act}")
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=(28, 28)))

        for units in arch:
            model.add(layers.Dense(units, activation=act))

        model.add(layers.Dense(10, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results[(tuple(arch), act)] = test_acc
        print(f"Test Accuracy: {test_acc:.4f}")
