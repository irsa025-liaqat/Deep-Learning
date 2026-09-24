import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """Sigmoid activation function: sigma(z) = 1 / (1 + e^(-z))"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid: sigma'(z) = a * (1 - a)"""
    return a * (1.0 - a)


class ConfigurableMLP:
    """
    Generalized implementation of the manual's neural network structure.
    Allows modifying layer count, neuron count, and learning rate.
    """
    def __init__(self, layer_dims, lr=0.5, seed=42):
        self.layer_dims = layer_dims
        self.lr = lr
        self.weights = []
        self.biases = []
        
        np.random.seed(seed)
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i + 1], layer_dims[i]) * 0.5
            b = np.zeros((layer_dims[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        a = X
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = sigmoid(z)
            self.activations.append(a)
        return a

    def backward(self, y_true):
        m = y_true.shape[1]
        L = len(self.weights)
        
        dZ = self.activations[-1] - y_true
        
        for l in range(L - 1, -1, -1):
            dW = np.dot(dZ, self.activations[l].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 0:
                dA_prev = np.dot(self.weights[l].T, dZ)
                dZ = dA_prev * sigmoid_derivative(self.activations[l])
            
           
            self.weights[l] -= self.lr * dW
            self.biases[l] -= self.lr * db

    def train(self, X, y, epochs=2500):
        losses = []
        for _ in range(epochs):
            a_out = self.forward(X)
            
            eps = 1e-15
            a_clipped = np.clip(a_out, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(a_clipped) + (1.0 - y) * np.log(1.0 - a_clipped))
            losses.append(loss)
            self.backward(y)
        return losses



X_train = np.array([[0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0]])
y_train = np.array([[0.0, 1.0, 1.0, 0.0]])

FIXED_ARCHITECTURE = [2, 4, 1]  
LEARNING_RATES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
EPOCHS = 2500

print("=" * 65)
print("LEARNING RATE EXPERIMENT: EFFECT ON CONVERGENCE")
print("=" * 65)

plt.figure(figsize=(10, 6))

for lr in LEARNING_RATES:

    model = ConfigurableMLP(layer_dims=FIXED_ARCHITECTURE, lr=lr, seed=42)
    loss_history = model.train(X_train, y_train, epochs=EPOCHS)
    
    
    preds = (model.forward(X_train) >= 0.5).astype(float)
    acc = np.mean(preds == y_train) * 100
    
    print(f"Learning Rate: {lr:<6} | Final Loss: {loss_history[-1]:.4f} | Accuracy: {acc:.1f}%")
    plt.plot(loss_history, label=f"lr = {lr}")

plt.title(f"Effect of Learning Rate on Convergence (Architecture: {FIXED_ARCHITECTURE})")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
