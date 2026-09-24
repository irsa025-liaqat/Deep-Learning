import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """Sigmoid activation function: sigma(z) = 1 / (1 + e^(-z))"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid: sigma'(z) = a * (1 - a)"""
    return a * (1.0 - a)

print("=" * 60)
print("PART 1: EXACT STEP-BY-STEP CALCULATION FROM MANUAL")
print("=" * 60)


X = np.array([[0.5], 
              [0.1]])

W1 = np.array([[0.4, 0.1], 
               [0.3, 0.2]])
b1 = np.array([[0.1], 
               [0.2]])

W2 = np.array([[0.3, 0.5]])
b2 = 0.1

y = 1.0  


Z1 = np.dot(W1, X) + b1
A1 = sigmoid(Z1)

Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)


loss = - (y * np.log(A2) + (1.0 - y) * np.log(1.0 - A2))

print(f"Z1:\n{Z1}")
print(f"A1:\n{A1}")
print(f"Z2: {Z2[0, 0]:.4f}")
print(f"A2: {A2[0, 0]:.4f}")
print(f"Loss: {loss[0, 0]:.4f}\n")


dL_dA2 = - (y / A2) + ((1.0 - y) / (1.0 - A2))
dA2_dZ2 = sigmoid_derivative(A2)
dL_dZ2 = dL_dA2 * dA2_dZ2
dL_dW2 = dL_dZ2 * A1.T
dL_db2 = dL_dZ2

dL_dA1 = np.dot(W2.T, dL_dZ2)
dA1_dZ1 = sigmoid_derivative(A1)
dL_dZ1 = dL_dA1 * dA1_dZ1
dL_dW1 = np.dot(dL_dZ1, X.T)
dL_db1 = dL_dZ1

print(f"dL/dZ2 (Output delta): {dL_dZ2[0, 0]:.4f}")
print(f"dL/dW2:\n{dL_dW2}")
print(f"dL/db2: {dL_db2[0, 0]:.4f}")
print(f"dL/dZ1 (Hidden delta):\n{dL_dZ1}")
print(f"dL/dW1:\n{dL_dW1}")
print(f"dL/db1:\n{dL_db1}\n")



class ConfigurableMLP:
    """
    Generalized implementation of the manual's neural network structure.
    Allows modifying layer count, neuron count, and learning rate.
    """
    def __init__(self, layer_dims, lr=0.5):
        self.layer_dims = layer_dims  
        self.lr = lr
        self.weights = []
        self.biases = []
        
        np.random.seed(42)
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

    def train(self, X, y, epochs=1500):
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

configurations = {
    "Manual Baseline: 1 Hidden Layer, 2 Neurons [2, 2, 1]": [2, 2, 1],
    "Wider: 1 Hidden Layer, 8 Neurons [2, 8, 1]": [2, 8, 1],
    "Deeper: 2 Hidden Layers, 4 Neurons each [2, 4, 4, 1]": [2, 4, 4, 1],
    "Deep & Wide: 3 Hidden Layers, 6 Neurons each [2, 6, 6, 6, 1]": [2, 6, 6, 6, 1]
}

print("=" * 60)
print("PART 2: ARCHITECTURE EXPERIMENTS & LEARNING CURVE COMPARISON")
print("=" * 60)

plt.figure(figsize=(9, 6))

for name, dims in configurations.items():
    model = ConfigurableMLP(layer_dims=dims, lr=1.0)
    loss_history = model.train(X_train, y_train, epochs=2000)
    
    
    preds = (model.forward(X_train) >= 0.5).astype(float)
    acc = np.mean(preds == y_train) * 100
    
    print(f"{name:<55} | Final Loss: {loss_history[-1]:.4f} | Accuracy: {acc:.1f}%")
    plt.plot(loss_history, label=name)

plt.title("Learning Curves for Different Hidden Layer Architectures")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()
