from sklearn.datasets import load_digits
import numpy as np

# Standardize each input feature (pixel).
def normalize(x):
    # Add a tiny epsilon to avoid divide-by-zero for constant features.
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

# One forward pass: input -> ReLU hidden layer -> softmax output.
def forward(x_in):
    layer1 = np.maximum(0, x_in @ W1 + b1)  # ReLU hidden layer
    probs = softmax(layer1 @ W2 + b2)  # class probabilities
    return layer1, probs

# Softmax across classes for each sample.
def softmax(x):
    e_x = np.exp(x - x.max(axis=1, keepdims=True))  # numerically stable exp
    return e_x / e_x.sum(axis=1, keepdims=True)

# Load and preprocess data.
x, y = load_digits(return_X_y=True)  # 1797 samples, 64 features
x = normalize(x)
targets = np.eye(10)[y]  # one-hot labels, shape: (1797, 10)

# Initialize network: 64 -> 32 -> 10
np.random.seed(42)
W1, b1 = np.random.randn(64, 32) * 0.1, np.zeros((1, 32))  # hidden layer
W2, b2 = np.random.randn(32, 10) * 0.1, np.zeros((1, 10))  # output layer
lr = 0.1

# Training loop (full-batch gradient descent).
for epoch in range(1000):
    # Forward pass.
    layer1, probs = forward(x)

    # Backward pass: gradient of cross-entropy with softmax.
    output_error = (probs - targets) / len(x)
    layer1_error = (output_error @ W2.T) * (layer1 > 0)  # dL/dZ1 (ReLU gate)

    # Parameter update.
    W2 -= lr * layer1.T @ output_error
    b2 -= lr * output_error.sum(0, keepdims=True)
    W1 -= lr * x.T @ layer1_error
    b1 -= lr * layer1_error.sum(0, keepdims=True)

    if epoch % 100 == 0:
        acc = np.mean(probs.argmax(1) == y)
        print(f"Epoch {epoch:4d} | Acc: {acc:.0%}")