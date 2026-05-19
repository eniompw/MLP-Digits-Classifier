from sklearn.datasets import load_digits
import numpy as np

def softmax(x):
    e_x = np.exp(x - x.max(axis=1, keepdims=True))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=1, keepdims=True)

x, y = load_digits(return_X_y=True)
x = (x - x.mean(0)) / (x.std(0) + 1e-8)  # standardise: zero mean, unit variance
targets = np.eye(10)[y]                    # one-hot encode labels

np.random.seed(42)
W = np.random.randn(64, 10) * 0.1  # weights: 64 inputs → 10 classes
b = np.zeros((1, 10))
lr = 0.1

for epoch in range(1000):
    probs = softmax(x @ W + b)               # forward pass
    error = (probs - targets) / len(x)       # cross-entropy + softmax gradient
    W -= lr * x.T @ error                    # update weights
    b -= lr * error.sum(0, keepdims=True)    # update biases
    if epoch % 100 == 0:
        acc = np.mean(probs.argmax(1) == y)
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")