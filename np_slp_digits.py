from sklearn.datasets import load_digits
import numpy as np

def softmax(x):
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))    # subtract max for numerical stability
    return exp_x / exp_x.sum(axis=1, keepdims=True)

X, y = load_digits(return_X_y=True)
X = (X - X.mean()) / (X.std())                          # standardize each input feature (pixel)
y_one_hot = np.eye(10)[y]                               # one-hot encode labels

np.random.seed(42)
W = np.random.randn(64, 10) * 0.1                       # weights: 64 inputs → 10 classes
b = np.zeros((1, 10))
learning_rate = 0.1

for epoch in range(1000):
    probs = softmax(X @ W + b)                          # forward pass
    dlogits = (probs - y_one_hot) / len(X)              # cross-entropy + softmax gradient
    W -= learning_rate * X.T @ dlogits                  # update weights
    b -= learning_rate * dlogits.sum(0, keepdims=True)  # update biases
    if epoch % 100 == 0:
        acc = np.mean(probs.argmax(1) == y)
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")