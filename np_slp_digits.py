from sklearn.datasets import load_digits
import numpy as np


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)          # subtract row max for stability
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


X, y = load_digits(return_X_y=True)              # X: features, y: labels
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # standardise features
y_one_hot = np.eye(10)[y]                        # one-hot encode labels

np.random.seed(42)
W = np.random.randn(64, 10) * 0.1               # weights: 64 inputs → 10 classes
b = np.zeros((1, 10))                            # biases: one per class
learning_rate = 0.1
num_epochs = 1000

for epoch in range(num_epochs):
    logits = X @ W + b                           # forward pass: scores
    probs = softmax(logits)                      # convert scores to probabilities

    dlogits = (probs - y_one_hot) / len(X)       # gradient w.r.t. logits
    dW = X.T @ dlogits                           # gradient w.r.t. weights
    db = dlogits.sum(axis=0, keepdims=True)      # gradient w.r.t. biases

    W -= learning_rate * dW                      # update weights
    b -= learning_rate * db                      # update biases

    if epoch % 100 == 0:
        preds = probs.argmax(axis=1)             # predicted class
        acc = (preds == y).mean()                # accuracy
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")