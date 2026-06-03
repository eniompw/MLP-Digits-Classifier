from sklearn.datasets import load_digits                                                # dataset loader from scikit-learn (1797 digits)
import numpy as np                                                                      # standard NumPy numerical computing library

def softmax(x):
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))                                   # subtract max for numerical stability (avoids overflow)
    return exp_x / exp_x.sum(axis=1, keepdims=True)                                    # normalize to compute probability distribution

# --- Data ---
X, y = load_digits(return_X_y=True)                                                     # load 8x8 handwritten digit dataset (1797 samples)
X = (X - X.mean()) / (X.std())                                                          # standardize input features (mean=0, std=1)
y_one_hot = np.eye(10)[y]                                                               # one-hot encode labels (shape: 1797x10)

# --- Model ---
np.random.seed(42)                                                                      # seed random number generator for reproducibility
W = np.random.randn(64, 10) * 0.1                                                       # weight matrix: 64 inputs to 10 class outputs
b = np.zeros((1, 10))                                                                   # bias vector: one per class, broadcastable
learning_rate = 0.1                                                                     # scaling factor for gradient updates

# --- Train ---
for epoch in range(1000):                                                               # train for 1000 epochs (full batch iterations)
    # Forward pass
    probs = softmax(X @ W + b)                                                          # softmax activation to compute probability distribution

    # Backward pass
    dlogits = (probs - y_one_hot) / len(X)                                              # local gradient of cross-entropy loss w.r.t logits
    W -= learning_rate * X.T @ dlogits                                                  # gradient descent step on weight parameter matrix
    b -= learning_rate * dlogits.sum(0, keepdims=True)                                  # gradient descent step on class bias term vector

    if epoch % 100 == 0:                                                                # evaluate and report progress every 100 epochs
        acc = np.mean(probs.argmax(1) == y)                                             # compute percentage of correctly predicted digits
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")                                     # output current epoch and accuracy value
