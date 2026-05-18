from sklearn.datasets import load_digits
import numpy as np

# Softmax activation for output probabilities
def softmax(x):
    e_x = np.exp(x - x.max(axis=1, keepdims=True))  # numerically stable exp
    return e_x / e_x.sum(axis=1, keepdims=True)     # normalise each row

# Forward pass through network
def forward(raw_image):
    img = np.array(raw_image).reshape(1, 64)
    x_norm = (img - x.mean()) / x.std()
    layer1 = np.maximum(0, x_norm @ W1 + b1)
    probs = softmax(layer1 @ W2 + b2)
    return layer1, probs

# Load and preprocess data
x, y = load_digits(return_X_y=True)           # 1797 samples, 64 pixels each
x = (x - x.mean()) / x.std()                  # normalize inputs
targets = np.eye(10)[y]                       # one-hot encode labels (1797, 10)

# Initialize network: 64 -> 32 -> 10
np.random.seed(42)
W1, b1 = np.random.randn(64, 32) * 0.1, np.zeros((1, 32))  # hidden layer
W2, b2 = np.random.randn(32, 10) * 0.1, np.zeros((1, 10))  # output layer
lr = 0.1

# Training loop with SGD
for epoch in range(1000):
    # Forward pass
    layer1 = np.maximum(0, x @ W1 + b1)             # ReLU hidden layer
    predictions = softmax(layer1 @ W2 + b2)         # output probabilities

    # Backward pass: cross-entropy + softmax gradient
    output_error = (predictions - targets) / len(x)      # dL/dZ2
    layer1_error = (output_error @ W2.T) * (layer1 > 0)  # dL/dZ1 (ReLU gate)

    # Update weights via gradient descent
    W2 -= lr * layer1.T @ output_error
    b2 -= lr * output_error.sum(0, keepdims=True)
    W1 -= lr * x.T @ layer1_error
    b1 -= lr * layer1_error.sum(0, keepdims=True)

    if epoch % 100 == 0:
        acc = np.mean(predictions.argmax(1) == y)
        print(f"Epoch {epoch:4d} | Acc: {acc:.0%}")