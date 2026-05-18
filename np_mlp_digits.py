from sklearn.datasets import load_digits
import numpy as np

def softmax(x):
    e_x = np.exp(x - x.max(axis=1, keepdims=True))  # numerically stable exp
    return e_x / e_x.sum(axis=1, keepdims=True)      # normalise each row

def forward(raw_image):
    img = np.array(raw_image).reshape(1, 64)
    x_norm = (img - x_mean) / x_std
    layer1 = np.maximum(0, x_norm @ W1 + b1)
    probs = softmax(layer1 @ W2 + b2)
    return layer1, probs

# x: input features (1797 samples, 64 pixels each)
# y: digit labels 0–9 (1797,)
x, y = load_digits(return_X_y=True)

x_mean, x_std = x.mean(), x.std()
x = (x - x_mean) / x_std  # normalise for stable gradient descent (1797, 64)
targets = np.eye(10)[y]        # one-hot encode labels (1797, 10)

# Network: 64 -> 32 -> 10
np.random.seed(42)
W1, b1 = np.random.randn(64, 32) * 0.1, np.zeros((1, 32))  # hidden layer weights/bias
W2, b2 = np.random.randn(32, 10) * 0.1, np.zeros((1, 10))  # output layer weights/bias
lr = 0.1

for epoch in range(1000):
    # Forward pass
    layer1 = np.maximum(0, x @ W1 + b1)     # ReLU hidden layer (1797, 32)
    predictions = softmax(layer1 @ W2 + b2)  # output probabilities (1797, 10)

    # Backward pass — cross-entropy + softmax gradient
    output_error = (predictions - targets) / len(x)      # dL/dZ2
    layer1_error = (output_error @ W2.T) * (layer1 > 0)  # dL/dZ1 (ReLU gate)

    # Gradient descent update
    W2 -= lr * layer1.T @ output_error
    b2 -= lr * output_error.sum(0, keepdims=True)
    W1 -= lr * x.T @ layer1_error
    b1 -= lr * layer1_error.sum(0, keepdims=True)

    if epoch % 100 == 0:
        acc = np.mean(predictions.argmax(1) == y)
        print(f"Epoch {epoch:4d} | Acc: {acc:.0%}")
