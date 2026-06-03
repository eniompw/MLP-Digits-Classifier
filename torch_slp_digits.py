from sklearn.datasets import load_digits
import torch

# Automatically place tensors on GPU if available.
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(x):
    exp_x = torch.exp(x - x.max(dim=1, keepdim=True).values)  # numerically stable exponentials
    return exp_x / exp_x.sum(dim=1, keepdim=True)


# Load and preprocess data.
X, y = load_digits(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X = (X - X.mean()) / X.std()
y_one_hot = torch.eye(10)[y]

# Initialize parameters for a single linear layer: 64 -> 10
torch.manual_seed(42)
W = torch.randn(64, 10) * 0.1
b = torch.zeros((1, 10))
learning_rate = 0.1

# Full-batch gradient descent with manual gradients.
for epoch in range(1000):
    logits = X @ W + b
    probs = softmax(logits)

    dlogits = (probs - y_one_hot) / len(X)
    W -= learning_rate * X.T @ dlogits
    b -= learning_rate * dlogits.sum(0, keepdim=True)

    if epoch % 100 == 0:
        acc = (probs.argmax(1) == y).float().mean()
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")
