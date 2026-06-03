from sklearn.datasets import load_digits                                                # dataset loader from scikit-learn (1797 digits)
import torch                                                                            # standard PyTorch machine learning library
import torch.nn.functional as F                                                         # builtin functions including softmax and cross_entropy

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')               # leverage CUDA if present

# --- Data ---
X, y = load_digits(return_X_y=True)                                                    # load 8x8 handwritten digit dataset (1797 samples)
X = torch.tensor(X, dtype=torch.float32)                                               # convert inputs to float32 tensors
y = torch.tensor(y, dtype=torch.long)                                                  # convert labels to long integer tensors
X = (X - X.mean()) / X.std()                                                           # standardize input features (mean=0, std=1)

# --- Model ---
torch.manual_seed(42)                                                                  # seed random number generator for reproducibility
W = torch.randn(64, 10, requires_grad=True) * 0.1                                      # weight matrix: 64 inputs to 10 class outputs
b = torch.zeros((1, 10), requires_grad=True)                                           # bias vector: one per class, broadcastable
learning_rate = 0.1                                                                    # scaling factor for gradient updates

# --- Train ---
for epoch in range(1000):                                                              # train for 1000 epochs (full batch iterations)
    # Forward pass
    logits = X @ W + b                                                                 # unnormalized log probabilities (logits)
    probs = F.softmax(logits, dim=1)                                                   # softmax activation to compute probability distribution
    loss = F.cross_entropy(logits, y)                                                  # cross-entropy loss (numerically stable log-softmax + NLL)

    # Backward pass
    loss.backward()                                                                    # autograd: compute all gradients automatically
    with torch.no_grad():                                                              # disable grad tracking during parameter update
        W -= learning_rate * W.grad                                                    # gradient descent step on weight parameter matrix
        b -= learning_rate * b.grad                                                    # gradient descent step on class bias term vector
        W.grad.zero_()                                                                 # clear gradients before next epoch
        b.grad.zero_()                                                                 # clear gradients before next epoch

    if epoch % 100 == 0:                                                               # evaluate and report progress every 100 epochs
        acc = (probs.argmax(1) == y).float().mean()                                    # compute percentage of correctly predicted digits
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")                                    # output current epoch and accuracy value
