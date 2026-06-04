from sklearn.datasets import load_digits                                    # dataset loader from scikit-learn (1797 digits)
import torch                                                                # standard PyTorch machine learning library
import torch.nn.functional as F                                             # builtin functions including relu and cross_entropy

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')   # leverage CUDA if present

# --- Data ---
X, y = load_digits(return_X_y=True)                                        # load 8x8 handwritten digit dataset (1797 samples)
X = (torch.tensor(X, dtype=torch.float32) - X.mean()) / X.std()            # convert to float32 and standardize (mean=0, std=1)
y = torch.tensor(y, dtype=torch.long)                                      # convert labels to long integer tensors

# --- Model ---
torch.manual_seed(42)                                                       # seed random number generator for reproducibility
W1 = torch.randn(64, 32, requires_grad=True) * 0.1                         # hidden weight matrix: 64 inputs to 32 hidden units
b1 = torch.zeros(1, 32, requires_grad=True)                                # hidden bias vector: one per hidden unit, broadcastable
W2 = torch.randn(32, 10, requires_grad=True) * 0.1                         # output weight matrix: 32 hidden units to 10 class outputs
b2 = torch.zeros(1, 10, requires_grad=True)                                # output bias vector: one per class, broadcastable
lr = 0.1                                                                    # scaling factor for gradient updates

# --- Train ---
params = [W1, b1, W2, b2]                                                  # collect all parameters for concise update loop
for epoch in range(1000):                                                   # train for 1000 epochs (full batch iterations)
    logits = F.relu(X @ W1 + b1) @ W2 + b2                                 # forward pass: ReLU hidden layer (64->32) then logits (32->10)
    loss = F.cross_entropy(logits, y)                                       # cross-entropy loss (numerically stable log-softmax + NLL)
    loss.backward()                                                         # autograd: compute all gradients automatically

    with torch.no_grad():                                                   # disable grad tracking during parameter update
        for p in params: p -= lr * p.grad; p.grad.zero_()                  # gradient descent + clear gradients for all parameters

    if epoch % 100 == 0:                                                    # evaluate and report progress every 100 epochs
        acc = (logits.argmax(1) == y).float().mean()                        # compute percentage of correctly predicted digits
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")                         # output current epoch and accuracy value