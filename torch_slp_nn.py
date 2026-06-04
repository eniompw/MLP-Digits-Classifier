from sklearn.datasets import load_digits                                    # dataset loader from scikit-learn (1797 digits)
import torch                                                                # standard PyTorch machine learning library
import torch.nn as nn                                                       # neural network module including Linear layers
import torch.nn.functional as F                                             # builtin functions including cross_entropy

# Automatically create all tensors on GPU if available, removing manual device boilerplate
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')   # leverage CUDA if present

# --- Data ---
X, y = load_digits(return_X_y=True)                                        # load 8x8 handwritten digit dataset (1797 samples)
X = (torch.tensor(X, dtype=torch.float32) - X.mean()) / X.std()            # convert to float32 and standardize (mean=0, std=1)
y = torch.tensor(y, dtype=torch.long)                                      # convert labels to long integer tensors

# --- Model ---
torch.manual_seed(42)                                                       # seed random number generator for reproducibility
model = nn.Sequential(nn.Linear(64, 10))                                   # single linear layer: 64 inputs to 10 class outputs
lr = 0.1                                                                    # scaling factor for gradient updates

# --- Train ---
for epoch in range(1000):                                                   # train for 1000 epochs (full batch iterations)
    logits = model(X)                                                       # forward pass through the sequential model
    loss = F.cross_entropy(logits, y)                                       # cross-entropy loss (numerically stable log-softmax + NLL)
    loss.backward()                                                         # autograd: compute all gradients automatically

    with torch.no_grad():                                                   # disable grad tracking during parameter update
        for p in model.parameters():                                        # iterate over all learnable parameters (W and b)
            p -= lr * p.grad; p.grad.zero_()                                # gradient descent + clear gradients

    if epoch % 100 == 0:                                                    # evaluate and report progress every 100 epochs
        acc = (logits.argmax(1) == y).float().mean()                        # compute percentage of correctly predicted digits
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")                         # output current epoch and accuracy value
