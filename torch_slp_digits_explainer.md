# torch_slp_digits.py - Step-by-Step Explainer

This script is the PyTorch version of the single-layer perceptron (SLP) digits classifier. It keeps the same manual softmax and gradient math as the NumPy version, but rewrites the tensor operations in PyTorch and adds automatic GPU selection when available.

The important idea is that the model structure did not change. What changed is the library used for tensor creation, matrix math, and a few PyTorch-specific details around dtype, broadcasting, and reductions.

## 1. What the script does

The training flow is still the same as the NumPy baseline:

1. Load the handwritten digits dataset
2. Standardize the input features
3. One-hot encode the labels
4. Initialize weights and biases
5. Run full-batch gradient descent with softmax probabilities
6. Print accuracy every 100 epochs

The goal is to show that the exact same learning algorithm can be expressed in PyTorch without introducing `torch.nn` or autograd.

## 2. Imports and device setup

```python
from sklearn.datasets import load_digits
import torch

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
```

The dataset still comes from scikit-learn, but all tensors are created in PyTorch.

The new line here is `torch.set_default_device(...)`.

- If CUDA is available, tensors are created on the GPU automatically.
- If not, the code falls back to CPU without any extra branching in the training loop.

This keeps the example compact and avoids manual `.to(device)` calls throughout the file.

## 3. Data section

```python
X, y = load_digits(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X = (X - X.mean()) / X.std()
y_one_hot = torch.eye(10)[y]
```

### NumPy to Torch changes

These are the direct equivalents of the NumPy version:

- `np.array` becomes `torch.tensor`
- `X` is explicitly `float32`
- `y` is explicitly `long`
- `np.eye(10)[y]` becomes `torch.eye(10)[y]`

The dtype choice matters in PyTorch:

- `X` must be floating point for matrix multiplication and normalization.
- `y` is kept as integer labels, which is the standard class-label dtype in PyTorch.

The standardization step is unchanged mathematically:

$$
X = \frac{X - \mu}{\sigma}
$$

Here the mean and standard deviation are computed over the whole dataset, just like the NumPy script.

## 4. Model section

```python
torch.manual_seed(42)
W = torch.randn(64, 10) * 0.1
b = torch.zeros((1, 10))
learning_rate = 0.1
```

### What changed from NumPy

- `np.random.seed(42)` becomes `torch.manual_seed(42)`
- `np.random.randn` becomes `torch.randn`
- `np.zeros` becomes `torch.zeros`

The weight matrix `W` still has shape `(64, 10)` because there are 64 input features and 10 digit classes.

The `* 0.1` scaling is intentional:

- It keeps the initial logits small.
- Small logits reduce the chance of early softmax saturation.
- That makes optimization smoother at the start of training.

The bias vector `b` has shape `(1, 10)`, which lets it broadcast cleanly across all 1797 samples during the forward pass.

## 5. Training loop

```python
for epoch in range(1000):
    # Forward pass
    exp_x = torch.exp(X @ W + b)
    probs = exp_x / exp_x.sum(dim=1, keepdim=True)

    # Backward pass
    dlogits = (probs - y_one_hot) / len(X)
    W -= learning_rate * X.T @ dlogits
    b -= learning_rate * dlogits.sum(0, keepdim=True)

    if epoch % 100 == 0:
        acc = (probs.argmax(1) == y).float().mean()
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")
```

### 5.1 Forward pass

The first line computes the raw class scores, or logits:

$$
\text{logits} = XW + b
$$

Shape-wise, that is:

$$
(1797, 64) @ (64, 10) \rightarrow (1797, 10)
$$

The softmax is written manually instead of using a library helper.

The Torch version keeps the same two-step structure as the NumPy version, but uses `dim=` and `keepdim=` instead of `axis=` and `keepdims=`:

- `exp_x.sum(dim=1, keepdim=True)` preserves the `(1797, 1)` shape needed for broadcasting.
- The division then normalizes each row into a probability distribution.

In a numerically stable PyTorch softmax, the row-wise maximum is commonly accessed with `x.max(dim=1, keepdim=True).values` before exponentiating. This particular script keeps the implementation direct and mirrors the manual math used in the NumPy version, so the stability step is not in the code itself.

### 5.2 Backward pass

The gradient is still the simplified softmax + cross-entropy derivative:

$$
\frac{\partial L}{\partial z} = \frac{p - y}{N}
$$

That is exactly what this line computes:

```python
dlogits = (probs - y_one_hot) / len(X)
```

This is the key reason the example stays compact. There is no autograd graph and no `torch.nn.CrossEntropyLoss`; the update rule is written directly.

Then the parameter gradients are applied manually:

- `X.T @ dlogits` gives the weight gradient for `W`
- `dlogits.sum(0, keepdim=True)` gives the bias gradient for `b`

The Torch version also mirrors the NumPy version’s use of full-batch gradient descent, so every epoch sees the entire dataset at once.

### 5.3 Accuracy reporting

```python
acc = (probs.argmax(1) == y).float().mean()
```

This line has one important PyTorch-specific change from NumPy:

- `np.mean(probs.argmax(1) == y)` becomes `(probs.argmax(1) == y).float().mean()`

That extra `.float()` is required because the comparison produces a boolean tensor, and PyTorch does not compute the mean directly on booleans in the same way NumPy does.

The prediction logic itself is unchanged:

- `probs.argmax(1)` picks the most likely class for each sample.
- Comparing it with `y` produces a correct/incorrect mask.
- Taking the mean gives classification accuracy.

## 6. Torch vs NumPy: the practical differences

The PyTorch version is intentionally close to the NumPy script. The changes are mostly about API shape and tensor semantics, not algorithm design.

| NumPy version | Torch version | Why it changed |
|---|---|---|
| `np.array(...)` | `torch.tensor(...)` | Create PyTorch tensors directly |
| `np.random.seed(42)` | `torch.manual_seed(42)` | Reproducible initial weights |
| `np.random.randn(...)` | `torch.randn(...)` | Parameter initialization |
| `np.zeros(...)` | `torch.zeros(...)` | Bias initialization |
| `axis=` | `dim=` | PyTorch reduction syntax |
| `keepdims=` | `keepdim=` | PyTorch reduction syntax |
| `x.max(axis=1, keepdims=True)` | `x.max(dim=1, keepdim=True).values` | PyTorch returns a named tuple when you use the stable-softmax pattern |
| `np.mean(...)` on booleans | `.float().mean()` | Boolean tensors need casting before averaging |

## 7. Why this version is useful

This file is a teaching bridge between NumPy and PyTorch:

- It shows that the math is identical across both libraries.
- It introduces PyTorch tensor creation and broadcasting without adding framework abstractions.
- It demonstrates how to move a simple model to GPU with almost no code changes.

If you read `np_slp_digits.py` first, this Torch version should feel familiar. The main difference is that the same operations are now expressed in PyTorch idioms, with comments and formatting adjusted to match the TorchMLP style used elsewhere in the repository.