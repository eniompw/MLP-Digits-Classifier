# mnist-mlp-from-scratch

A from-scratch **Multi-Layer Perceptron (MLP)** built with NumPy only, trained to classify handwritten digits (0–9) using scikit-learn's built-in `digits` dataset.

No deep learning frameworks — just NumPy and a clear implementation of forward pass, backpropagation, and gradient descent.

## Network Architecture

```
Input (64)  →  Hidden ReLU (32)  →  Output Softmax (10)
```

| Layer | Size | Activation |
|-------|------|------------|
| Input | 64 (8×8 pixel values) | — |
| Hidden | 32 neurons | ReLU |
| Output | 10 neurons (digits 0–9) | Softmax |

## How It Works

1. **Data** — Loads `sklearn.datasets.load_digits` (1797 samples, 64 features each); normalises features to zero mean and unit variance.
2. **One-hot encoding** — Converts integer labels to a `(1797, 10)` target matrix.
3. **Forward pass** — Computes ReLU activations in the hidden layer, then softmax probabilities at the output.
4. **Backward pass** — Derives gradients analytically using the cross-entropy + softmax combined gradient, then propagates through the ReLU gate.
5. **Update** — Applies vanilla gradient descent (`lr = 0.1`) for 1000 epochs, printing accuracy every 100 epochs.

## Key Implementation Details

- **Numerically stable softmax** — subtracts `x.max(axis=1)` (per-row max) before `exp` to prevent overflow.
- **Cross-entropy gradient** — `(predictions - targets) / N` gives the combined loss gradient directly.
- **ReLU gate** — `layer1 > 0` masks the backpropagated error, zeroing gradients where the hidden unit was inactive.
- **Weight initialisation** — `randn * 0.1` keeps initial activations in a sensible range; biases start at zero.

## Files

| File | Description |
|------|-------------|
| `np_mlp_digits.py` | Clean Python script — full training loop |
| `np_mlp_digits.ipynb` | Jupyter Notebook version with cell-by-cell walkthrough |

## Requirements

```
numpy
scikit-learn
```

Install with:

```bash
pip install numpy scikit-learn
```

## Usage

```bash
python np_mlp_digits.py
```

Expected output:

```
Epoch    0 | Acc: 10%
Epoch  100 | Acc: 72%
...
Epoch  900 | Acc: 99%
```

## License

MIT
