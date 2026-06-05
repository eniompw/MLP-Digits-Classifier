# MLP Digits Classifier — NumPy and PyTorch

A hands-on introduction to neural networks built from scratch — covering both a **Single-Layer Perceptron (SLP)** and a **Multi-Layer Perceptron (MLP)** implemented in NumPy and PyTorch, trained to classify handwritten digits (0–9) using scikit-learn's built-in `digits` dataset (1797 samples of 8×8 images).

Includes NumPy scratch implementations, equivalent PyTorch versions (manual gradients and autograd), and step-by-step explainers.

> Follows on from [LinearRegressionGradientDescent](https://github.com/eniompw/LinearRegressionGradientDescent), extending gradient descent from a single-output linear model to a multi-class neural network.
>
> The [np_slp_digits.py](np_slp_digits.py) implementation builds directly on [linreg_ames.py](https://github.com/eniompw/LinearRegressionGD/blob/main/linreg_ames.py), extending the same gradient descent framework from continuous targets in linear regression to categorical targets in multi-class classification.
>
> This series continues with [TinyLM](https://github.com/eniompw/TinyLM), where the SLP and MLP are applied to text.

## Contents

- [Files](#files)
- [Network Architectures](#network-architectures)
- [How It Works](#how-it-works)
- [Key Implementation Details](#key-implementation-details)
- [Hyperparameters](#hyperparameters)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Files

| File | Description |
|------|-------------|
| [np_slp_digits.py](np_slp_digits.py) | NumPy single-layer (no hidden layer) softmax classifier |
| [np_slp_digits.ipynb](np_slp_digits.ipynb) | Jupyter Notebook version of the single-layer classifier |
| [np_slp_digits_explainer.md](np_slp_digits_explainer.md) | Step-by-step explainer for the NumPy single-layer classifier |
| [torch_slp_digits.py](torch_slp_digits.py) | PyTorch single-layer softmax classifier with manual gradient updates |
| [torch_slp_autograd.py](torch_slp_autograd.py) | PyTorch single-layer classifier using builtin softmax and autograd |
| [torch_slp_digits_explainer.md](torch_slp_digits_explainer.md) | Step-by-step explainer for the PyTorch single-layer classifier |
| [np_mlp_digits.py](np_mlp_digits.py) | NumPy MLP — clean Python script with full training loop |
| [np_mlp_digits.ipynb](np_mlp_digits.ipynb) | Jupyter Notebook version with cell-by-cell walkthrough |
| [torch_mlp_autograd.py](torch_mlp_autograd.py) | PyTorch MLP classifier using autograd |
| [torch_mlp_sequential.py](torch_mlp_sequential.py) | PyTorch MLP classifier using `nn.Sequential` |
| [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) | Detailed code walkthrough covering all implementations and what changes at each step |

## Network Architectures

### Single-Layer Perceptron (SLP)

A minimal baseline — one linear mapping from pixels to class logits. See `np_slp_digits.py`.

```
Input (64)  →  Output Softmax (10)
```

| Layer  | Size                    | Activation |
|--------|-------------------------|------------|
| Input  | 64 (8×8 pixel values)   | —          |
| Output | 10 neurons (digits 0–9) | Softmax    |

### Multi-Layer Perceptron (MLP)

Adds a hidden ReLU layer, enabling the network to learn non-linear features.

```
Input (64)  →  Hidden ReLU (32)  →  Output Softmax (10)
```

| Layer  | Size                    | Activation |
|--------|-------------------------|------------|
| Input  | 64 (8×8 pixel values)   | —          |
| Hidden | 32 neurons              | ReLU       |
| Output | 10 neurons (digits 0–9) | Softmax    |

## How It Works

1. **Data** — Loads `sklearn.datasets.load_digits` (1797 samples, 64 features each); applies per-feature z-score standardisation (`mean=0, std=1`) with a small epsilon to avoid division by zero on constant pixels.
2. **One-hot encoding** — Converts integer labels to a `(1797, 10)` target matrix via `np.eye(10)[y]`.
3. **Forward pass** — Computes ReLU activations in the hidden layer, then numerically stable softmax probabilities at the output.
4. **Backward pass** — Uses the combined cross-entropy + softmax gradient `(predictions - targets) / N`, then propagates through the ReLU gate using a binary mask (`layer1 > 0`).
5. **Update** — Applies full-batch gradient descent for 1000 epochs, printing accuracy every 100 epochs.

## Key Implementation Details

| Detail | Description |
|--------|-------------|
| Numerically stable softmax | Subtracts per-row max before `exp` to prevent overflow |
| Combined CE + softmax gradient | `(probs - targets) / N` — avoids computing loss explicitly |
| ReLU gate | `layer1 > 0` binary mask zeros gradients for inactive units |
| Weight initialisation | `randn * 0.1` — small random weights; biases initialised to zero |
| Reproducibility | `np.random.seed(42)` |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.1 |
| Epochs | 1000 |
| Batch size | Full batch (1797 samples) |
| Hidden units | 32 |

## Requirements

```
numpy
scikit-learn
torch
```

Install with:

```bash
pip install numpy scikit-learn torch
```

## Usage

```bash
python np_slp_digits.py          # single-layer (NumPy)
python torch_slp_digits.py       # single-layer (PyTorch, manual gradients)
python torch_slp_autograd.py     # single-layer (PyTorch, autograd)
python np_mlp_digits.py          # multi-layer (NumPy)
python torch_mlp_autograd.py     # multi-layer (PyTorch, autograd)
python torch_mlp_sequential.py   # multi-layer (PyTorch, nn.Sequential)
```

| Epoch | SLP (`np_slp_digits.py`) | MLP (`np_mlp_digits.py`) |
|-------|--------------------------|--------------------------| 
| 0     | 13%                      | 10%                      |
| 100   | 94%                      | 72%                      |
| 200   | 96%                      | 88%                      |
| 300   | 97%                      | 94%                      |
| 400   | 97%                      | 96%                      |
| 500   | 98%                      | 97%                      |
| 600   | 98%                      | 98%                      |
| 700   | 98%                      | 98%                      |
| 800   | 98%                      | 99%                      |
| 900   | 98%                      | 99%                      |

## License

MIT
