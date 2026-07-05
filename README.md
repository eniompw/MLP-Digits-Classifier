# MLP Digits Classifier — NumPy & PyTorch

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eniompw/MLP-Digits-Classifier/blob/main/np_mlp_digits.ipynb)

## What This Repo Is

This is the **middle step** in a three-part series that builds up to a language model from first principles:

| Repo | What you learn | Core concept |
|------|----------------|--------------|
| [LinearRegressionGD](https://github.com/eniompw/LinearRegressionGD) | Predict a single number | 1 neuron, MSE loss, gradient descent |
| **MLP-Digits-Classifier** ← you are here | Classify 10 digit classes | Hidden layer, ReLU, softmax, cross-entropy |
| [TinyLM](https://github.com/eniompw/TinyLM) | Generate text | Same MLP applied to character sequences |

The series is deliberately incremental: every new concept introduced here builds directly on the previous repo. `np_slp_digits.py` is a direct extension of [`linreg_ames.py`](https://github.com/eniompw/LinearRegressionGD/blob/main/linreg_ames.py) — the same gradient descent loop, now with softmax instead of MSE and 10 outputs instead of 1. The MLP then adds a hidden ReLU layer, which is the exact same architecture used in [`NameSLP.py`](https://github.com/eniompw/TinyLM/blob/main/NameSLP.py) and [`TinyMLP.py`](https://github.com/eniompw/TinyLM/blob/main/TinyMLP.py) in TinyLM.

By the end of this repo you will have built the same core network — in pure NumPy and in PyTorch — that powers the first two levels of TinyLM.

| | LinearRegressionGD | MLP-Digits-Classifier | TinyLM |
|---|---|---|---|
| **Architecture** | 1 neuron | SLP → MLP (images) | MLP → Transformer (text) |
| **Loss** | MSE loss | cross-entropy loss | cross-entropy loss |
| **Stack** | NumPy only | NumPy → PyTorch | NumPy → CuPy → PyTorch |

---

## Architecture

SLP: Input (64) → Softmax (10)
MLP: Input (64) → ReLU (32) → Softmax (10)


The MLP reaches **99% accuracy** on scikit-learn's `digits` dataset (1797 samples, 8×8 handwritten digits 0–9). The SLP plateaus at 98%.

---

## Learning Path

Start with `np_slp_digits.py` and follow the progression:

**NumPy — baseline**

| File | Description |
|------|-------------|
| [np_slp_digits.py](np_slp_digits.py) | Single-layer softmax classifier |
| [np_slp_digits.ipynb](np_slp_digits.ipynb) | Notebook version |
| [np_slp_digits_explainer.md](np_slp_digits_explainer.md) | Line-by-line explainer |

**NumPy — adds hidden layer**

| File | Description |
|------|-------------|
| [np_mlp_digits.py](np_mlp_digits.py) | MLP with full training loop |
| [np_mlp_digits.ipynb](np_mlp_digits.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eniompw/MLP-Digits-Classifier/blob/main/np_mlp_digits.ipynb) | Notebook with cell-by-cell walkthrough |

**PyTorch — bridge**

| File | Description |
|------|-------------|
| [torch_slp_digits.py](torch_slp_digits.py) | Manual gradient updates |
| [torch_slp_autograd.py](torch_slp_autograd.py) | Built-in autograd |
| [torch_slp_digits_explainer.md](torch_slp_digits_explainer.md) | Step-by-step explainer |

**PyTorch — destination**

| File | Description |
|------|-------------|
| [torch_mlp_autograd.py](torch_mlp_autograd.py) | MLP with autograd |
| [torch_mlp_sequential.py](torch_mlp_sequential.py) | MLP with `nn.Sequential` |
| [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) | Full walkthrough across all files |

---

## How the MLP Works

Steps describe `np_mlp_digits.py`; the SLP skips steps 3–4.

1. **Data** — `load_digits` (1797 × 64); z-score standardisation per feature
2. **One-hot** — `np.eye(10)[y]` → `(1797, 10)` target matrix
3. **Forward** — ReLU hidden activations, then numerically stable softmax
4. **Backward** — Combined CE+softmax gradient `(probs - targets) / N`; ReLU gate via `layer1 > 0` mask
5. **Update** — Full-batch gradient descent, 1000 epochs

## Key Details

| Detail | Description |
|--------|-------------|
| Stable softmax | Subtracts row-max before `exp` |
| CE + softmax gradient | `(probs - targets) / N` — no explicit loss needed |
| ReLU gate | Binary mask `layer1 > 0` zeros inactive gradients |
| Weight init | `randn * 0.1`; biases = 0 |
| Reproducibility | `np.random.seed(42)` |
| Hyperparameters | lr=0.1, epochs=1000, hidden=32, full-batch |

---

## Results

| Epoch | SLP | MLP |
|-------|-----|-----|
| 0     | 13% | 10% |
| 100   | 94% | 72% |
| 300   | 97% | 94% |
| 600   | 98% | 98% |
| 800   | 98% | 99% |

---

## Usage & Requirements

```bash
pip install numpy scikit-learn torch

python np_slp_digits.py          # single-layer (NumPy)
python np_mlp_digits.py          # multi-layer (NumPy)
python torch_slp_digits.py       # single-layer (PyTorch, manual gradients)
python torch_slp_autograd.py     # single-layer (PyTorch, autograd)
python torch_mlp_autograd.py     # multi-layer (PyTorch, autograd)
python torch_mlp_sequential.py   # multi-layer (PyTorch, nn.Sequential)
```

## License

MIT