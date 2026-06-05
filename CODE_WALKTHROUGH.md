# Code Walkthrough: From NumPy SLP to PyTorch MLP

This repo walks a digits classifier from a minimal NumPy single-layer perceptron up to a multi-layer PyTorch model using both low-level autograd and high-level `nn.Sequential` APIs. The progression is:

1. `np_slp_digits.py`: single linear layer + softmax in pure NumPy  
2. `np_mlp_digits.py`: 1-hidden-layer MLP with ReLU, still pure NumPy  
3. `torch_slp_digits.py`: SLP reimplemented in PyTorch with manual training loop  
4. `torch_slp_autograd.py`: explicit use of autograd and optimizers  
5. `torch_mlp_autograd.py` / `torch_mlp_sequential.py`: multi-layer PyTorch models (functional vs `nn.Sequential`)  

Jupyter notebooks mirror the NumPy scripts for an interactive, pedagogical version of the same steps.

---

## 1. NumPy SLP: `np_slp_digits.py`

**Goal:** Show that a single linear layer with softmax is enough to get a decent classifier on `sklearn`'s digits dataset, implemented from scratch with full-batch gradient descent.

Key components:

- **Data pipeline**

  - Load digits with `load_digits(return_X_y=True)` giving `X` of shape `(1797, 64)` and integer labels `y` in `0–9`.  
  - Standardize globally:  
    ```python
    X = (X - X.mean()) / (X.std())
    ```  
    This treats all 64 pixels as one feature distribution. Note: this is a simplification — it applies a single mean and std across all pixels. The MLP (step 2) refines this to per-feature normalization, which better conditions the optimization.
  - One-hot encode labels with `np.eye(10)[y]` to get `y_one_hot` of shape `(1797, 10)`.

- **Model definition**

  - Single linear layer: `W` is `(64, 10)` and `b` is `(1, 10)`. Forward pass is:
    - logits: `X @ W + b`  
    - probabilities: `softmax(logits)` with max-subtraction for stability.

- **Training loop**

  - Full-batch gradient descent for 1000 epochs.  
  - Gradient of cross-entropy with softmax:  
    ```python
    dlogits = (probs - y_one_hot) / len(X)
    W -= learning_rate * X.T @ dlogits
    b -= learning_rate * dlogits.sum(0, keepdims=True)
    ```  
    This shows the "softmax + CE → probs − one_hot" simplification explicitly.
  - Accuracy is printed every 100 epochs using `probs.argmax(1)` vs `y`.

**Pedagogical move:** This file establishes the full training loop (forward, loss gradient, backward, update) in the simplest possible setting: a single linear map. Students see that nothing "magical" is required beyond linear algebra and the softmax trick.

---

## 2. NumPy MLP: `np_mlp_digits.py`

**Goal:** Introduce depth (one hidden layer with ReLU) and a more careful preprocessing pipeline, while keeping everything in NumPy and still using explicit gradients.

What changes relative to `np_slp_digits.py`?

- **Per-feature normalization**

  - The `normalize` function standardizes each feature (pixel) independently:
    ```python
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    ```  
    This better conditions the optimization, especially when features have different scales. The `1e-8` epsilon prevents division by zero on constant pixels.

- **Network architecture: 64 → 32 → 10**

  - Hidden layer projection: `W1 ∈ ℝ^{64×32}`, `b1 ∈ ℝ^{1×32}`.  
  - Output layer: `W2 ∈ ℝ^{32×10}`, `b2 ∈ ℝ^{1×10}`.  
  - ReLU nonlinearity: `layer1 = np.maximum(0, X @ W1 + b1)`.  
  - Softmax output as before.

- **Forward function**

  - Encapsulated as:
    ```python
    def forward(x_in):
        layer1 = np.maximum(0, x_in @ W1 + b1)
        probs = softmax(layer1 @ W2 + b2)
        return layer1, probs
    ```  
    This introduces the idea of modular forward passes that will map naturally to PyTorch's `forward` methods.

- **Backprop through ReLU and two layers**

  - Output error (same CE + softmax simplification):
    ```python
    output_error = (probs - y_one_hot) / len(X)
    ```
  - Backprop through output layer into hidden activations:
    ```python
    layer1_error = (output_error @ W2.T) * (layer1 > 0)
    ```  
    The `(layer1 > 0)` mask is the ReLU derivative.
  - Parameter updates use full-batch gradients:
    ```python
    W2 -= learning_rate * layer1.T @ output_error
    b2 -= learning_rate * output_error.sum(0, keepdims=True)
    W1 -= learning_rate * X.T @ layer1_error
    b1 -= learning_rate * layer1_error.sum(0, keepdims=True)
    ```

**Pedagogical move:** This file introduces genuine backprop in a way that mirrors the matrix calculus in textbooks: one hidden layer, explicit derivatives, and clear separation between forward and backward. It sets students up to appreciate PyTorch's autograd as an automation of what they just did manually.

---

## 3. PyTorch SLP: `torch_slp_digits.py`

**Goal:** Rebuild the same single-layer classifier, but now with PyTorch tensors, a training loop that looks nearly identical to the NumPy version, and manual `.backward()` + `torch.no_grad()` updates.

Typical elements:

- **Data loading and tensor conversion**

  - Uses `sklearn.datasets.load_digits` then wraps the `ndarray` features and labels into `torch.tensor` objects.  
  - Normalization parallels the NumPy scripts, but performed on tensors.

- **Model**

  - Explicit weight and bias tensors (`W` and `b`) with `requires_grad=True` — no `nn.Linear` or any `nn.Module` yet. This keeps a one-to-one correspondence with the NumPy version so students can compare directly.  
  - Softmax computed via `torch.softmax` or `logits.exp() / logits.exp().sum(dim=1, keepdim=True)` to match the NumPy implementation.

- **Training loop**

  - Compute logits, softmax probabilities, and a cross-entropy loss (either manually or via `nn.CrossEntropyLoss`).  
  - Call `loss.backward()` to let autograd compute parameter gradients.  
  - Manual SGD update under `torch.no_grad()` to mirror the NumPy gradient descent:
    ```python
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()
    ```
  - Accuracy logging similar to NumPy (`probs.argmax(dim=1)` vs `y`).

**Pedagogical move:** This file is the bridge: same SLP as before, but now with PyTorch. Students see a one-to-one mapping from NumPy arrays to tensors, and from hand-coded grads to autograd-backed `.grad` fields. The introduction of `nn.Module` is deliberately deferred to the next file.

---

## 4. PyTorch SLP with Autograd Abstractions: `torch_slp_autograd.py`

**Goal:** Keep the architecture simple (still an SLP) but lean more into PyTorch conventions: modules, optimizers, built-in losses.

Key shifts:

- **Module-based model**

  - Define a small `nn.Module` (e.g., `SLP(nn.Module)`) with an internal `nn.Linear(64, 10)` layer and a `forward` that returns logits.

- **Loss and optimizer**

  - Use `nn.CrossEntropyLoss()` which combines `log_softmax` and NLL loss, so the code drops the explicit softmax step.  
  - Introduce `torch.optim.SGD(model.parameters(), lr=...)` so the parameter update loop becomes:
    ```python
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ```

- **Cleaner training loop**

  - Loop is now "canonical PyTorch": forward → loss → backward → optimizer step, with accuracy and loss logging.

**Pedagogical move:** This file shows that once autograd and `nn.Module` are in play, the training loop simplifies and uses standard building blocks, while still being conceptually the same as the NumPy gradient descent.

---

## 5. PyTorch MLP: `torch_mlp_autograd.py` and `torch_mlp_sequential.py`

**Goal:** Mirror the NumPy MLP architecture (hidden layer + nonlinearity) and then show two equivalent PyTorch styles: a hand-written `forward` versus a stacked `nn.Sequential`.

- **Architecture**

  - Hidden size matches the NumPy version (e.g., `64 → 32 → 10`), using `nn.Linear` layers and a nonlinearity like `nn.ReLU()`.

- **Two styles**

  - `torch_mlp_autograd.py`:  
    - Explicit `MLP(nn.Module)` with `self.fc1`, `self.fc2`, etc., and a `forward` that composes them:
      ```python
      x = self.fc1(x)
      x = torch.relu(x)
      x = self.fc2(x)
      return x
      ```
    - Parameter updates are applied in a loop over all learnable tensors, avoiding per-parameter update lines:
      ```python
      for p in params:
          p -= lr * p.grad
          p.grad.zero_()
      ```
      This is a notable difference from `torch_slp_digits.py` where `W` and `b` were updated individually.
  - `torch_mlp_sequential.py`:  
    - Same architecture expressed as:
      ```python
      model = nn.Sequential(
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 10)
      )
      ```
    - Emphasizes that for plain feed-forward stacks, `Sequential` is concise and idiomatic.

- **Training**

  - The overall training structure (forward → loss → backward → update) is the same as `torch_slp_autograd.py`, only with a deeper network and (likely) better performance.

**Pedagogical move:** Students see that the conceptual jump from "one layer + softmax" to "MLP" is small once they are in PyTorch, and that there are multiple equivalent ways to express the same computation graph.

---

## 6. Explainer Docs and Notebooks

Two explainer markdowns back up the code:

- `np_slp_digits_explainer.md`: step-by-step explanation of the NumPy SLP script and the math behind softmax + cross-entropy.  
- `torch_slp_digits_explainer.md`: similar walkthrough for the PyTorch SLP version, highlighting autograd and the training loop differences.

Notebooks:

- `np_slp_digits.ipynb`: interactive, cell-based version of the NumPy SLP script, useful for stepping through the data pipeline, visualizing digit samples, and experimenting with the training loop.  
- `np_mlp_digits.ipynb`: replicates the NumPy MLP script in notebook format, useful for visualizing intermediate quantities or making small experimental changes (e.g., different hidden sizes, learning rates).

---

## How the Ideas Evolve

Conceptually, the repo is structured along two axes:

- **Model complexity**
  - Start: linear classifier only (`np_slp_digits.py`, `torch_slp_digits.py`).  
  - Next: one hidden layer with ReLU (`np_mlp_digits.py`, PyTorch MLP scripts).

- **Framework abstraction**
  - Start: pure NumPy, fully manual gradients.  
  - Middle: PyTorch tensors with manual update steps (`torch_slp_digits.py`).  
  - End: idiomatic PyTorch with `nn.Module`, `nn.Sequential`, `CrossEntropyLoss`, and optimizers.

A learner can follow the path:

1. Understand every scalar and matrix operation in the NumPy SLP.  
2. Generalize that to a hidden layer and ReLU in the NumPy MLP.  
3. Port the mental model to PyTorch tensors and autograd.  
4. Finally, collapse boilerplate using `nn.Sequential` and standard PyTorch abstractions.
