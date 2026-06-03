# np_slp_digits.py - Step-by-Step Explainer

This script trains a single-layer softmax classifier (multiclass logistic regression) on scikit-learn's digits dataset using only NumPy.

The full flow is:

1. Load and standardize the input data
2. One-hot encode labels
3. Initialize weights and biases
4. Run gradient descent with softmax + cross-entropy gradient
5. Print training accuracy during learning

## 1. Imports and softmax

The script imports:

- `load_digits` from scikit-learn to get digit features and labels
- `numpy as np` for vectorized math

It defines:

```python
def softmax(x):
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)
```

Why subtract the row-wise max first?

- Numerical stability. It prevents very large logits from causing overflow in `np.exp`.
- The softmax result is unchanged because adding or subtracting the same constant from all logits in a row does not change probabilities.

## 2. Load and preprocess data

```python
X, y = load_digits(return_X_y=True)
X = (X - X.mean()) / (X.std())
y_one_hot = np.eye(10)[y]
```

Shapes in this dataset:

- `X`: `(N, 64)` where each sample is an 8x8 image flattened to 64 features
- `y`: `(N,)` integer labels from 0 to 9
- `y_one_hot`: `(N, 10)` one-hot target vectors

Preprocessing details:

- `X` is standardized to roughly zero mean and unit variance using global dataset mean/std.
- `np.eye(10)[y]` maps each label to a one-hot vector, such as class `3 -> [0,0,0,1,0,0,0,0,0,0]`.

## 3. Initialize model parameters

```python
np.random.seed(42)
W = np.random.randn(64, 10) * 0.1
b = np.zeros((1, 10))
learning_rate = 0.1
```

Interpretation:

- `W` has shape `(64, 10)`: one weight vector per class
- `b` has shape `(1, 10)`: one bias per class
- `learning_rate` controls update size
- Random seed makes runs reproducible

## 4. Training loop

```python
for epoch in range(1000):
    probs = softmax(X @ W + b)
    dlogits = (probs - y_one_hot) / len(X)
    W -= learning_rate * X.T @ dlogits
    b -= learning_rate * dlogits.sum(0, keepdims=True)

    if epoch % 100 == 0:
        acc = np.mean(probs.argmax(1) == y)
        print(f"Epoch {epoch:4d} | Acc: {acc:.1%}")
```

### 4.1 Forward pass

- `X @ W + b` computes class logits for every sample, shape `(N, 10)`.
- `softmax(...)` converts logits to class probabilities.

### 4.2 Gradient term

For softmax with cross-entropy, the gradient with respect to logits simplifies to:

$$
\frac{\partial L}{\partial z} = \frac{p - y}{N}
$$

That is exactly:

- `dlogits = (probs - y_one_hot) / len(X)`

### 4.3 Parameter gradients and updates

- `X.T @ dlogits` gives gradient for `W` with shape `(64, 10)`.
- `dlogits.sum(0, keepdims=True)` gives gradient for `b` with shape `(1, 10)`.
- Parameters move opposite the gradient via gradient descent.

## 5. Accuracy monitoring

Every 100 epochs, the script reports training accuracy:

- `probs.argmax(1)` selects the predicted class index
- It compares predictions with `y`
- `np.mean(...)` returns fraction correct

This provides a quick view of whether learning is improving.

## 6. What this file demonstrates

This script is a compact baseline for digit classification and shows:

- A single-layer perceptron for multiclass classification is linear logits + softmax
- Softmax + cross-entropy gives a clean vectorized gradient
- End-to-end training can be implemented in a small amount of NumPy code

In this repository, it serves as the simple baseline to compare against the deeper model in [np_mlp_digits.py](np_mlp_digits.py).
