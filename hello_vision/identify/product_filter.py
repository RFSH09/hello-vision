"""Product vs not-product classifier: L2-regularised logistic regression on crop embeddings.

Small on purpose (about a hundred labelled tracks): it rejects hands, phones, shoes, radiator
parts and shelf clutter before SKU identification.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


class ProductFilter:
    """Binary classifier ``embedding -> probability of being a product``.

    Args:
        l2: Regularisation strength.
        steps: Gradient steps.
        lr: Learning rate.
    """

    def __init__(self, l2: float = 1e-3, steps: int = 600, lr: float = 2.0) -> None:
        self.l2 = l2
        self.steps = steps
        self.lr = lr
        self.weights: NDArray[np.float64] | None = None
        self.bias = 0.0

    def fit(self, x: Matrix, y: Sequence[int]) -> ProductFilter:
        """Train on embeddings ``x`` (N x D) with labels 1 = product, 0 = not.

        Classes are weighted equally regardless of how many samples each has.

        Raises:
            ValueError: If only one class is present.
        """
        target = np.asarray(y, dtype=np.float64)
        positives, negatives = float(target.sum()), float(len(target) - target.sum())
        if positives == 0 or negatives == 0:
            raise ValueError("need both product and not-product samples")
        sample = np.where(target == 1, 0.5 / positives, 0.5 / negatives)
        data = x.astype(np.float64)
        w = np.zeros(data.shape[1])
        b = 0.0
        for _ in range(self.steps):
            p = 1.0 / (1.0 + np.exp(-np.clip(data @ w + b, -30, 30)))
            error = (p - target) * sample
            w -= self.lr * (data.T @ error + self.l2 * w)
            b -= self.lr * float(error.sum())
        self.weights, self.bias = w, b
        return self

    def predict(self, x: Matrix) -> NDArray[np.float64]:
        """Probability of "product" for each row of ``x``."""
        if self.weights is None:
            raise RuntimeError("ProductFilter is not trained")
        z = np.clip(x.astype(np.float64) @ self.weights + self.bias, -30, 30)
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=np.float64)


def leave_one_group_out(x: Matrix, y: Sequence[int], groups: Sequence[str]) -> NDArray[np.float64]:
    """Probability for every sample from a model trained without that sample's group."""
    target = np.asarray(y)
    names = np.asarray(groups)
    out = np.full(len(target), np.nan)
    for g in np.unique(names):
        test = names == g
        out[test] = ProductFilter().fit(x[~test], list(target[~test])).predict(x[test])
    return out
