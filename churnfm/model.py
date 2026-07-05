"""A small, dependency-free logistic-regression churn classifier.

Standing in for a tabular foundation model (the ``google/tabfm``-style class of
pretrained tabular classifiers trending on Hugging Face): the point of ChurnFM
isn't the model architecture, it's the **monitoring and retraining loop** around
it. Swap ``ChurnModel`` for a real tabular foundation model or a gradient-
boosted-tree model via the same ``fit``/``predict_proba`` interface.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .data import Row, features


def _standardize(rows: list[Row]) -> tuple[list[list[float]], list[float], list[float]]:
    X = [features(r) for r in rows]
    n_feat = len(X[0])
    means = [sum(x[j] for x in X) / len(X) for j in range(n_feat)]
    stds = [
        math.sqrt(sum((x[j] - means[j]) ** 2 for x in X) / len(X)) or 1.0
        for j in range(n_feat)
    ]
    return X, means, stds


@dataclass
class ChurnModel:
    """Logistic regression trained with plain gradient descent."""

    lr: float = 0.1
    epochs: int = 300
    l2: float = 0.01
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)

    def fit(self, rows: list[Row], seed: int = 0) -> "ChurnModel":
        rng = random.Random(seed)
        X_raw, self.means, self.stds = _standardize(rows)
        X = [[(x[j] - self.means[j]) / self.stds[j] for j in range(len(x))] for x in X_raw]
        y = [r.churned for r in rows]
        n_feat = len(X[0])
        self.weights = [rng.uniform(-0.01, 0.01) for _ in range(n_feat)]
        self.bias = 0.0

        n = len(X)
        for _ in range(self.epochs):
            grad_w = [0.0] * n_feat
            grad_b = 0.0
            for xi, yi in zip(X, y):
                z = self.bias + sum(w * f for w, f in zip(self.weights, xi))
                p = 1 / (1 + math.exp(-z))
                err = p - yi
                for j in range(n_feat):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(n_feat):
                grad_w[j] = grad_w[j] / n + self.l2 * self.weights[j]
                self.weights[j] -= self.lr * grad_w[j]
            self.bias -= self.lr * (grad_b / n)
        return self

    def predict_proba(self, row: Row) -> float:
        x = features(row)
        # clip: a stale model's standardization stats can be wildly
        # out-of-distribution against drifted inputs (e.g. a price range that
        # tripled), which would otherwise blow up z-scores and spuriously
        # dominate ranking. Clipping is standard practice against this.
        xs = [max(-5.0, min(5.0, (x[j] - self.means[j]) / self.stds[j])) for j in range(len(x))]
        z = self.bias + sum(w * f for w, f in zip(self.weights, xs))
        return 1 / (1 + math.exp(-z))

    def predict_proba_batch(self, rows: list[Row]) -> list[float]:
        return [self.predict_proba(r) for r in rows]
