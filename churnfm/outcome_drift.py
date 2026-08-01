"""Detecting drift from what actually happened, not just what the model predicted.

PSI compares the distribution of predicted scores between a reference window
and the current one. That is a real signal, and it is blind to a specific and
common failure: **the relationship between features and outcome can invert
while the predicted-score distribution barely moves.**

Concretely: before a product pivot, high-usage customers were safe and
low-usage customers churned. After, it's reversed. If usage happens to be
roughly symmetric around its mean, a logistic model trained pre-drift assigns
scores that are, on average, just as spread out as before, because it is
scoring the same input distribution through the same fitted function. The
scores look like a healthy, stable model. They are just answering the wrong
question. Measured directly: PSI stayed under 0.05 against a threshold of 0.25
while precision fell from roughly 48% to 12%.

The fix is not a better score-distribution test. It is a second signal that
does not depend on the score distribution at all: **once labels for a batch
are available, compare how well the reference-window model actually predicted
those outcomes against how well it predicted the reference window's own
outcomes.** A model whose relationship to the world has changed gets
measurably worse at labels it has never seen adjusted for, even when its
score distribution has not moved.

This trades immediacy for ground truth. PSI can flag drift before any label
exists for the new data; outcome drift needs labels, which arrive with a lag
in a real churn pipeline (you know someone churned weeks after the fact, not
the instant you scored them). The monitor in :mod:`churnfm.monitor_v2` runs
both and retrains on either firing, because they catch different failures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .data import Row


def log_loss(probs: List[float], labels: List[int], eps: float = 1e-6) -> float:
    """Mean binary cross-entropy: how surprised the model was by what happened."""
    if not probs:
        return 0.0
    total = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, eps), 1 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(probs)


@dataclass
class OutcomeDriftReport:
    reference_log_loss: float
    batch_log_loss: float
    #: batch_log_loss / reference_log_loss. 1.0 means no change; the model is
    #: exactly as surprised by new outcomes as it was by the ones it trained on.
    degradation_ratio: float
    drifted: bool
    threshold: float
    #: How many labeled examples backed this judgment; a ratio computed from
    #: a handful of labels is close to noise, not a signal.
    batch_size: int
    min_batch_size: int


def assess_outcomes(
    reference_probs: List[float],
    reference_labels: List[int],
    batch_probs: List[float],
    batch_labels: List[int],
    threshold: float = 1.3,
    min_batch_size: int = 30,
) -> OutcomeDriftReport:
    """Compare predictive quality on labeled outcomes, reference vs. current.

    ``threshold`` is a ratio, not a probability: batch log-loss 30% worse than
    reference log-loss (ratio 1.3) is the default trigger. A ratio near 1.0 is
    a model performing on new data the way it performed on the data it was fit
    to, which is what "no drift" actually means once labels exist.
    """
    reference_loss = log_loss(reference_probs, reference_labels)
    batch_loss = log_loss(batch_probs, batch_labels)

    # Too few labeled examples to trust: a ratio from 5 points is noise, and
    # reporting "drifted" from noise would trigger retrains on nothing.
    if len(batch_labels) < min_batch_size:
        return OutcomeDriftReport(
            round(reference_loss, 4), round(batch_loss, 4), 1.0, False,
            threshold, len(batch_labels), min_batch_size,
        )

    ratio = batch_loss / reference_loss if reference_loss > 1e-9 else (
        1.0 if batch_loss < 1e-9 else float("inf")
    )
    return OutcomeDriftReport(
        round(reference_loss, 4), round(batch_loss, 4), round(ratio, 4),
        ratio >= threshold, threshold, len(batch_labels), min_batch_size,
    )
