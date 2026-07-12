"""Population Stability Index (PSI) drift detection on model predictions.

PSI compares the distribution of predicted churn probabilities between a
reference window (when the model was trained) and the current window. A
silently-drifting relationship between features and outcome shows up as a
shift in the prediction distribution even before accuracy visibly collapses.
That's what should trigger a retrain, not a fixed schedule.
"""
from __future__ import annotations

from dataclasses import dataclass


def _bucketize(values: list[float], edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
                counts[i] += 1
                break
    return counts


def psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index between two score distributions.

    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant drift.
    """
    edges = [i / bins for i in range(bins + 1)]
    ref_counts = _bucketize(reference, edges)
    cur_counts = _bucketize(current, edges)
    ref_n, cur_n = len(reference) or 1, len(current) or 1
    eps = 1e-4
    total = 0.0
    for rc, cc in zip(ref_counts, cur_counts):
        ref_pct = max(rc / ref_n, eps)
        cur_pct = max(cc / cur_n, eps)
        total += (cur_pct - ref_pct) * (math_log(cur_pct) - math_log(ref_pct))
    return total


def math_log(x: float) -> float:
    import math
    return math.log(x)


@dataclass
class DriftReport:
    psi: float
    drifted: bool
    threshold: float


def assess(reference_scores: list[float], current_scores: list[float],
           threshold: float = 0.25) -> DriftReport:
    score = psi(reference_scores, current_scores)
    return DriftReport(round(score, 4), score >= threshold, threshold)
