"""Streaming churn monitor: score batches, detect drift, retrain on trigger."""
from __future__ import annotations

from dataclasses import dataclass, field

from .data import Row
from .drift import DriftReport, assess
from .model import ChurnModel


@dataclass
class BatchResult:
    batch_index: int
    precision: float
    recall: float
    drift: DriftReport
    retrained: bool


def _precision_recall(rows: list[Row], probs: list[float]):
    """Precision@k / recall@k with k = number of true positives (R-precision).

    Churn is rare (single-digit base rate here), so a fixed 0.5 probability
    cutoff never fires and silently reports 0% precision regardless of model
    quality. Ranking-based evaluation — flag the top-k highest-risk accounts,
    where k matches the actual positive count — is the standard way churn/fraud
    models are evaluated in practice, and it isolates ranking quality from
    calibration.
    """
    actual_pos = sum(r.churned for r in rows)
    if actual_pos == 0:
        return 0.0, 0.0
    k = actual_pos
    order = sorted(range(len(rows)), key=lambda i: probs[i], reverse=True)
    top_k = order[:k]
    tp = sum(1 for i in top_k if rows[i].churned == 1)
    precision = tp / k
    recall = tp / actual_pos
    return precision, recall


@dataclass
class ChurnMonitor:
    """Trains once on a reference window, then scores streaming batches,
    retraining automatically whenever the prediction distribution drifts.
    """

    model: ChurnModel
    reference_scores: list[float] = field(default_factory=list)
    auto_retrain: bool = True
    drift_threshold: float = 0.25
    retrain_count: int = 0

    @classmethod
    def fit(cls, reference_rows: list[Row], auto_retrain: bool = True,
             drift_threshold: float = 0.25) -> "ChurnMonitor":
        model = ChurnModel().fit(reference_rows)
        ref_scores = model.predict_proba_batch(reference_rows)
        return cls(model, ref_scores, auto_retrain, drift_threshold)

    def process_batch(self, batch_index: int, batch: list[Row],
                      labeled_history: list[Row]) -> BatchResult:
        """Score a batch, check for drift, and retrain on accumulated labeled
        history if drift is detected and auto-retraining is enabled."""
        probs = self.model.predict_proba_batch(batch)
        precision, recall = _precision_recall(batch, probs)
        report = assess(self.reference_scores, probs, threshold=self.drift_threshold)

        retrained = False
        if report.drifted and self.auto_retrain:
            self.model = ChurnModel().fit(labeled_history)
            self.reference_scores = self.model.predict_proba_batch(labeled_history)
            self.retrain_count += 1
            retrained = True

        return BatchResult(batch_index, precision, recall, report, retrained)
