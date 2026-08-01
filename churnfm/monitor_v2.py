"""A monitor that retrains on either signal firing, not PSI alone.

PSI answers "has the model's opinion about the world changed shape". Outcome
drift answers "is the model's opinion still matching what actually happens".
A model whose relationship to the world has quietly inverted can leave the
first question answered "no" indefinitely while the second says "yes" the
moment labels arrive. :mod:`churnfm.monitor` asks only the first question, so
this failure mode never triggers a retrain: precision can sit at 10-20% for
the rest of the stream's life with the monitor reporting nothing wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .data import Row
from .drift import DriftReport, assess
from .model import ChurnModel
from .monitor import _precision_recall
from .outcome_drift import OutcomeDriftReport, assess_outcomes


@dataclass
class BatchResultV2:
    batch_index: int
    precision: float
    recall: float
    psi: DriftReport
    outcome: OutcomeDriftReport
    retrained: bool
    #: Which signal actually caused the retrain, when one happened. Auditable
    #: rather than a single opaque "drifted" flag.
    trigger: str = ""


@dataclass
class ChurnMonitorV2:
    """Retrains whenever PSI or outcome drift fires, whichever comes first."""

    model: ChurnModel
    reference_scores: List[float] = field(default_factory=list)
    reference_labels: List[int] = field(default_factory=list)
    auto_retrain: bool = True
    psi_threshold: float = 0.25
    outcome_threshold: float = 1.3
    min_labeled_batch: int = 30
    retrain_count: int = 0
    #: How many retrains were triggered by outcome drift specifically, i.e.
    #: how many PSI would have missed entirely on its own.
    outcome_triggered_count: int = 0

    @classmethod
    def fit(
        cls,
        reference_rows: List[Row],
        auto_retrain: bool = True,
        psi_threshold: float = 0.25,
        outcome_threshold: float = 1.3,
    ) -> "ChurnMonitorV2":
        model = ChurnModel().fit(reference_rows)
        ref_scores = model.predict_proba_batch(reference_rows)
        ref_labels = [r.churned for r in reference_rows]
        return cls(model, ref_scores, ref_labels, auto_retrain,
                   psi_threshold, outcome_threshold)

    def process_batch(
        self, batch_index: int, batch: List[Row], labeled_history: List[Row]
    ) -> BatchResultV2:
        probs = self.model.predict_proba_batch(batch)
        labels = [r.churned for r in batch]
        precision, recall = _precision_recall(batch, probs)

        psi_report = assess(self.reference_scores, probs, threshold=self.psi_threshold)
        outcome_report = assess_outcomes(
            self.reference_scores, self.reference_labels, probs, labels,
            threshold=self.outcome_threshold, min_batch_size=self.min_labeled_batch,
        )

        drifted = psi_report.drifted or outcome_report.drifted
        trigger = ""
        if psi_report.drifted:
            trigger = "psi"
        if outcome_report.drifted:
            trigger = "outcome" if not psi_report.drifted else "psi+outcome"

        retrained = False
        if drifted and self.auto_retrain:
            self.model = ChurnModel().fit(labeled_history)
            self.reference_scores = self.model.predict_proba_batch(labeled_history)
            self.reference_labels = [r.churned for r in labeled_history]
            self.retrain_count += 1
            if outcome_report.drifted and not psi_report.drifted:
                self.outcome_triggered_count += 1
            retrained = True

        return BatchResultV2(
            batch_index, precision, recall, psi_report, outcome_report,
            retrained, trigger,
        )
