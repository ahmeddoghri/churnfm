"""ChurnFM, a churn classifier with drift-triggered automated retraining.

>>> from churnfm import ChurnModel, generate
>>> rows = generate(n=500, seed=0)
>>> model = ChurnModel().fit(rows)
>>> 0.0 <= model.predict_proba(rows[0]) <= 1.0
True
"""
from .data import FEATURE_NAMES, Row, features, generate
from .drift import DriftReport, assess, psi
from .model import ChurnModel
from .monitor import BatchResult, ChurnMonitor
from .monitor_v2 import BatchResultV2, ChurnMonitorV2
from .outcome_drift import OutcomeDriftReport, assess_outcomes, log_loss
from .scenarios import (
    covariate_shift_confound,
    holdout_concept_drift,
    pure_concept_drift,
)

__all__ = [
    "Row", "generate", "features", "FEATURE_NAMES",
    "ChurnModel",
    "DriftReport", "assess", "psi",
    "ChurnMonitor", "BatchResult",
    "ChurnMonitorV2", "BatchResultV2",
    "OutcomeDriftReport", "assess_outcomes", "log_loss",
    "covariate_shift_confound", "pure_concept_drift", "holdout_concept_drift",
]
__version__ = "0.1.0"
