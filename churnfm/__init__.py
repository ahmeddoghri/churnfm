"""ChurnFM — a churn classifier with drift-triggered automated retraining.

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

__all__ = [
    "Row", "generate", "features", "FEATURE_NAMES",
    "ChurnModel",
    "DriftReport", "assess", "psi",
    "ChurnMonitor", "BatchResult",
]
__version__ = "0.1.0"
