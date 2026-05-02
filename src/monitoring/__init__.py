"""Monitoring & continual-learning helpers for the Battery GEIS pipeline.

Implements the Lecture 06 (*Monitoring & Continual Learning*) playbook:

- ``drift`` — population-level drift tests (KS, PSI) on numerical features and
  on model targets. The reference distribution is the training CSV, the
  *current* distribution is whatever new measurements arrive (e.g. a fresh
  battery characterisation campaign).
- ``logging`` — append-only JSON-Lines store for predictions, used both as an
  audit trail and as the source of *current* distributions for drift tests.
"""

from src.monitoring.drift import (
    DriftReport,
    FeatureDrift,
    detect_dataset_drift,
    ks_drift,
    psi_drift,
)
from src.monitoring.prediction_log import (
    PredictionRecord,
    log_prediction,
    read_predictions,
)

__all__ = [
    "DriftReport",
    "FeatureDrift",
    "PredictionRecord",
    "detect_dataset_drift",
    "ks_drift",
    "log_prediction",
    "psi_drift",
    "read_predictions",
]
