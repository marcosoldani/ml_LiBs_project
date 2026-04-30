"""Evaluation metrics and helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass
class RegressionScore:
    mse: float
    mae: float
    r2: float
    r2_real: float
    r2_imag: float

    def as_dict(self) -> dict:
        return {
            "MSE": round(self.mse, 6),
            "MAE": round(self.mae, 6),
            "R2": round(self.r2, 6),
            "R2_real": round(self.r2_real, 6),
            "R2_imag": round(self.r2_imag, 6),
        }


@dataclass
class ClassificationScore:
    accuracy: float
    f1: float
    auc: float

    def as_dict(self) -> dict:
        return {
            "Accuracy": round(self.accuracy, 6),
            "F1": round(self.f1, 6),
            "AUC": round(self.auc, 6),
        }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionScore:
    """Compute regression metrics, reporting per-component R² as well."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        r2_real = r2_score(y_true, y_pred)
        r2_imag = r2_real
    else:
        r2_real = r2_score(y_true[:, 0], y_pred[:, 0])
        r2_imag = r2_score(y_true[:, 1], y_pred[:, 1])
    return RegressionScore(
        mse=float(mean_squared_error(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float((r2_real + r2_imag) / 2),
        r2_real=float(r2_real),
        r2_imag=float(r2_imag),
    )


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> ClassificationScore:
    """Compute classification metrics. AUC defaults to NaN if probabilities missing."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    auc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None else float("nan")
    return ClassificationScore(accuracy=acc, f1=f1, auc=auc)
