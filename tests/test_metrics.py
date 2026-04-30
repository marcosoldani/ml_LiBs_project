"""Tests for the metrics dataclasses."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import classification_metrics, regression_metrics


def test_regression_metrics_perfect_prediction() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    score = regression_metrics(y, y)
    assert score.mse == 0.0
    assert score.mae == 0.0
    assert score.r2 == 1.0
    assert score.r2_real == 1.0
    assert score.r2_imag == 1.0


def test_classification_metrics_perfect_prediction() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = y_true.copy()
    score = classification_metrics(y_true, y_pred, y_proba=np.array([0.1, 0.9, 0.2, 0.8]))
    assert score.accuracy == 1.0
    assert score.f1 == 1.0
    assert score.auc == 1.0


def test_classification_metrics_auc_is_nan_without_proba() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = y_true.copy()
    score = classification_metrics(y_true, y_pred)
    assert np.isnan(score.auc)
