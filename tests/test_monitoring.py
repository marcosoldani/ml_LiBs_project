"""Tests for `src/monitoring/` (drift detection + prediction log)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    detect_dataset_drift,
    ks_drift,
    log_prediction,
    psi_drift,
    read_predictions,
)


@pytest.fixture
def stable_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, 5_000), rng.normal(0, 1, 5_000)


@pytest.fixture
def drifted_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, 5_000), rng.normal(2.0, 1, 5_000)


# ── KS ─────────────────────────────────────────────────────────────────────
def test_ks_drift_no_drift_under_same_distribution(stable_pair) -> None:
    ref, cur = stable_pair
    stat, p, drift = ks_drift(ref, cur, alpha=0.01)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= p <= 1.0
    assert drift is False


def test_ks_drift_flags_shifted_distribution(drifted_pair) -> None:
    ref, cur = drifted_pair
    _stat, p, drift = ks_drift(ref, cur, alpha=0.05)
    assert drift is True
    assert p < 0.05


def test_ks_drift_handles_empty_arrays() -> None:
    stat, p, drift = ks_drift(np.array([]), np.array([1.0, 2.0]))
    assert np.isnan(stat) and np.isnan(p) and drift is False


# ── PSI ────────────────────────────────────────────────────────────────────
def test_psi_drift_no_drift_under_same_distribution(stable_pair) -> None:
    ref, cur = stable_pair
    psi, drift = psi_drift(ref, cur, threshold=0.25)
    assert psi < 0.25
    assert drift is False


def test_psi_drift_flags_shifted_distribution(drifted_pair) -> None:
    ref, cur = drifted_pair
    psi, drift = psi_drift(ref, cur, threshold=0.25)
    assert psi >= 0.25
    assert drift is True


def test_psi_drift_handles_constant_reference() -> None:
    """All-equal reference must not crash (we widen the bin range)."""
    psi, drift = psi_drift(np.ones(100), np.linspace(0, 5, 100))
    assert isinstance(psi, float)
    assert isinstance(drift, bool)


# ── Dataset-level report ───────────────────────────────────────────────────
def test_detect_dataset_drift_no_drift_when_resampled() -> None:
    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 2000),
            "y": rng.uniform(-1, 1, 2000),
        }
    )
    half = len(base) // 2
    report = detect_dataset_drift(base.iloc[:half], base.iloc[half:])
    assert len(report.features) == 2
    assert report.drift_detected is False


def test_detect_dataset_drift_flags_shift() -> None:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 2000), "y": rng.normal(0, 1, 2000)})
    current = pd.DataFrame({"x": rng.normal(2, 1, 2000), "y": rng.normal(0, 1, 2000)})
    report = detect_dataset_drift(reference, current)
    flags = {f.feature: f.drift for f in report.features}
    assert flags["x"] is True
    assert flags["y"] is False
    assert 0 <= report.share_drifted <= 1


def test_detect_dataset_drift_respects_feature_subset() -> None:
    rng = np.random.default_rng(1)
    df_a = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
    df_b = pd.DataFrame({"a": rng.normal(3, 1, 500), "b": rng.normal(0, 1, 500)})
    report = detect_dataset_drift(df_a, df_b, features=["b"])
    assert [f.feature for f in report.features] == ["b"]
    # b is i.i.d. → should not be flagged
    assert report.features[0].drift is False


# ── Prediction log ─────────────────────────────────────────────────────────
def test_log_prediction_appends_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "preds.jsonl"
    log_prediction(
        task="task2",
        inputs={"soc": 2},
        outputs={"best_model": "SVM (RBF)", "accuracy": 0.99},
        log_file=target,
    )
    log_prediction(
        task="task2",
        inputs={"soc": 3},
        outputs={"best_model": "SVM (RBF)", "accuracy": 0.97},
        log_file=target,
    )
    rows = read_predictions(target)
    assert len(rows) == 2
    assert rows[0]["task"] == "task2"
    assert rows[1]["inputs"]["soc"] == 3
    assert "timestamp" in rows[0]


def test_read_predictions_returns_empty_when_missing(tmp_path: Path) -> None:
    assert read_predictions(tmp_path / "nope.jsonl") == []
