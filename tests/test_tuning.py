"""Smoke tests for the GridSearchCV pipeline in `src/models/tuning.py`.

The aim is *coverage of the wiring*, not benchmarking — we shrink the grid to a
single combination so the search returns in a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.models.tuning import (
    _classification_candidates,
    _regression_candidates,
    load_best_params,
    tune_task1,
    tune_task2,
    tune_task3,
)


class _MiniGrid:
    """Tiny stand-in for the per-task tuning config (1 value per axis)."""

    class _RF:
        n_estimators = [25]
        max_depth = [5]

    class _GB:
        n_estimators = [25]
        max_depth = [2]
        learning_rate = [0.1]

    class _KNN:
        n_neighbors = [5]
        weights = ["distance"]

    class _SVM:
        C = [1.0]
        gamma = ["scale"]

    random_forest = _RF()
    gradient_boosting = _GB()
    knn = _KNN()
    svm = _SVM()


def test_regression_candidates_shape() -> None:
    grid = _regression_candidates(_MiniGrid())
    assert {"Random Forest", "Gradient Boosting", "K-Nearest Neighbors"} <= set(grid)
    for _name, (estimator, params) in grid.items():
        assert estimator is not None
        assert isinstance(params, dict)
        assert all(isinstance(v, list) for v in params.values())


def test_classification_candidates_shape() -> None:
    grid = _classification_candidates(_MiniGrid())
    assert {"Random Forest", "Gradient Boosting", "SVM (RBF)"} <= set(grid)
    for _name, (estimator, params) in grid.items():
        assert estimator is not None
        assert isinstance(params, dict)


@pytest.mark.slow
def test_tune_task1_writes_best_params(tmp_path: Path, dataset, monkeypatch) -> None:
    """End-to-end: tune_task1 must persist a JSON with best_params per model."""
    from src.models import tuning as tuning_module

    monkeypatch.setattr(tuning_module, "_out_dir", lambda: tmp_path)

    sub = dataset[dataset["Aging"].isin([0, 1, 2])].copy()
    with patch.object(tuning_module, "_regression_candidates", return_value=_regression_candidates(_MiniGrid())):
        out = tune_task1(df=sub)

    assert out.exists()
    payload = json.loads(out.read_text())
    assert {"Random Forest", "K-Nearest Neighbors"} <= set(payload)
    for _name, body in payload.items():
        assert "best_params" in body


@pytest.mark.slow
def test_tune_task3_writes_best_params(tmp_path: Path, dataset, monkeypatch) -> None:
    from src.models import tuning as tuning_module

    monkeypatch.setattr(tuning_module, "_out_dir", lambda: tmp_path)

    sub = dataset[dataset["Aging"].isin([0, 1, 2, 3])].copy()
    with patch.object(tuning_module, "_regression_candidates", return_value=_regression_candidates(_MiniGrid())):
        out = tune_task3(df=sub)

    assert out.exists()
    payload = json.loads(out.read_text())
    assert {"Random Forest", "K-Nearest Neighbors"} <= set(payload)


@pytest.mark.slow
def test_tune_task2_writes_best_params(tmp_path: Path, dataset, monkeypatch) -> None:
    from src.models import tuning as tuning_module

    monkeypatch.setattr(tuning_module, "_out_dir", lambda: tmp_path)

    with patch.object(tuning_module, "_classification_candidates", return_value=_classification_candidates(_MiniGrid())):
        out = tune_task2(soc=2, df=dataset)

    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["soc"] == 2
    assert {"Random Forest", "SVM (RBF)"} <= set(payload["results"])


def test_load_best_params_returns_empty_when_missing(tmp_path: Path, monkeypatch) -> None:
    from src.models import tuning as tuning_module

    monkeypatch.setattr(tuning_module, "_out_dir", lambda: tmp_path)
    assert load_best_params("task1") == {}


def test_load_best_params_reads_persisted_file(tmp_path: Path, monkeypatch) -> None:
    from src.models import tuning as tuning_module

    monkeypatch.setattr(tuning_module, "_out_dir", lambda: tmp_path)
    payload = {
        "Random Forest": {
            "best_params": {"n_estimators": 100, "max_depth": 10},
            "best_score_neg_mse": -0.42,
        }
    }
    (tmp_path / "task1_best.json").write_text(json.dumps(payload))
    out = load_best_params("task1")
    assert out["Random Forest"]["n_estimators"] == 100


def test_load_best_params_task2_unwraps_results(tmp_path: Path, monkeypatch) -> None:
    """task2 persists results under a `results` key — load_best_params must unwrap it."""
    from src.models import tuning as tuning_module

    monkeypatch.setattr(tuning_module, "_out_dir", lambda: tmp_path)
    payload = {
        "soc": 2,
        "results": {
            "SVM (RBF)": {
                "best_params": {"C": 10.0, "gamma": "scale"},
                "best_score_f1": 0.95,
            }
        },
    }
    (tmp_path / "task2_best.json").write_text(json.dumps(payload))
    out = load_best_params("task2")
    assert out["SVM (RBF)"] == {"C": 10.0, "gamma": "scale"}


def test_synthetic_dataset_has_expected_shape(dataset: pd.DataFrame) -> None:
    """Sanity check that the dataset fixture is in the expected ballpark."""
    assert len(dataset) > 100
    assert "Aging" in dataset.columns
    assert np.issubdtype(dataset["Frequency"].dtype, np.floating)
