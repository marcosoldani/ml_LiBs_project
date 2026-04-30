"""Tests for src/tracking.py and src/models/tuning.py integration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.registry import regression_models
from src.models.tuning import load_best_params
from src.tracking import dataset_sha256, list_runs


def test_dataset_sha256_is_stable() -> None:
    a = dataset_sha256()
    b = dataset_sha256()
    assert a == b
    assert a is None or len(a) == 16  # truncated hex


def test_load_best_params_returns_dict() -> None:
    best = load_best_params("task1")
    assert isinstance(best, dict)
    # Each entry must be ``{model_name: {param: value}}``
    for name, params in best.items():
        assert isinstance(name, str)
        assert isinstance(params, dict)


def test_use_tuned_overrides_baseline_when_file_exists(tmp_path: Path) -> None:
    """If a tuning file is present, regression_models(use_tuned=True) must apply it."""
    baseline = regression_models()
    tuned = regression_models(use_tuned=True, task="task1")
    # The Ridge baseline should be unchanged (not in the grid); RF should match the file
    best = load_best_params("task1")
    if "Random Forest" in best and best["Random Forest"]:
        tuned_rf = tuned["Random Forest"].get_params()
        for k, v in best["Random Forest"].items():
            assert tuned_rf[k] == v
    # Baseline RF still has its hard-coded defaults
    assert baseline["Random Forest"].get_params()["n_estimators"] == 200


def test_list_runs_is_dataframe() -> None:
    df = list_runs("task1")
    assert isinstance(df, pd.DataFrame)


def test_tracking_no_op_when_disabled(monkeypatch, tmp_path: Path) -> None:
    """Disabling tracking must make the helpers idempotent no-ops."""
    from src import tracking

    monkeypatch.setattr(tracking, "_is_enabled", lambda: False)
    # Should not raise, should not open any mlflow run
    with tracking.track_pipeline(task="test", run_name="noop"):
        pass
    tracking.log_model_run(name="x", params={}, metrics={"R2": 0.9}, duration_s=0.1)
    assert tracking.list_runs("test").empty


def test_load_best_params_missing_returns_empty(tmp_path: Path, monkeypatch) -> None:
    from src.models import tuning

    monkeypatch.setattr(tuning, "_out_dir", lambda: tmp_path)
    assert tuning.load_best_params("doesnotexist") == {}


def test_persist_and_reload_round_trip(tmp_path: Path, monkeypatch) -> None:
    from src.models import tuning

    monkeypatch.setattr(tuning, "_out_dir", lambda: tmp_path)
    payload = {
        "Random Forest": {"best_params": {"n_estimators": 123}, "best_score_mse": 0.01},
    }
    target = tmp_path / "taskX_best.json"
    target.write_text(json.dumps(payload))
    assert tuning.load_best_params("taskX") == {"Random Forest": {"n_estimators": 123}}
