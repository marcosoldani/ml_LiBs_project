"""Tests for the ``persist_default_benchmark`` entry points of each task.

These are fast integration tests that verify the on-disk artefact
contract used by ``/api/benchmarks/{task}``: file written, valid JSON,
expected top-level keys present, and inner consistency
(``best_model`` is one of the keys in ``metrics``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import task1_loo, task2_classification, task3_aging


@pytest.mark.slow
def test_task1_persist_default_benchmark(tmp_path: Path, dataset) -> None:
    out = task1_loo.persist_default_benchmark(out_dir=tmp_path)
    assert out.exists()
    data = json.loads(out.read_text())
    assert {"excluded_aging", "excluded_temperature", "best_model",
            "metrics", "training_time_s"} <= set(data)
    # The "best_model" must be one of the keys reported in "metrics" —
    # protects against silent renames or scoring-function divergence.
    assert data["best_model"] in data["metrics"]
    assert data["best_model"] in data["training_time_s"]
    # `last_run.joblib` is also dumped next to it.
    assert (tmp_path / "last_run.joblib").exists()


@pytest.mark.slow
def test_task2_persist_default_benchmark(tmp_path: Path, dataset) -> None:
    out = task2_classification.persist_default_benchmark(out_dir=tmp_path)
    assert out.exists()
    data = json.loads(out.read_text())
    assert {"aging", "soc", "true_class", "true_label", "best_model",
            "metrics", "training_time_s"} <= set(data)
    assert data["true_class"] in (0, 1)
    assert data["true_label"] in {"Young", "Old"}
    assert data["best_model"] in data["metrics"]
    # Each model must report at least Accuracy for the single-class hold-out.
    assert "Accuracy" in data["metrics"][data["best_model"]]


@pytest.mark.slow
def test_task2_benchmark_full_grid_on_subset(tmp_path: Path, dataset) -> None:
    """Run the full grid on a 2×2 dataset slice to keep the test cheap."""
    sub = dataset[
        (dataset["Aging"].isin([0, 4])) & (dataset["SOC"].isin([0, 3]))
    ].reset_index(drop=True)
    out = task2_classification.benchmark_full_grid(out_dir=tmp_path, df=sub)
    assert out.exists()
    data = json.loads(out.read_text())
    # Two agings × two SOCs = four entries — but only those that did not
    # raise ValueError. Each entry follows the same shape as the default.
    assert len(data) >= 1
    for entry in data.values():
        assert {"aging", "soc", "true_class", "true_label", "best_model",
                "metrics", "training_time_s"} <= set(entry)
        assert entry["best_model"] in entry["metrics"]
        assert 0.0 <= entry["metrics"][entry["best_model"]]["Accuracy"] <= 1.0


@pytest.mark.slow
def test_task3_persist_default_benchmark(tmp_path: Path, dataset) -> None:
    out = task3_aging.persist_default_benchmark(out_dir=tmp_path)
    assert out.exists()
    data = json.loads(out.read_text())
    assert {"excluded_aging", "best_model", "metrics", "training_time_s",
            "per_temperature"} <= set(data)
    assert isinstance(data["per_temperature"], list) and len(data["per_temperature"]) > 0
    assert data["best_model"] in data["metrics"]
    # Per-temperature rows must carry the canonical schema.
    sample = data["per_temperature"][0]
    assert {"Temperature", "R2", "MAE", "N"} <= set(sample)
    assert (tmp_path / "last_run.joblib").exists()
