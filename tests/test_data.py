"""Tests for the data layer."""

from __future__ import annotations

from src.data.loader import dataset_summary


def test_dataset_has_expected_shape(dataset) -> None:
    assert list(dataset.columns) == [
        "Aging",
        "Temperature",
        "SOC",
        "Frequency",
        "Z_real",
        "Z_imag",
    ]
    assert dataset["Aging"].nunique() == 5
    assert dataset["Temperature"].nunique() == 8
    assert dataset["SOC"].nunique() == 5


def test_dataset_summary_contract(dataset) -> None:
    summary = dataset_summary(dataset)
    assert summary["n_combinations"] == 40
    assert summary["n_curves"] == 200
    assert summary["freq_min"] < 1.0
    assert summary["freq_max"] > 1000.0
