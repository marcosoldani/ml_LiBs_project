"""Tests for src/visualization/plots.py.

Plotly figures are pure outputs of model results, so we can test their
structural invariants (traces, titles, axis labels, -Im(Z) convention)
without rendering them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.constants import CLASS_NAMES
from src.visualization.plots import (
    actual_vs_predicted_nyquist,
    confusion_matrix_figure,
    metric_bar_chart,
    nyquist_aging_evolution,
    nyquist_by_soc,
    nyquist_classification_map,
    nyquist_grid_40,
    roc_curves_figure,
)


@pytest.fixture
def mini_nyquist_df() -> pd.DataFrame:
    """Small but realistic DataFrame covering every column the plots read."""
    rng = np.random.default_rng(0)
    rows = []
    for aging in [0, 1, 2, 3, 4]:
        for temp in [20.0, 30.0, 40.0]:
            for soc in [0, 1, 2]:
                for freq in np.logspace(-1, 4, 6):
                    rows.append(
                        {
                            "Aging": aging,
                            "Temperature": temp,
                            "SOC": soc,
                            "Frequency": float(freq),
                            "Z_real": float(rng.uniform(10, 40)),
                            "Z_imag": float(rng.uniform(-15, 5)),
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def prediction_df(mini_nyquist_df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe augmented with Z_real_pred / Z_imag_pred (needed by the grid)."""
    df = mini_nyquist_df.copy()
    df["Z_real_pred"] = df["Z_real"] + 0.5
    df["Z_imag_pred"] = df["Z_imag"] - 0.5
    return df


# ---------------------------------------------------------------------------
# nyquist_by_soc
# ---------------------------------------------------------------------------


def test_nyquist_by_soc_returns_figure(mini_nyquist_df: pd.DataFrame) -> None:
    sub = mini_nyquist_df[
        (mini_nyquist_df["Aging"] == 0) & (mini_nyquist_df["Temperature"] == 20.0)
    ]
    fig = nyquist_by_soc(sub, title="Test")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test"
    assert fig.layout.xaxis.title.text == "Re(Z) [mΩ]"
    assert fig.layout.yaxis.title.text == "-Im(Z) [mΩ]"
    assert len(fig.data) == sub["SOC"].nunique()


def test_nyquist_by_soc_uses_negative_imaginary(mini_nyquist_df: pd.DataFrame) -> None:
    sub = mini_nyquist_df[
        (mini_nyquist_df["Aging"] == 0)
        & (mini_nyquist_df["Temperature"] == 20.0)
        & (mini_nyquist_df["SOC"] == 0)
    ]
    fig = nyquist_by_soc(sub, title="Sign check")
    raw = sub.sort_values("Frequency", ascending=False)["Z_imag"].to_numpy()
    plotted = np.asarray(fig.data[0].y)
    np.testing.assert_allclose(plotted, -raw)


# ---------------------------------------------------------------------------
# actual_vs_predicted_nyquist
# ---------------------------------------------------------------------------


def test_actual_vs_predicted_two_traces_per_soc(mini_nyquist_df: pd.DataFrame) -> None:
    sub = (
        mini_nyquist_df[
            (mini_nyquist_df["Aging"] == 0) & (mini_nyquist_df["Temperature"] == 20.0)
        ]
        .reset_index(drop=True)
    )
    y_pred = sub[["Z_real", "Z_imag"]].to_numpy() + 1.0
    fig = actual_vs_predicted_nyquist(sub, y_pred, title="Overlay")
    assert isinstance(fig, go.Figure)
    # 2 traces (actual + predicted) per SOC present in the fixture
    assert len(fig.data) == 2 * sub["SOC"].nunique()
    names = [tr.name for tr in fig.data]
    assert any("actual" in n for n in names)
    assert any("predicted" in n for n in names)


# ---------------------------------------------------------------------------
# metric_bar_chart
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_key", ["R2", "MSE", "Accuracy"])
def test_metric_bar_chart_sorts_and_labels(metric_key: str) -> None:
    metrics = {
        "A": {"R2": 0.9, "MSE": 0.1, "Accuracy": 0.95},
        "B": {"R2": 0.7, "MSE": 0.3, "Accuracy": 0.80},
        "C": {"R2": 0.85, "MSE": 0.2, "Accuracy": 0.90},
    }
    fig = metric_bar_chart(metrics, metric_key, title=f"Ordered by {metric_key}")
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == metric_key
    bar = fig.data[0]
    values = list(bar.x)
    assert values == sorted(values, reverse=True)
    assert len(bar.y) == 3


# ---------------------------------------------------------------------------
# nyquist_grid_40
# ---------------------------------------------------------------------------


def test_nyquist_grid_adapts_to_input_shape(prediction_df: pd.DataFrame) -> None:
    sub = prediction_df[prediction_df["Aging"] == 0]
    n_temp = sub["Temperature"].nunique()
    n_soc = sub["SOC"].nunique()
    fig = nyquist_grid_40(sub, title="Grid")
    # Each cell gets 2 traces (actual + predicted); only non-empty cells
    # produce traces — the fixture fully populates the grid.
    assert len(fig.data) == 2 * n_temp * n_soc
    assert fig.layout.title.text == "Grid"


def test_nyquist_grid_uses_negative_imaginary(prediction_df: pd.DataFrame) -> None:
    sub = prediction_df[
        (prediction_df["Aging"] == 0) & (prediction_df["Temperature"] == 20.0)
    ]
    fig = nyquist_grid_40(sub, title="Grid sign check")
    first = sub[sub["SOC"] == 0].sort_values("Frequency", ascending=False)
    np.testing.assert_allclose(
        np.asarray(fig.data[0].y), -first["Z_imag"].to_numpy()
    )


# ---------------------------------------------------------------------------
# nyquist_classification_map
# ---------------------------------------------------------------------------


def test_classification_map_two_traces(mini_nyquist_df: pd.DataFrame) -> None:
    y_pred = np.zeros(len(mini_nyquist_df), dtype=int)
    y_pred[: len(y_pred) // 2] = 1
    fig = nyquist_classification_map(mini_nyquist_df, y_pred)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    trace_names = {tr.name for tr in fig.data}
    assert trace_names == {f"Pred: {CLASS_NAMES[0]}", f"Pred: {CLASS_NAMES[1]}"}


# ---------------------------------------------------------------------------
# confusion_matrix_figure
# ---------------------------------------------------------------------------


def test_confusion_matrix_figure_content() -> None:
    cm = np.array([[10, 2], [3, 15]])
    fig = confusion_matrix_figure(cm, CLASS_NAMES)
    assert isinstance(fig, go.Figure)
    heatmap = fig.data[0]
    np.testing.assert_array_equal(np.asarray(heatmap.z), cm)
    assert list(heatmap.x) == CLASS_NAMES
    assert list(heatmap.y) == CLASS_NAMES
    assert fig.layout.xaxis.title.text == "Predicted"
    assert fig.layout.yaxis.title.text == "Actual"


# ---------------------------------------------------------------------------
# roc_curves_figure
# ---------------------------------------------------------------------------


def test_roc_curves_trace_count() -> None:
    rng = np.random.default_rng(1)
    y_test = rng.integers(0, 2, size=120)
    proba = {
        "M1": rng.uniform(0, 1, size=120),
        "M2": rng.uniform(0, 1, size=120),
    }
    fig = roc_curves_figure(proba, y_test)
    assert isinstance(fig, go.Figure)
    # 2 models + 1 diagonal baseline
    assert len(fig.data) == 3
    named = [tr for tr in fig.data if tr.showlegend is not False]
    assert {"M1", "M2"}.issubset({tr.name.split(" ")[0] for tr in named})


# ---------------------------------------------------------------------------
# nyquist_aging_evolution
# ---------------------------------------------------------------------------


def test_nyquist_aging_evolution_structure(mini_nyquist_df: pd.DataFrame) -> None:
    fig = nyquist_aging_evolution(mini_nyquist_df, soc=0, excluded_aging=2)
    assert isinstance(fig, go.Figure)
    assert "Aging 2 dashed" in fig.layout.title.text
    dashed = [tr for tr in fig.data if tr.line.dash == "dash"]
    assert len(dashed) > 0
