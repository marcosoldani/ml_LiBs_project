"""Plotly figures used by the React frontend (via the FastAPI bridge).

Pure functions: they take model outputs / DataFrames and return a
``plotly.graph_objects.Figure``. Keeping plotting separate from model
training lets the UI redraw results cheaply without retraining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.constants import (
    AGING_COLORS,
    AGING_LABELS,
    CLASS_COLORS,
    CLASS_NAMES,
    SOC_LABELS,
    SOC_MARKERS,
    TEMP_COLORS,
)


def _soc_marker(soc: int) -> str:
    return SOC_MARKERS.get(int(soc), "circle")


def nyquist_by_soc(
    df: pd.DataFrame,
    title: str,
    height: int = 480,
    symbol_by_soc: bool = True,
    line: bool = True,
) -> go.Figure:
    """Single Nyquist plot split by SOC from a (Freq × rows) DataFrame."""
    fig = go.Figure()
    for soc in sorted(df["SOC"].unique()):
        sub = df[df["SOC"] == soc].sort_values("Frequency", ascending=False)
        fig.add_trace(
            go.Scatter(
                x=sub["Z_real"],
                y=-sub["Z_imag"],
                mode="lines+markers" if line else "markers",
                name=SOC_LABELS.get(int(soc), f"SOC {soc}"),
                marker={
                    "size": 7,
                    "symbol": _soc_marker(int(soc)) if symbol_by_soc else "circle",
                },
                line={"width": 1.5},
                hovertemplate=(
                    "Re(Z)=%{x:.3f} mΩ<br>-Im(Z)=%{y:.3f} mΩ<br>"
                    f"SOC={soc}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Re(Z) [mΩ]",
        yaxis_title="-Im(Z) [mΩ]",
        height=height,
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="plotly_white",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def actual_vs_predicted_nyquist(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    title: str,
    height: int = 500,
) -> go.Figure:
    """Overlay of ground-truth and model prediction, split by SOC."""
    fig = go.Figure()
    for soc in sorted(df_test["SOC"].unique()):
        mask = df_test["SOC"].values == soc
        idx = df_test[mask].sort_values("Frequency", ascending=False).index
        z_real_true = df_test.loc[idx, "Z_real"].values
        z_imag_true = df_test.loc[idx, "Z_imag"].values
        z_real_pred = y_pred[idx.values, 0]
        z_imag_pred = y_pred[idx.values, 1]

        fig.add_trace(
            go.Scatter(
                x=z_real_true,
                y=-z_imag_true,
                mode="lines+markers",
                name=f"SOC {soc} — actual",
                legendgroup=f"soc-{soc}",
                line={"width": 2},
                marker={"size": 6, "symbol": _soc_marker(int(soc))},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=z_real_pred,
                y=-z_imag_pred,
                mode="lines+markers",
                name=f"SOC {soc} — predicted",
                legendgroup=f"soc-{soc}",
                line={"width": 2, "dash": "dash"},
                marker={"size": 6, "symbol": "x"},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Re(Z) [mΩ]",
        yaxis_title="-Im(Z) [mΩ]",
        height=height,
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="plotly_white",
        legend={"orientation": "v"},
    )
    return fig


def metric_bar_chart(metrics: dict[str, dict[str, float]], metric_key: str, title: str) -> go.Figure:
    """Horizontal bar chart ordering models by the chosen metric."""
    items = sorted(metrics.items(), key=lambda kv: kv[1][metric_key], reverse=True)
    names = [name for name, _ in items]
    values = [m[metric_key] for _, m in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            marker={"color": "#2E86C1"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=metric_key,
        yaxis={"autorange": "reversed"},
        template="plotly_white",
        height=380,
    )
    return fig


def nyquist_grid_40(df_test: pd.DataFrame, title: str) -> go.Figure:
    """Nyquist grid used for Task 3, one subplot per (Temperature, SOC).

    Grid dimensions adapt to the distinct values present in ``df_test`` so the
    function keeps working if the dataset schema changes.
    Note: plots use ``-Im(Z)`` on the Y axis (standard EIS convention).
    """
    temps = sorted(df_test["Temperature"].unique())
    socs = sorted(df_test["SOC"].unique())
    fig = make_subplots(
        rows=len(temps),
        cols=len(socs),
        subplot_titles=[
            f"T={t}°C · SOC={s}" for t in temps for s in socs
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
    )
    for i, temp in enumerate(temps, start=1):
        color = TEMP_COLORS.get(float(temp), "#555555")
        for j, soc in enumerate(socs, start=1):
            sub = df_test[
                (df_test["Temperature"] == temp) & (df_test["SOC"] == soc)
            ].sort_values("Frequency", ascending=False)
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["Z_real"],
                    y=-sub["Z_imag"],
                    mode="lines+markers",
                    name="actual",
                    line={"color": color, "width": 2},
                    marker={"size": 4},
                    showlegend=False,
                ),
                row=i,
                col=j,
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["Z_real_pred"],
                    y=-sub["Z_imag_pred"],
                    mode="lines+markers",
                    name="pred",
                    line={"color": "black", "dash": "dash", "width": 1.5},
                    marker={"size": 4, "symbol": "x"},
                    showlegend=False,
                ),
                row=i,
                col=j,
            )
    fig.update_layout(
        title=title,
        height=180 * len(temps),
        template="plotly_white",
        margin={"t": 80},
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = {"size": 10}
    fig.update_xaxes(showticklabels=True, tickfont={"size": 8})
    fig.update_yaxes(showticklabels=True, tickfont={"size": 8})
    return fig


def nyquist_classification_map(df: pd.DataFrame, y_pred: np.ndarray) -> go.Figure:
    """Nyquist plot for Task 2 colored by predicted class."""
    fig = go.Figure()
    for cls, label, color in zip([0, 1], CLASS_NAMES, CLASS_COLORS, strict=False):
        mask = y_pred == cls
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "Z_real"],
                y=-df.loc[mask, "Z_imag"],
                mode="markers",
                name=f"Pred: {label}",
                marker={"color": color, "size": 5, "opacity": 0.55},
            )
        )
    fig.update_layout(
        title="Predicted classes on the Nyquist plane",
        xaxis_title="Re(Z) [mΩ]",
        yaxis_title="-Im(Z) [mΩ]",
        template="plotly_white",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def confusion_matrix_figure(cm: np.ndarray, labels: list[str]) -> go.Figure:
    """Blue heatmap confusion matrix with overlaid counts."""
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=380,
        yaxis={"autorange": "reversed"},
    )
    return fig


def roc_curves_figure(probabilities: dict[str, np.ndarray], y_test: np.ndarray) -> go.Figure:
    """Overlay ROC curves for every model with a diagonal baseline."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fig = go.Figure()
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC={auc:.3f})",
                line={"width": 2},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#aaaaaa"},
            name="Random",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=420,
    )
    return fig


def nyquist_aging_evolution(df: pd.DataFrame, soc: int, excluded_aging: int) -> go.Figure:
    """Task 3 EDA — impedance curves vs aging at several temperatures."""
    temps = [20.0, 30.0, 40.0]
    fig = make_subplots(
        rows=1,
        cols=len(temps),
        subplot_titles=[f"T={t}°C" for t in temps],
        shared_yaxes=True,
    )
    for col, temp in enumerate(temps, start=1):
        for aging in sorted(df["Aging"].unique()):
            sub = df[
                (df["Aging"] == aging)
                & (df["Temperature"] == temp)
                & (df["SOC"] == soc)
            ].sort_values("Frequency", ascending=False)
            if sub.empty:
                continue
            dash = "dash" if aging == excluded_aging else "solid"
            fig.add_trace(
                go.Scatter(
                    x=sub["Z_real"],
                    y=-sub["Z_imag"],
                    mode="lines+markers",
                    name=AGING_LABELS.get(int(aging), f"Aging {aging}"),
                    legendgroup=str(aging),
                    showlegend=(col == 1),
                    line={
                        "color": AGING_COLORS.get(int(aging), "#333"),
                        "width": 2.5 if aging == excluded_aging else 1.8,
                        "dash": dash,
                    },
                    marker={"size": 5},
                ),
                row=1,
                col=col,
            )
    fig.update_layout(
        title=f"Impedance evolution across aging (SOC={soc}) — Aging {excluded_aging} dashed",
        template="plotly_white",
        height=420,
    )
    fig.update_xaxes(title_text="Re(Z) [mΩ]")
    fig.update_yaxes(title_text="-Im(Z) [mΩ]", col=1)
    return fig
