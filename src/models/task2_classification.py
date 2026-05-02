"""Task 2 — Young vs Old classification on a held-out (Aging, SOC).

The user picks an (Aging, SOC) pair: every row of that combination (8
temperatures × ~49 frequencies ≈ 392 rows) is removed from training. The
classifier learns from the remaining ~9 400 rows and must predict whether
the held-out curves belong to a Young (Aging ≤ 2) or Old (Aging ≥ 3) cell.

Aging is **never** used as a feature — it defines the target. The model
only sees impedance shape (Re(Z), Im(Z), magnitude, phase, Temperature,
Frequency, …), so it cannot trivially recover the aging from the input.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import load_config, models_path
from src.data.features import build_classification_features
from src.data.loader import load_dataset
from src.evaluation.metrics import classification_metrics
from src.logger import get_logger
from src.models.registry import classification_models
from src.tracking import log_model_run, track_pipeline

logger = get_logger(__name__)

TASK_DIR = models_path() / "task2"


def _young_max_aging() -> int:
    """Highest aging level still labelled as ``Young`` (read from config.yaml)."""
    return int(load_config().task2.young_max_aging)


@dataclass
class ClassificationResult:
    aging: int
    soc: int
    true_class: int
    true_label: str
    best_model_name: str
    metrics_per_model: dict[str, dict[str, float]]
    predictions: dict[str, np.ndarray]
    probabilities: dict[str, np.ndarray]
    df_test: pd.DataFrame
    y_test: np.ndarray
    training_time: dict[str, float]
    feature_importance: dict[str, float] | None
    per_temperature: pd.DataFrame


def build_labels(df: pd.DataFrame, young_max_aging: int | None = None) -> pd.DataFrame:
    """Attach Age_class (0 = Young, 1 = Old) to the DataFrame."""
    threshold = _young_max_aging() if young_max_aging is None else int(young_max_aging)
    df = df.copy()
    df["Age_class"] = (df["Aging"] > threshold).astype(int)
    df["Age_label"] = df["Age_class"].map({0: "Young", 1: "Old"})
    return df


def run_classification(
    aging: int,
    soc: int,
    df: pd.DataFrame | None = None,
    models: dict[str, Any] | None = None,
) -> ClassificationResult:
    """Hold out (Aging, SOC) and predict its class from impedance shape."""
    df = df if df is not None else load_dataset()
    df = build_labels(df)

    test_mask = (df["Aging"] == aging) & (df["SOC"] == soc)
    if not test_mask.any():
        raise ValueError(f"No rows for Aging={aging}, SOC={soc}")
    train_mask = ~test_mask

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_test = df.loc[test_mask].reset_index(drop=True)

    X_train_raw = build_classification_features(df_train)
    X_test_raw = build_classification_features(df_test)
    y_train = df_train["Age_class"].values
    y_test = df_test["Age_class"].values
    true_class = int(y_test[0])
    true_label = "Young" if true_class == 0 else "Old"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    models = models if models is not None else classification_models()
    metrics_per_model: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    logger.info(
        "Task 2 classification: held-out (Aging=%d, SOC=%d) → true class=%s, "
        "train=%d test=%d",
        aging, soc, true_label, len(df_train), len(df_test),
    )

    with track_pipeline(
        task="task2",
        run_name=f"task2-cls-a{aging}-s{soc}",
        params={
            "aging": aging,
            "soc": soc,
            "true_class": true_class,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "n_models": len(models),
        },
    ):
        for name, estimator in models.items():
            t0 = time.time()
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            if hasattr(estimator, "predict_proba"):
                y_proba = estimator.predict_proba(X_test)[:, 1]
            else:
                y_proba = estimator.decision_function(X_test)
            timings[name] = time.time() - t0
            predictions[name] = y_pred
            probabilities[name] = y_proba

            # On a single-class test fold AUC is undefined; we keep accuracy
            # (and a degenerate F1) but skip ROC entirely in the payload.
            try:
                m = classification_metrics(y_test, y_pred, y_proba).as_dict()
            except Exception:
                m = {
                    "Accuracy": float(np.mean(y_pred == y_test)),
                    "F1": float("nan"),
                    "AUC": float("nan"),
                }
            metrics_per_model[name] = m
            log_model_run(
                name=name,
                params=getattr(estimator, "get_params", lambda: {})(),
                metrics={k: v for k, v in m.items() if v == v},  # drop NaN
                duration_s=timings[name],
            )
            logger.info(
                "  %-20s Acc=%.4f time=%.2fs",
                name,
                metrics_per_model[name]["Accuracy"],
                timings[name],
            )

    best = max(metrics_per_model, key=lambda n: metrics_per_model[n]["Accuracy"])
    best_pred = predictions[best]

    fi: dict[str, float] | None = None
    best_est = models[best]
    if hasattr(best_est, "feature_importances_"):
        fi = {
            name: float(v)
            for name, v in zip(
                X_train_raw.columns, best_est.feature_importances_, strict=False
            )
        }

    # Per-temperature accuracy on the held-out (Aging, SOC)
    rows = []
    for temp in sorted(df_test["Temperature"].unique()):
        m = (df_test["Temperature"] == temp).values
        n = int(m.sum())
        n_corr = int(np.sum(best_pred[m] == y_test[m]))
        n_young = int(np.sum(best_pred[m] == 0))
        n_old = int(np.sum(best_pred[m] == 1))
        rows.append(
            {
                "Temperature": float(temp),
                "N": n,
                "N_correct": n_corr,
                "Accuracy": (n_corr / n) if n else 0.0,
                "N_predicted_young": n_young,
                "N_predicted_old": n_old,
            }
        )
    per_temp_df = pd.DataFrame(rows)

    logger.info(
        "  → Best model: %s (Acc=%.4f, true=%s)",
        best, metrics_per_model[best]["Accuracy"], true_label,
    )

    return ClassificationResult(
        aging=aging,
        soc=soc,
        true_class=true_class,
        true_label=true_label,
        best_model_name=best,
        metrics_per_model=metrics_per_model,
        predictions=predictions,
        probabilities=probabilities,
        df_test=df_test,
        y_test=y_test,
        training_time=timings,
        feature_importance=fi,
        per_temperature=per_temp_df,
    )


def persist_default_benchmark(out_dir: Path | None = None) -> Path:
    """Benchmark on the default (Aging, SOC) pair from ``config.yaml``.

    Run a single hold-out — the (``task2.default_aging``,
    ``task2.default_soc``) pair declared in config — to keep the
    pipeline fast (~3 s vs ~15 min for the full 5×5 grid). Use
    :func:`benchmark_full_grid` if you need the cross-pair heatmap.
    """
    cfg = load_config().task2
    aging = int(cfg.default_aging)
    soc = int(cfg.default_soc)
    cfg_dir = out_dir or TASK_DIR
    cfg_dir.mkdir(parents=True, exist_ok=True)

    res = run_classification(aging=aging, soc=soc)
    payload = {
        "aging": aging,
        "soc": soc,
        "true_class": res.true_class,
        "true_label": res.true_label,
        "best_model": res.best_model_name,
        "metrics": res.metrics_per_model,
        "training_time_s": res.training_time,
    }
    benchmark_file = cfg_dir / "benchmark.json"
    benchmark_file.write_text(json.dumps(payload, indent=2))
    logger.info("Task 2 benchmark saved → %s", benchmark_file)
    return benchmark_file


def benchmark_full_grid(
    out_dir: Path | None = None,
    df: pd.DataFrame | None = None,
) -> Path:
    """Run all 25 hold-out combinations (Aging × SOC) — slow but exhaustive.

    Used to feed ``task2_cross_pair_heatmap.pdf`` in the report; not
    invoked by ``scripts.train_all`` to keep the default pipeline fast.
    Accepts an optional ``df`` so tests can pass a slimmer slice.
    """
    cfg_dir = out_dir or TASK_DIR
    cfg_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for aging in range(5):
        for soc in range(5):
            try:
                res = run_classification(aging=aging, soc=soc, df=df)
            except ValueError:
                continue
            summary[f"a{aging}_s{soc}"] = {
                "aging": aging,
                "soc": soc,
                "true_class": res.true_class,
                "true_label": res.true_label,
                "best_model": res.best_model_name,
                "metrics": res.metrics_per_model,
                "training_time_s": res.training_time,
            }
    out_file = cfg_dir / "benchmark_full_grid.json"
    out_file.write_text(json.dumps(summary, indent=2))
    logger.info("Task 2 full-grid benchmark saved → %s", out_file)
    return out_file


__all__ = [
    "ClassificationResult",
    "benchmark_full_grid",
    "build_labels",
    "persist_default_benchmark",
    "run_classification",
]
