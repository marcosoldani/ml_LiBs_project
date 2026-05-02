"""Task 3 — Leave-One-Aging-Out interpolation.

Hold out an entire aging level (default: Aging 2, the central one) and
predict all 40 Nyquist plots (8 Temperatures × 5 SOC) from the remaining
four aging levels. Harder than Task 1 because no point at the held-out
aging is ever visible at training time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import load_config, models_path
from src.data.features import build_regression_features
from src.data.loader import load_dataset
from src.evaluation.metrics import regression_metrics
from src.logger import get_logger
from src.models.registry import regression_models
from src.tracking import log_model_run, track_pipeline

logger = get_logger(__name__)

TASK_DIR = models_path() / "task3"


@dataclass
class AgingInterpResult:
    excluded_aging: int
    best_model_name: str
    metrics_per_model: dict[str, dict[str, float]]
    predictions: dict[str, np.ndarray]
    y_test: np.ndarray
    df_test: pd.DataFrame
    training_time: dict[str, float]
    per_temperature: pd.DataFrame


def run_aging_interpolation(
    excluded_aging: int | None = None,
    df: pd.DataFrame | None = None,
    models: dict[str, Any] | None = None,
) -> AgingInterpResult:
    """Train on 4 aging levels, predict all Nyquist plots of the held-out one."""
    if excluded_aging is None:
        excluded_aging = int(load_config().task3.default_aging)
    df = df if df is not None else load_dataset()
    models = models if models is not None else regression_models()

    train_mask = df["Aging"] != excluded_aging
    test_mask = df["Aging"] == excluded_aging
    if not test_mask.any():
        raise ValueError(f"No rows for Aging={excluded_aging}")

    X_train = build_regression_features(df[train_mask])
    X_test = build_regression_features(df[test_mask])
    y_train = df.loc[train_mask, ["Z_real", "Z_imag"]].values
    y_test = df.loc[test_mask, ["Z_real", "Z_imag"]].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)

    metrics_per_model: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    logger.info(
        "Task 3 Leave-One-Aging-Out: aging=%d train=%d test=%d",
        excluded_aging,
        int(train_mask.sum()),
        int(test_mask.sum()),
    )

    with track_pipeline(
        task="task3",
        run_name=f"task3-loao-a{excluded_aging}",
        params={
            "excluded_aging": excluded_aging,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_models": len(models),
        },
    ):
        for name, estimator in models.items():
            t0 = time.time()
            estimator.fit(X_train_s, y_train_s)
            y_pred = scaler_y.inverse_transform(estimator.predict(X_test_s))
            timings[name] = time.time() - t0
            predictions[name] = y_pred
            metrics_per_model[name] = regression_metrics(y_test, y_pred).as_dict()
            log_model_run(
                name=name,
                params=getattr(estimator, "get_params", lambda: {})(),
                metrics=metrics_per_model[name],
                duration_s=timings[name],
            )
            logger.info(
                "  %-20s R²=%.4f MSE=%.4f time=%.2fs",
                name,
                metrics_per_model[name]["R2"],
                metrics_per_model[name]["MSE"],
                timings[name],
            )

    best = max(metrics_per_model, key=lambda n: metrics_per_model[n]["R2"])
    df_test = df.loc[test_mask].reset_index(drop=True).copy()
    df_test["Z_real_pred"] = predictions[best][:, 0]
    df_test["Z_imag_pred"] = predictions[best][:, 1]

    per_temp_rows = []
    for temp in sorted(df_test["Temperature"].unique()):
        sub = df_test[df_test["Temperature"] == temp]
        per_temp_rows.append(
            {
                "Temperature": float(temp),
                "R2": float(
                    regression_metrics(
                        sub[["Z_real", "Z_imag"]].values,
                        sub[["Z_real_pred", "Z_imag_pred"]].values,
                    ).r2
                ),
                "MAE": float(
                    regression_metrics(
                        sub[["Z_real", "Z_imag"]].values,
                        sub[["Z_real_pred", "Z_imag_pred"]].values,
                    ).mae
                ),
                "N": int(len(sub)),
            }
        )
    per_temp_df = pd.DataFrame(per_temp_rows)

    logger.info("  → Best model: %s (R²=%.4f)", best, metrics_per_model[best]["R2"])

    return AgingInterpResult(
        excluded_aging=excluded_aging,
        best_model_name=best,
        metrics_per_model=metrics_per_model,
        predictions=predictions,
        y_test=y_test,
        df_test=df_test,
        training_time=timings,
        per_temperature=per_temp_df,
    )


def persist_default_benchmark(out_dir: Path | None = None) -> Path:
    """Benchmark with Aging=2 (default) and persist metrics."""
    cfg_dir = out_dir or TASK_DIR
    cfg_dir.mkdir(parents=True, exist_ok=True)
    result = run_aging_interpolation()
    payload = {
        "excluded_aging": result.excluded_aging,
        "best_model": result.best_model_name,
        "metrics": result.metrics_per_model,
        "training_time_s": result.training_time,
        "per_temperature": result.per_temperature.to_dict(orient="records"),
    }
    benchmark_file = cfg_dir / "benchmark.json"
    benchmark_file.write_text(json.dumps(payload, indent=2))
    joblib.dump({"result": result}, cfg_dir / "last_run.joblib")
    logger.info("Task 3 benchmark saved → %s", benchmark_file)
    return benchmark_file
