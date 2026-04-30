"""Task 1 — Leave-One-Out curve reconstruction.

Given an (Aging, Temperature) pair held out, train on the remaining 39
combinations and predict the 40th Nyquist plot (5 SOC × frequency points).
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

TASK_DIR = models_path() / "task1"


@dataclass
class LOOResult:
    model_name: str
    excluded_aging: int
    excluded_temperature: float
    metrics_per_model: dict[str, dict[str, float]]
    best_model_name: str
    best_predictions: np.ndarray
    y_test: np.ndarray
    df_test: pd.DataFrame
    training_time: dict[str, float]


def split_dataset(
    df: pd.DataFrame, aging: int, temperature: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) splits with the given (Aging, Temp) as the test fold."""
    mask = (df["Aging"] == aging) & (df["Temperature"] == temperature)
    return df.loc[~mask].copy(), df.loc[mask].copy()


def run_leave_one_out(
    aging: int,
    temperature: float,
    df: pd.DataFrame | None = None,
    models: dict[str, Any] | None = None,
) -> LOOResult:
    """Train all baseline regression models on 39 combos, predict the 40th."""
    df = df if df is not None else load_dataset()
    models = models if models is not None else regression_models()

    df_train, df_test = split_dataset(df, aging, temperature)
    if df_test.empty:
        raise ValueError(
            f"No data for Aging={aging}, Temperature={temperature}°C."
        )

    X_train = build_regression_features(df_train)
    y_train = df_train[["Z_real", "Z_imag"]].values
    X_test = build_regression_features(df_test)
    y_test = df_test[["Z_real", "Z_imag"]].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)

    metrics_per_model: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    logger.info(
        "Task 1 LOO: Aging=%d Temp=%.1f°C train=%d test=%d",
        aging,
        temperature,
        len(df_train),
        len(df_test),
    )

    with track_pipeline(
        task="task1",
        run_name=f"task1-loo-a{aging}-t{temperature}",
        params={
            "aging": aging,
            "temperature": temperature,
            "n_train": len(df_train),
            "n_test": len(df_test),
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

    best = min(metrics_per_model, key=lambda n: metrics_per_model[n]["MSE"])
    logger.info("  → Best model: %s", best)

    return LOOResult(
        model_name=best,
        excluded_aging=aging,
        excluded_temperature=temperature,
        metrics_per_model=metrics_per_model,
        best_model_name=best,
        best_predictions=predictions[best],
        y_test=y_test,
        df_test=df_test.reset_index(drop=True),
        training_time=timings,
    )


def persist_default_benchmark(out_dir: Path | None = None) -> Path:
    """Train the default benchmark (from config.yaml) and save metrics to disk."""
    cfg_dir = out_dir or TASK_DIR
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    result = run_leave_one_out(
        aging=int(cfg.task1.default_aging),
        temperature=float(cfg.task1.default_temperature),
    )

    payload = {
        "excluded_aging": result.excluded_aging,
        "excluded_temperature": result.excluded_temperature,
        "best_model": result.best_model_name,
        "metrics": result.metrics_per_model,
        "training_time_s": result.training_time,
    }
    benchmark_file = cfg_dir / "benchmark.json"
    benchmark_file.write_text(json.dumps(payload, indent=2))

    # Save a reusable scaler/model pair for the best model
    joblib.dump(
        {
            "result": result,
        },
        cfg_dir / "last_run.joblib",
    )
    logger.info("Task 1 benchmark saved → %s", benchmark_file)
    return benchmark_file
