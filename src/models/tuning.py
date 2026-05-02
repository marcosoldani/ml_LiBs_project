"""Hyperparameter tuning via GridSearchCV with task-coherent splits.

Each ``tune_*`` function returns a dict ``{model_name: {best_params, best_score}}``
and persists it to ``models/tuning/<task>_best.json``. The same results are
logged as MLflow nested runs under an experiment called
``battery-geis-<task>-tuning``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PROJECT_ROOT, load_config
from src.data.features import (
    build_classification_features,
    build_regression_features,
)
from src.data.loader import load_dataset
from src.logger import get_logger
from src.models.task2_classification import build_labels
from src.tracking import log_model_run, track_pipeline

logger = get_logger(__name__)

RANDOM_SEED = 42


def _out_dir() -> Path:
    cfg = load_config()
    path = PROJECT_ROOT / cfg.tuning.out_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prefix(grid: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in grid.items()}


def _persist(task: str, payload: dict[str, Any]) -> Path:
    f = _out_dir() / f"{task}_best.json"
    f.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Tuning results saved → %s", f)
    return f


# ---------------------------------------------------------------------------
# Task 1 + Task 3 — regression tuning
# ---------------------------------------------------------------------------


def _regression_candidates(cfg_task: Any) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Estimator + param grid per tunable regressor."""
    rf = cfg_task.random_forest
    gb = cfg_task.gradient_boosting
    knn = cfg_task.knn
    return {
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            {
                "n_estimators": list(rf.n_estimators),
                "max_depth": list(rf.max_depth),
            },
        ),
        "Gradient Boosting": (
            MultiOutputRegressor(
                GradientBoostingRegressor(random_state=RANDOM_SEED)
            ),
            _prefix(
                {
                    "n_estimators": list(gb.n_estimators),
                    "max_depth": list(gb.max_depth),
                    "learning_rate": list(gb.learning_rate),
                },
                prefix="estimator__",
            ),
        ),
        "K-Nearest Neighbors": (
            KNeighborsRegressor(n_jobs=-1),
            {
                "n_neighbors": list(knn.n_neighbors),
                "weights": list(knn.weights),
            },
        ),
    }


def _run_regression_search(
    task: str,
    cv_splitter: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
) -> dict[str, dict[str, Any]]:
    cfg = load_config()
    cfg_task = getattr(cfg.tuning, task)
    results: dict[str, dict[str, Any]] = {}
    run_name = f"{task}-tuning"

    with track_pipeline(
        task=f"{task}-tuning",
        run_name=run_name,
        params={"cv_folds": int(cfg.tuning.cv_folds), "task": task},
    ):
        for name, (est, grid) in _regression_candidates(cfg_task).items():
            logger.info("[%s] GridSearchCV → %s (%d combos)", task, name, int(np.prod([len(v) for v in grid.values()])))
            t0 = time.time()
            search = GridSearchCV(
                estimator=est,
                param_grid=grid,
                scoring="neg_mean_squared_error",
                cv=cv_splitter,
                n_jobs=-1,
                refit=True,
            )
            if groups is not None:
                search.fit(X, y, groups=groups)
            else:
                search.fit(X, y)
            dt = time.time() - t0
            best_params = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
            results[name] = {
                "best_params": best_params,
                "best_score_neg_mse": float(search.best_score_),
                "best_score_mse": float(-search.best_score_),
                "duration_s": dt,
            }
            log_model_run(
                name=name,
                params=best_params,
                metrics={"best_mse": -search.best_score_},
                duration_s=dt,
            )
            logger.info(
                "  → best MSE=%.4f params=%s time=%.1fs",
                -search.best_score_,
                best_params,
                dt,
            )

    return results


def tune_task1(df: pd.DataFrame | None = None) -> Path:
    """Tune Task 1 using Leave-One-Group-Out on (Aging, Temperature)."""
    df = df if df is not None else load_dataset()
    X = build_regression_features(df).values
    y = df[["Z_real", "Z_imag"]].values
    groups = (df["Aging"].astype(str) + "_" + df["Temperature"].astype(str)).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    splitter = LeaveOneGroupOut()
    # Subsample groups to stay under a few minutes: pass only the first N groups.
    unique = pd.unique(groups)
    if len(unique) > 10:
        keep = set(unique[:10])
        mask = np.array([g in keep for g in groups])
        X, y, groups = X[mask], y[mask], groups[mask]
        logger.info("Task 1 tuning subsampled to %d (Aging,Temp) groups", len(keep))

    results = _run_regression_search("task1", splitter, X, y, groups)
    return _persist("task1", results)


def tune_task3(df: pd.DataFrame | None = None) -> Path:
    """Tune Task 3 using Leave-One-Group-Out on Aging."""
    df = df if df is not None else load_dataset()
    X = build_regression_features(df).values
    y = df[["Z_real", "Z_imag"]].values
    groups = df["Aging"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    splitter = LeaveOneGroupOut()
    results = _run_regression_search("task3", splitter, X, y, groups)
    return _persist("task3", results)


# ---------------------------------------------------------------------------
# Task 2 — classification tuning
# ---------------------------------------------------------------------------


def _classification_candidates(cfg_task: Any) -> dict[str, tuple[Any, dict[str, Any]]]:
    rf = cfg_task.random_forest
    gb = cfg_task.gradient_boosting
    svm = cfg_task.svm
    return {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            {
                "n_estimators": list(rf.n_estimators),
                "max_depth": list(rf.max_depth),
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {
                "n_estimators": list(gb.n_estimators),
                "max_depth": list(gb.max_depth),
                "learning_rate": list(gb.learning_rate),
            },
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
            {
                "C": list(svm.C),
                "gamma": list(svm.gamma),
            },
        ),
    }


def tune_task2(soc: int | None = None, df: pd.DataFrame | None = None) -> Path:
    """Tune Task 2 using StratifiedGroupKFold on the default SOC."""
    cfg = load_config()
    if soc is None:
        soc = int(cfg.task2.default_soc)
    df = df if df is not None else load_dataset()
    df = build_labels(df)
    df_soc = df[df["SOC"] == soc].reset_index(drop=True)

    X_raw = build_classification_features(df_soc).values
    y = df_soc["Age_class"].values
    groups = (
        df_soc["Aging"].astype(str) + "_" + df_soc["Temperature"].astype(str)
    ).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    cfg_task = cfg.tuning.task2
    cv = StratifiedGroupKFold(n_splits=int(cfg.tuning.cv_folds), shuffle=True, random_state=RANDOM_SEED)
    results: dict[str, dict[str, Any]] = {}

    with track_pipeline(
        task="task2-tuning",
        run_name=f"task2-tuning-soc{soc}",
        params={"soc": soc, "cv_folds": int(cfg.tuning.cv_folds)},
    ):
        for name, (est, grid) in _classification_candidates(cfg_task).items():
            logger.info("[task2] GridSearchCV → %s (%d combos)", name, int(np.prod([len(v) for v in grid.values()])))
            t0 = time.time()
            search = GridSearchCV(
                estimator=est,
                param_grid=grid,
                scoring="f1_weighted",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y, groups=groups)
            dt = time.time() - t0
            results[name] = {
                "best_params": search.best_params_,
                "best_score_f1": float(search.best_score_),
                "duration_s": dt,
            }
            log_model_run(
                name=name,
                params=search.best_params_,
                metrics={"best_f1": search.best_score_},
                duration_s=dt,
            )
            logger.info(
                "  → best F1=%.4f params=%s time=%.1fs",
                search.best_score_,
                search.best_params_,
                dt,
            )

    payload = {"soc": soc, "results": results}
    return _persist("task2", payload)


# ---------------------------------------------------------------------------
# Best-params loaders (used by registry)
# ---------------------------------------------------------------------------


def load_best_params(task: str) -> dict[str, dict[str, Any]]:
    """Return ``{model_name: best_params}`` if a tuning file exists, else ``{}``."""
    f = _out_dir() / f"{task}_best.json"
    if not f.exists():
        return {}
    raw = json.loads(f.read_text())
    if task == "task2":
        raw = raw.get("results", raw)
    return {name: body.get("best_params", {}) for name, body in raw.items()}
