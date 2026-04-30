"""Scikit-learn model registries shared across tasks.

If ``use_tuned=True`` is passed (or the environment opts into it) we load the
best hyperparameters from ``models/tuning/<task>_best.json`` (produced by
``scripts/tune.py``) and override the defaults. Models for which no tuned
entry exists keep their baseline parameters — so the fallback is always safe.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC

RANDOM_SEED = 42


def _apply_tuned(models: dict[str, Any], task: str) -> dict[str, Any]:
    """Mutate models with best params from the tuning JSON, if present."""
    from src.models.tuning import load_best_params

    best = load_best_params(task)
    if not best:
        return models
    for name, params in best.items():
        if name in models and params:
            with suppress(ValueError):
                models[name].set_params(**params)
    return models


def regression_models(use_tuned: bool = False, task: str = "task1") -> dict:
    """Baseline regression models used in Task 1 and Task 3."""
    models: dict[str, Any] = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
            )
        ),
        "K-Nearest Neighbors": KNeighborsRegressor(
            n_neighbors=10, weights="distance", n_jobs=-1
        ),
        "Bagging (Ridge)": BaggingRegressor(
            estimator=Ridge(alpha=1.0),
            n_estimators=30,
            max_samples=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }
    if use_tuned:
        models = _apply_tuned(models, task)
    return models


def classification_models(use_tuned: bool = False, task: str = "task2") -> dict:
    """Baseline classifiers for Task 2."""
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=10, weights="distance", n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10.0, probability=True, random_state=RANDOM_SEED
        ),
    }
    if use_tuned:
        models = _apply_tuned(models, task)
    return models
