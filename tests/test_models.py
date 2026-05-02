"""End-to-end smoke tests for the three task pipelines."""

from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from src.models.task1_loo import run_leave_one_out
from src.models.task2_classification import run_classification
from src.models.task3_aging import run_aging_interpolation


def _fast_regression_models() -> dict:
    """Two cheap regressors used to make the smoke tests run in seconds.

    Note: ``"Bagging (Ridge)"`` is *not* a real BaggingRegressor — it is
    a multi-output Ridge wrapped in ``MultiOutputRegressor`` to keep the
    test fast (~0.1 s vs ~5 s for the production registry). The smoke
    test is checking the pipeline shape (returns dataclass, predictions
    have the right shape), not the modelling quality. The production
    benchmark in ``src/models/registry.py`` uses the real bagging.
    """
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Bagging (Ridge)": MultiOutputRegressor(Ridge(alpha=1.0)),
    }


def test_task1_loo_runs(dataset) -> None:
    result = run_leave_one_out(
        aging=2,
        temperature=22.5,
        df=dataset,
        models=_fast_regression_models(),
    )
    assert result.best_model_name in result.metrics_per_model
    assert result.best_predictions.shape == (len(result.df_test), 2)
    assert result.y_test.shape == result.best_predictions.shape


def test_task2_classification_runs(dataset) -> None:
    result = run_classification(aging=2, soc=2, df=dataset)
    assert 0.0 <= result.metrics_per_model[result.best_model_name]["Accuracy"] <= 1.0
    assert result.df_test.shape[0] == len(result.y_test)
    assert result.true_class in (0, 1)
    assert {"Temperature", "N", "Accuracy"} <= set(result.per_temperature.columns)


def test_task3_aging_interpolation_runs(dataset) -> None:
    result = run_aging_interpolation(
        excluded_aging=2,
        df=dataset,
        models=_fast_regression_models(),
    )
    assert result.excluded_aging == 2
    assert "R2" in next(iter(result.metrics_per_model.values()))
    assert {"Temperature", "R2", "MAE", "N"} <= set(result.per_temperature.columns)
