"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np

from src.data.features import (
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
    build_classification_features,
    build_regression_features,
)


def test_regression_features_schema(dataset) -> None:
    X = build_regression_features(dataset.head(100))
    assert list(X.columns) == REGRESSION_FEATURES
    assert not X.isna().any().any()


def test_regression_feature_values(dataset) -> None:
    X = build_regression_features(dataset.head(5))
    # inv_Temp must be positive and below 1 for any sane Celsius reading.
    assert (X["inv_Temp"] > 0).all()
    assert (X["inv_Temp"] < 1).all()
    # log_Freq should equal log10(Frequency).
    np.testing.assert_allclose(X["log_Freq"], np.log10(X["Frequency"]))


def test_classification_features_schema(dataset) -> None:
    X = build_classification_features(dataset.head(100))
    assert list(X.columns) == CLASSIFICATION_FEATURES
    # No NaNs — features should be well-defined on the entire dataset.
    assert not X.isna().any().any()
    # Z_magnitude must be non-negative.
    assert (X["Z_magnitude"] >= 0).all()
