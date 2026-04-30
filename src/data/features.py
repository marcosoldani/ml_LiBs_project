"""Feature engineering functions for all three tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd

_REGRESSION_REQUIRED = {"Aging", "Temperature", "SOC", "Frequency"}
_CLASSIFICATION_REQUIRED = {"Temperature", "Frequency", "Z_real", "Z_imag"}


def _require_columns(data: pd.DataFrame, required: set[str], task: str) -> None:
    missing = required - set(data.columns)
    if missing:
        raise KeyError(
            f"{task}: input DataFrame is missing required columns: {sorted(missing)}"
        )


def build_regression_features(data: pd.DataFrame) -> pd.DataFrame:
    """9-feature set for the regression tasks (Task 1 and Task 3).

    Features:
        Aging, Temperature, SOC, Frequency, log_Freq, inv_Temp (Arrhenius),
        Aging×Temp, SOC×Temp, SOC×Aging.
    """
    _require_columns(data, _REGRESSION_REQUIRED, "build_regression_features")
    return pd.DataFrame(
        {
            "Aging": data["Aging"].values,
            "Temperature": data["Temperature"].values,
            "SOC": data["SOC"].values,
            "Frequency": data["Frequency"].values,
            "log_Freq": np.log10(data["Frequency"].values),
            "inv_Temp": 1.0 / (data["Temperature"].values + 273.15),
            "Aging_x_Temp": data["Aging"].values * data["Temperature"].values,
            "SOC_x_Temp": data["SOC"].values * data["Temperature"].values,
            "SOC_x_Aging": data["SOC"].values * data["Aging"].values,
        }
    )


def build_classification_features(data: pd.DataFrame) -> pd.DataFrame:
    """10-feature set for the Young/Old classifier (Task 2).

    Impedance becomes the *input*; Aging is excluded (defines the target).
    """
    _require_columns(data, _CLASSIFICATION_REQUIRED, "build_classification_features")
    z_real = data["Z_real"].values
    z_imag = data["Z_imag"].values
    temp = data["Temperature"].values
    freq = data["Frequency"].values
    return pd.DataFrame(
        {
            "Temperature": temp,
            "Frequency": freq,
            "log_Freq": np.log10(freq),
            "Z_real": z_real,
            "Z_imag": z_imag,
            "Z_magnitude": np.sqrt(z_real ** 2 + z_imag ** 2),
            "Z_phase": np.arctan2(z_imag, z_real),
            "inv_Temp": 1.0 / (temp + 273.15),
            "Z_real_x_Temp": z_real * temp,
            "sqrt_Freq": np.sqrt(freq),
        }
    )


REGRESSION_FEATURES = [
    "Aging",
    "Temperature",
    "SOC",
    "Frequency",
    "log_Freq",
    "inv_Temp",
    "Aging_x_Temp",
    "SOC_x_Temp",
    "SOC_x_Aging",
]

CLASSIFICATION_FEATURES = [
    "Temperature",
    "Frequency",
    "log_Freq",
    "Z_real",
    "Z_imag",
    "Z_magnitude",
    "Z_phase",
    "inv_Temp",
    "Z_real_x_Temp",
    "sqrt_Freq",
]
