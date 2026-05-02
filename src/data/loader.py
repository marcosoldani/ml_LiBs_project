"""Dataset loading utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import processed_csv_path
from src.data.preprocessing import run_preprocessing
from src.logger import get_logger

logger = get_logger(__name__)


def ensure_dataset(csv_path: Path | None = None) -> Path:
    """Return path to the processed CSV, generating it if missing."""
    path = csv_path or processed_csv_path()
    if not path.exists():
        logger.warning("Processed CSV not found at %s. Running preprocessing…", path)
        run_preprocessing(path)
    return path


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """Load the cleaned battery GEIS dataset into a DataFrame."""
    path = ensure_dataset()
    df = pd.read_csv(path)
    df = df.sort_values(["Aging", "Temperature", "SOC", "Frequency"]).reset_index(drop=True)
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return key summary statistics for the dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "agings": sorted(df["Aging"].unique().tolist()),
        "temperatures": sorted(df["Temperature"].unique().tolist()),
        "socs": sorted(df["SOC"].unique().tolist()),
        "freq_min": float(df["Frequency"].min()),
        "freq_max": float(df["Frequency"].max()),
        "n_combinations": int(df.groupby(["Aging", "Temperature"]).ngroups),
        "n_curves": int(df.groupby(["Aging", "Temperature", "SOC"]).ngroups),
    }
