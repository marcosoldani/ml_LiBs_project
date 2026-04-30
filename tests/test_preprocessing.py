"""Tests for `src/data/preprocessing.py` (MATLAB → tidy CSV)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import raw_mat_path
from src.data.preprocessing import mat_to_dataframe, run_preprocessing

EXPECTED_COLUMNS = {"Aging", "Temperature", "SOC", "Frequency", "Z_real", "Z_imag"}


def _raw_available() -> bool:
    try:
        return raw_mat_path().exists()
    except Exception:
        return False


@pytest.mark.skipif(not _raw_available(), reason="data/raw/GEIS.mat is not bundled")
def test_mat_to_dataframe_schema() -> None:
    """Parsed DataFrame must expose the canonical 6-column schema."""
    df = mat_to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == EXPECTED_COLUMNS
    assert not df.empty


@pytest.mark.skipif(not _raw_available(), reason="data/raw/GEIS.mat is not bundled")
def test_mat_to_dataframe_value_ranges() -> None:
    """Aging ∈ {0..4}, SOC ∈ {0..4}, Frequency strictly positive, no NaNs."""
    df = mat_to_dataframe()
    assert set(df["Aging"].unique()) <= {0, 1, 2, 3, 4}
    assert set(df["SOC"].unique()) <= {0, 1, 2, 3, 4}
    assert (df["Frequency"] > 0).all()
    assert df.notna().all().all()


@pytest.mark.skipif(not _raw_available(), reason="data/raw/GEIS.mat is not bundled")
def test_run_preprocessing_writes_csv(tmp_path: Path) -> None:
    """The CLI helper must produce a non-empty CSV at the requested path."""
    out = tmp_path / "cleaned.csv"
    written = run_preprocessing(out_csv=out)
    assert written == out
    assert out.exists()
    reread = pd.read_csv(out)
    assert set(reread.columns) == EXPECTED_COLUMNS
    assert len(reread) > 0


def test_mat_to_dataframe_missing_file_errors(tmp_path: Path) -> None:
    """Pointing at a non-existent .mat file must raise (no silent fallback)."""
    bogus = tmp_path / "nope.mat"
    with pytest.raises((FileNotFoundError, OSError)):
        mat_to_dataframe(bogus)
