"""MATLAB .mat → cleaned CSV preprocessing (mirrors notebook 0)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import scipy.io

from src.config import (
    data_processed_path,
    load_config,
    processed_csv_path,
    raw_mat_path,
)
from src.logger import get_logger

logger = get_logger(__name__)


def mat_to_dataframe(mat_file: Path | None = None) -> pd.DataFrame:
    """Convert the raw GEIS.mat file into a tidy DataFrame.

    Output columns: Aging, Temperature, SOC, Frequency, Z_real, Z_imag.
    Z_real and Z_imag are in mΩ (matching the paper's convention).
    """
    cfg = load_config()
    path = mat_file or raw_mat_path()
    logger.info("Loading raw MATLAB file: %s", path)

    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    records = []
    temp_map = dict(cfg.data.temperature_map)

    for aging_idx in cfg.data.agings:
        aging_key = f"Aging{aging_idx}"
        aging_struct = getattr(mat, aging_key, mat[aging_key])

        for temp_key, temp_val in temp_map.items():
            data = getattr(aging_struct, temp_key)
            for soc_idx, per_soc in enumerate(data):
                for row in per_soc:
                    freq = float(row[0])
                    z_real = float(row[1]) * 1000.0
                    z_imag = float(row[2]) * 1000.0
                    records.append(
                        {
                            "Aging": aging_idx,
                            "Temperature": temp_val,
                            "SOC": soc_idx,
                            "Frequency": freq,
                            "Z_real": z_real,
                            "Z_imag": z_imag,
                        }
                    )

    df = pd.DataFrame.from_records(records)
    logger.info("Parsed %d rows across %d (Aging, Temp, SOC) groups",
                len(df), df.groupby(["Aging", "Temperature", "SOC"]).ngroups)
    return df


def run_preprocessing(out_csv: Path | None = None) -> Path:
    """Execute the full preprocessing and persist the cleaned CSV."""
    out = out_csv or processed_csv_path()
    data_processed_path().mkdir(parents=True, exist_ok=True)

    df = mat_to_dataframe()
    df.to_csv(out, index=False)
    logger.info("Saved cleaned dataset → %s", out)
    return out


if __name__ == "__main__":
    run_preprocessing()
