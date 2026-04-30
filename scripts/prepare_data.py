"""CLI: generate the processed CSV from the raw GEIS.mat file."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preprocessing import run_preprocessing
from src.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare battery GEIS data")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = parser.parse_args()
    run_preprocessing(args.out)


if __name__ == "__main__":
    main()
