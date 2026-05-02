"""CLI: data drift monitoring (Lecture 06 — Monitoring & Continual Learning).

Compares a *reference* distribution (typically the training CSV) against a
*current* distribution (a freshly collected campaign) using KS and PSI tests
on every shared numerical feature, then prints a JSON drift report and writes
it to ``logs/drift_report.json`` for later inspection.

Usage::

    # Compare two CSVs
    python -m scripts.monitor \
        --reference data/processed/batteries_cleaned_dataset.csv \
        --current   data/processed/recent_campaign.csv

    # Compare training CSV vs the JSON-Lines prediction log
    python -m scripts.monitor \
        --reference data/processed/batteries_cleaned_dataset.csv \
        --current   logs/predictions.jsonl --current-format jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import logs_path, processed_csv_path
from src.logger import get_logger
from src.monitoring import detect_dataset_drift
from src.monitoring.prediction_log import read_predictions

logger = get_logger(__name__)


def _load_frame(path: Path, fmt: str) -> pd.DataFrame:
    """Load a CSV or JSON-Lines prediction log into a flat DataFrame."""
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "jsonl":
        records = read_predictions(path)
        if not records:
            return pd.DataFrame()
        # Flatten {task, inputs, outputs, extra, timestamp} into a single row per call.
        flat = []
        for rec in records:
            row: dict[str, object] = {"task": rec.get("task")}
            for key, val in (rec.get("inputs") or {}).items():
                if isinstance(val, (int, float, str, bool)) or val is None:
                    row[f"input_{key}"] = val
            for key, val in (rec.get("outputs") or {}).items():
                if isinstance(val, (int, float, str, bool)) or val is None:
                    row[f"output_{key}"] = val
            row["timestamp"] = rec.get("timestamp")
            flat.append(row)
        return pd.DataFrame(flat)
    raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect data drift between two distributions")
    parser.add_argument(
        "--reference",
        type=Path,
        default=processed_csv_path(),
        help="Reference distribution (default: training CSV)",
    )
    parser.add_argument(
        "--reference-format",
        choices=("csv", "jsonl"),
        default="csv",
        help="Format of the reference file",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Current distribution (CSV or JSONL)",
    )
    parser.add_argument(
        "--current-format",
        choices=("csv", "jsonl"),
        default="csv",
        help="Format of the current file",
    )
    parser.add_argument(
        "--ks-alpha",
        type=float,
        default=0.05,
        help="Significance threshold for the KS test (default: 0.05)",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.25,
        help="PSI value above which drift is flagged (default: 0.25)",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Restrict the analysis to these columns (default: all numerical)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the JSON drift report (default: logs/drift_report.json)",
    )
    args = parser.parse_args()

    logger.info("Reference: %s (%s)", args.reference, args.reference_format)
    logger.info("Current:   %s (%s)", args.current, args.current_format)

    reference = _load_frame(args.reference, args.reference_format)
    current = _load_frame(args.current, args.current_format)
    if reference.empty or current.empty:
        raise SystemExit("Reference or current distribution is empty — nothing to compare.")

    report = detect_dataset_drift(
        reference,
        current,
        features=args.features,
        ks_alpha=args.ks_alpha,
        psi_threshold=args.psi_threshold,
    )

    out_path = args.out or (logs_path() / "drift_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))

    logger.info(
        "Drift summary: %d/%d features drifted (%.0f%%) — overall=%s",
        report.n_drifted,
        len(report.features),
        100 * report.share_drifted,
        "YES" if report.drift_detected else "no",
    )
    logger.info("Full report → %s", out_path)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
