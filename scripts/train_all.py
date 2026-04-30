"""CLI: run default benchmarks for all three tasks and persist results."""

from __future__ import annotations

import argparse
import json

from src.logger import get_logger
from src.models import task1_loo, task2_classification, task3_aging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmarks for all tasks")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["1", "2", "3"],
        help="Subset of tasks to run (default: all)",
    )
    args = parser.parse_args()
    outputs = {}

    if "1" in args.tasks:
        logger.info("— Running Task 1 (Leave-One-Out) benchmark —")
        outputs["task1"] = str(task1_loo.persist_default_benchmark())
    if "2" in args.tasks:
        logger.info("— Running Task 2 (Classification) benchmark —")
        outputs["task2"] = str(task2_classification.persist_default_benchmark())
    if "3" in args.tasks:
        logger.info("— Running Task 3 (Aging Interpolation) benchmark —")
        outputs["task3"] = str(task3_aging.persist_default_benchmark())

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
