"""CLI: run GridSearchCV for one or more tasks and persist best params."""

from __future__ import annotations

import argparse
import json

from src.logger import get_logger
from src.models.tuning import tune_task1, tune_task2, tune_task3

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for battery-geis tasks")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["1", "2", "3"],
        help="Subset of tasks to tune (default: all)",
    )
    args = parser.parse_args()
    outputs: dict[str, str] = {}

    if "1" in args.tasks:
        logger.info("— Tuning Task 1 (LOO regression) —")
        outputs["task1"] = str(tune_task1())
    if "2" in args.tasks:
        logger.info("— Tuning Task 2 (classification) —")
        outputs["task2"] = str(tune_task2())
    if "3" in args.tasks:
        logger.info("— Tuning Task 3 (Leave-One-Aging-Out) —")
        outputs["task3"] = str(tune_task3())

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
