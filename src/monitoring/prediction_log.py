"""Append-only JSON-Lines prediction store used by drift checks and audits.

The motivation is twofold:

1. **Audit trail** — every API call to a task pipeline can be appended here,
   so we know *who/when/what* the system predicted long after the response was
   returned to the client.
2. **Drift source** — :func:`src.monitoring.drift.detect_dataset_drift` can
   ingest the resulting JSONL as the *current* distribution and compare it
   against the training CSV.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import logs_path

DEFAULT_LOG_FILE = "predictions.jsonl"


@dataclass
class PredictionRecord:
    """Single prediction event flushed to disk."""

    task: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_json_default)


def log_prediction(
    task: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    extra: dict[str, Any] | None = None,
    log_file: Path | None = None,
) -> Path:
    """Append a prediction event to ``logs/predictions.jsonl``.

    The directory is created lazily; the file is opened in append mode so
    concurrent writers do not stomp on each other.
    """
    record = PredictionRecord(
        task=task,
        inputs=inputs,
        outputs=outputs,
        extra=extra or {},
    )
    target = log_file or (logs_path() / DEFAULT_LOG_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json())
        fh.write("\n")
    return target


def read_predictions(log_file: Path | None = None) -> list[dict[str, Any]]:
    """Read every JSON-Lines record back into a list of dicts."""
    target = log_file or (logs_path() / DEFAULT_LOG_FILE)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _json_default(value: Any) -> Any:
    """Best-effort JSON encoder for numpy / pandas values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Type {type(value)!r} is not JSON serialisable")
