"""Thin MLflow wrapper for experiment tracking.

Every pipeline run opens a parent MLflow run, logs the config snapshot,
the dataset SHA-256 and the current git SHA (best effort), then logs a
child run per trained model with its parameters, metrics and duration.

The tracking store is a local file backend (``mlruns/``) so nothing needs
to be running for logging to work. Browse the UI with ``make mlflow-ui``.

MLflow is an **optional** dependency: if the package is not installed,
every helper degrades to a no-op and :func:`list_runs` returns an empty
DataFrame. Install it with ``pip install -r requirements-tracking.txt``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import mlflow  # type: ignore[import-not-found]

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False

from src.config import PROJECT_ROOT, load_config, processed_csv_path
from src.logger import get_logger

logger = get_logger(__name__)

_INITIALIZED: bool = False


def _is_enabled() -> bool:
    if not _MLFLOW_AVAILABLE:
        return False
    try:
        return bool(load_config().tracking.enabled)
    except (AttributeError, KeyError):
        return False


def _init_tracking() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    cfg = load_config()
    backend = cfg.tracking.backend_uri
    backend_path = Path(backend)
    if not backend_path.is_absolute():
        backend_path = PROJECT_ROOT / backend
    backend_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(backend_path.as_uri())
    _INITIALIZED = True


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def dataset_sha256(csv_path: Path | None = None) -> str | None:
    """SHA-256 of the processed CSV; ``None`` if the file is missing."""
    path = csv_path or processed_csv_path()
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _experiment_name(task: str) -> str:
    cfg = load_config()
    return f"{cfg.tracking.experiment_prefix}-{task}"


@contextmanager
def track_pipeline(
    task: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Open a parent run for a whole pipeline execution.

    If tracking is disabled in config, this is a no-op context (yields None).
    """
    if not _is_enabled():
        yield None
        return

    _init_tracking()
    mlflow.set_experiment(_experiment_name(task))
    base_tags = {
        "task": task,
        "git_sha": _git_sha() or "unknown",
        "data_sha256": dataset_sha256() or "unknown",
    }
    if tags:
        base_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=base_tags) as run:
        if params:
            # MLflow caps param value length, so stringify compound values
            safe = {k: _safe_param(v) for k, v in params.items()}
            mlflow.log_params(safe)
        logger.info("MLflow run started: experiment=%s run_id=%s", _experiment_name(task), run.info.run_id)
        yield run


def log_model_run(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    duration_s: float,
) -> None:
    """Nested child run for a single (model, fold) result."""
    if not _is_enabled():
        return
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params({k: _safe_param(v) for k, v in params.items()})
        mlflow.log_metrics({_safe_metric_key(k): float(v) for k, v in metrics.items()})
        mlflow.log_metric("duration_s", float(duration_s))


def log_artifact_json(name: str, payload: dict[str, Any]) -> None:
    """Dump a dict as a JSON artifact attached to the current run."""
    if not _is_enabled():
        return
    path = PROJECT_ROOT / "mlruns" / "_tmp"
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / name
    file_path.write_text(json.dumps(payload, indent=2, default=str))
    mlflow.log_artifact(str(file_path))


def _safe_param(value: Any) -> str:
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)[:500]
    return str(value)[:500]


def _safe_metric_key(key: str) -> str:
    return key.replace(" ", "_").replace("(", "").replace(")", "")


def list_runs(task: str) -> pd.DataFrame:
    """Return all runs for the task as a DataFrame."""
    if not _is_enabled():
        return pd.DataFrame()
    _init_tracking()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(_experiment_name(task))
    if exp is None:
        return pd.DataFrame()
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if df.empty:
        return df
    df = df.sort_values("start_time", ascending=False)
    return df
