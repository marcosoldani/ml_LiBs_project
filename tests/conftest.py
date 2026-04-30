"""Shared pytest fixtures."""

from __future__ import annotations

import atexit
import contextlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Disable subprocess coverage tracking *before* pytest-cov initialises:
# scikit-learn's joblib (n_jobs=-1) spawns workers that would otherwise
# emit `.coverage.HOSTNAME.pidN.UUID` per-pid files which pytest-cov
# never combines on macOS. We don't measure worker code anyway.
os.environ.pop("COVERAGE_PROCESS_START", None)


@pytest.fixture(scope="session")
def dataset():
    from src.data.loader import load_dataset

    return load_dataset()


def _sweep_orphan_coverage_files() -> None:
    """Remove ``.coverage.HOSTNAME.pidN.UUID`` files left by joblib workers.

    scikit-learn with ``n_jobs=-1`` spawns subprocess workers; when
    pytest-cov is active each worker emits its own per-pid coverage
    file. On macOS these can be written *after* `pytest_sessionfinish`,
    so we sweep at three points:

      1. ``atexit`` — late in interpreter shutdown;
      2. ``pytest_unconfigure`` — pytest's final hook;
      3. a brief ``time.sleep`` inside atexit, to give late-flushing
         joblib workers a chance to land before we delete.
    """
    import time

    time.sleep(0.4)
    for orphan in ROOT.glob(".coverage.*"):
        with contextlib.suppress(OSError):
            orphan.unlink()


atexit.register(_sweep_orphan_coverage_files)


def pytest_unconfigure(config):  # noqa: ARG001
    """Pytest's last hook — sweep orphans here too (belt + braces)."""
    _sweep_orphan_coverage_files()
