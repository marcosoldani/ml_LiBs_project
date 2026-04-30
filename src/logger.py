"""Centralized logging setup."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import load_config, logs_path

_CONFIGURED = False

LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
LOG_FILE_BACKUP_COUNT = 5  # keep 5 rotated files (≈50 MB cap on disk)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with project-wide settings."""
    global _CONFIGURED
    if not _CONFIGURED:
        _configure_root()
        _CONFIGURED = True
    return logging.getLogger(name)


def _configure_root() -> None:
    cfg = load_config()
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    fmt = cfg.logging.format
    datefmt = cfg.logging.date_format

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    log_dir = logs_path()
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "pipeline.log",
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
