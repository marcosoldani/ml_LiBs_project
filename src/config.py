"""Configuration loader with path resolution and caching."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config(dict):
    """Dict-like config with attribute access and path resolution."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            return Config(value)
        return value


@lru_cache(maxsize=1)
def load_config(path: Path | str | None = None) -> Config:
    """Load the YAML configuration into a Config object."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def resolve_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def data_raw_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg.paths.data_raw)


def data_processed_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg.paths.data_processed)


def models_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg.paths.models)


def logs_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg.paths.logs)


def processed_csv_path() -> Path:
    cfg = load_config()
    return data_processed_path() / cfg.paths.processed_csv


def raw_mat_path() -> Path:
    cfg = load_config()
    return data_raw_path() / cfg.paths.raw_mat_file
