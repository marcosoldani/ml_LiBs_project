# Architecture & MLOps design

This document covers both the layering of the Battery GEIS MLOps platform
*and* the MLOps decisions behind it, mapped to the SUPSI *Machine Learning
Operations* lecture program. It complements the top-level
[`README.md`](../README.md) and the operational
[`USAGE.md`](USAGE.md) / [`MONITORING.md`](MONITORING.md) playbooks.

## Layered design

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Front-end layer                               │
│  React + TypeScript SPA (frontend/)                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP (JSON)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI bridge (app.py)                       │
│  /health · /api/project · /api/dataset/* · /api/task{1,2,3}/run      │
│  /api/benchmarks/{task} · /api/predictions/{log,recent}              │
│  /api/monitoring/drift                                               │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ calls Python pipelines directly
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Task modules (src/models)                     │
│  task1_loo · task2_classification · task3_aging · tuning · registry  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ uses
          ┌────────────────────┼─────────────────────┬────────────────┐
          ▼                    ▼                     ▼                ▼
   Feature layer        Evaluation layer    Visualization      Monitoring
  (src/data/…)        (src/evaluation/…)   (src/visualization)  (src/monitoring)
          │                    ▲                                     ▲
          │                    │                                     │
          ▼                    │                                     │
   Data layer ─────────────────┘                                     │
  (src/data/loader, preprocessing) ── data/raw/GEIS.mat              │
                                      data/processed/*.csv ──────────┘

                       Cross-cutting: src/config (YAML), src/logger,
                                      src/tracking (MLflow optional)
```

Each layer only imports from lower layers — the dependency graph is
acyclic and trivially testable.

## Module reference

| Module | Responsibility |
|---|---|
| `src/config.py` | Load `config/config.yaml` with `functools.lru_cache`, resolve paths |
| `src/logger.py` | Console + rotating file handler at `logs/pipeline.log` |
| `src/constants.py` | Shared colours, labels and markers for the UIs |
| `src/tracking.py` | MLflow integration with graceful fallback when missing |
| `src/data/preprocessing.py` | Parse `GEIS.mat`, emit tidy CSV |
| `src/data/loader.py` | Cached `load_dataset()` + `dataset_summary()` |
| `src/data/features.py` | `build_regression_features`, `build_classification_features` |
| `src/models/registry.py` | Default hyperparameters for regression & classification baselines |
| `src/models/tuning.py` | GridSearchCV with task-coherent CV splitters; persists best params |
| `src/models/task1_loo.py` | Leave-One-Out curve reconstruction (regression) |
| `src/models/task2_classification.py` | Young/Old classification (binary) |
| `src/models/task3_aging.py` | Leave-One-Aging-Out interpolation (regression) |
| `src/evaluation/metrics.py` | `regression_metrics`, `classification_metrics` dataclasses |
| `src/visualization/plots.py` | Reusable Plotly figures |
| `src/monitoring/drift.py` | KS + PSI drift detection over numerical features |
| `src/monitoring/prediction_log.py` | JSON-Lines append-only prediction store |
| `app.py` | FastAPI bridge — exposes the pipelines and monitoring as JSON |
| `frontend/` | React + TypeScript + Vite SPA (the only UI) |
| `scripts/prepare_data.py` | CLI for preprocessing |
| `scripts/train_all.py` | CLI to benchmark all three tasks |
| `scripts/tune.py` | CLI for GridSearchCV across all tasks |
| `scripts/monitor.py` | CLI for drift detection (Lecture 06) |

## Data flow

1. `scripts/prepare_data.py` → writes
   `data/processed/batteries_cleaned_dataset.csv`.
2. `src/data/loader.load_dataset()` → cached pandas DataFrame.
3. Feature functions in `src/data/features.py` produce NumPy arrays for
   models.
4. Each task module runs the benchmark and returns a **typed dataclass**
   (`LOOResult`, `ClassificationResult`, `AgingInterpResult`).
5. The FastAPI bridge serialises those dataclasses into JSON; React
   consumes them and renders Plotly plots / metric grids.
6. `src/monitoring/` consumes either the same CSV (for offline drift
   checks) or the `logs/predictions.jsonl` populated by
   `/api/predictions/log` (for online drift checks against deployed
   traffic).

## Design principles

- **Single source of truth** — notebooks, scripts and the FastAPI bridge
  all import the same functions from `src/`. Feature engineering, label
  thresholds and paths never drift across surfaces.
- **Pure functions in `visualization/`** — easy to reuse and test.
- **Typed outputs** — dataclasses make contract changes loud and explicit.
- **Layered imports** — upper layers only depend on lower ones (UI →
  app.py → models → data); the dependency graph stays acyclic.
- **Stateless business logic** — global mutable state is confined to the
  prediction log file.
- **Optional infrastructure** — MLflow and the React build are *graceful*
  add-ons (the rest of the system runs fine without either).

---

# MLOps design decisions

The rest of the document maps each implementation choice to the relevant
SUPSI lecture so the reader can cross-check against the slides.

## 1. Reproducibility — Lectures 01, 02

- `random_state=42` everywhere (centralised in `src/models/registry.py`
  and `config.yaml`).
- Deterministic data splits: `StratifiedGroupKFold` for Task 2,
  Leave-One-Group-Out for Tasks 1 and 3.
- Two `requirements*.txt` files, both with versioned upper bounds:
  `requirements.txt` covers the **runtime** (numpy, scikit-learn,
  FastAPI, MLflow-skinny, …); `requirements-dev.txt` adds the dev
  tooling (`pytest`, `pytest-cov`, `httpx`, `ruff`, `bandit`,
  `pip-audit`) on top. Production deploys install only the runtime
  set; CI installs the dev set.
- The preprocessing step is idempotent and cached via
  `functools.lru_cache` on `load_dataset()`.

## 2. Configuration management — Lecture 02

- Single `config/config.yaml` defines paths, task metadata, data
  dimensions, CV folds, and tuning grids.
- `src.config.load_config()` returns a dict-like `Config` object
  (attribute access), cached for the lifetime of the process.
- Path helpers (`data_raw_path()`, `models_path()`, …) centralise
  filesystem layout changes.
- Task 2's *Young vs Old* threshold (`task2.young_max_aging`) is read
  from config — no hard-coded constants.

## 3. Logging & observability — Lecture 03

- `src/logger.py` configures a single root logger with:
  - stream handler (stdout)
  - **rotating** file handler (`logs/pipeline.log`, 10 MB × 5 backups)
  - formatter `[%(asctime)s] [%(levelname)s] %(name)s - %(message)s`
- Task modules emit structured progress lines (per-model metrics +
  timing).
- Predictions made through `app.py` can be appended to
  `logs/predictions.jsonl` via `POST /api/predictions/log` — see
  Section 7 below.

## 4. Experiment tracking — Lecture 03

- `src/tracking.py` wraps MLflow behind a `try/except ImportError`, so
  the package is *optional* and installing it just unlocks extra
  capabilities.
- Two-level run hierarchy: a parent run per task and a child run per
  model.
- Each run logs git SHA, dataset SHA-256, configuration snapshot,
  hyperparameters, training time, and per-metric values.
- Backend defaults to a local file store (`mlruns/`); override with
  `MLFLOW_TRACKING_URI` to point at a remote server.

## 5. Testing — Lecture 04

- 81 pytest tests (70 fast + 11 marked `slow`) covering: dataset loading & schema, feature
  invariants, metric dataclasses, end-to-end smoke tests per task,
  every Plotly figure, tracking + tuning wiring, MATLAB → CSV
  preprocessing, drift detection, prediction log, the FastAPI HTTP
  layer (one test per endpoint) and the persisted benchmark contracts.
- `[tool.pytest.ini_options]` lives in `pyproject.toml`.
- Coverage on `src/` + `app.py` is enforced at ≥ 90 % in CI (currently
  95 %).

## 6. Continuous integration — Lecture 04

`.github/workflows/ci.yml` runs on every push to `main` / `master` and
on every pull request, on Python 3.11 and 3.12:

1. `ruff check src tests scripts app.py` (lint, blocking)
2. `pytest --cov=src --cov=app --cov-fail-under=90` (blocking; current
   coverage is 95 %)
3. `bandit -r src app.py --severity-level high` (security audit,
   **blocking on HIGH severity**)
4. `pip-audit -r requirements.txt --strict` (CVE scan, **blocking on
   any advisory**; currently passes because the platform-conditional
   `cryptography` pin selects ≥ 46.0.7 on amd64)
5. `docker build .` on PRs and `main` (smoke-tests the production image)

## 7. Containerisation & deployment — Lectures 02, 05

`Dockerfile` is a multi-stage build:

- **Stage 1** — `node:20-alpine` compiles the React frontend into
  `/frontend/dist`.
- **Stage 2** — `python:3.12-slim` installs `requirements.txt`, copies
  `src/`, `app.py`, `scripts/`, `config/`, `data/` and the static React
  bundle, then runs `uvicorn app:app` on `0.0.0.0:8000`.
- Health-check on `GET /health`.

`docker-compose.yml` mounts `data/`, `models/` and `logs/` as volumes.
FastAPI publishes typed Pydantic schemas plus an OpenAPI doc at
`/docs` and `/redoc`. CORS origins are read from `CORS_ALLOWED_ORIGINS`
so the same image deploys against different frontends without code
changes.

## 8. Monitoring & continual learning — Lecture 06

- `src/monitoring/drift.py` implements **Kolmogorov–Smirnov** and
  **Population Stability Index** checks; the dataset-level reporter
  flags drift when ≥ 25 % of features individually drift.
- `src/monitoring/prediction_log.py` provides a JSON-Lines append-only
  store, written to `logs/predictions.jsonl` from
  `POST /api/predictions/log`.
- `scripts/monitor.py` is a CLI that compares any reference / current
  pair of distributions (CSV or JSONL) and writes
  `logs/drift_report.json`. The same logic backs the
  `GET /api/monitoring/drift` endpoint.
- See [`MONITORING.md`](MONITORING.md) for the operational playbook.

## 9. Artefact persistence

- `models/taskN/benchmark.json` — summary of metrics per model.
- `models/taskN/last_run.joblib` — full dataclass of the last run for
  Task 1 and Task 3 (useful for offline debugging).
- `models/tuning/<task>_best.json` — best hyperparameters from
  GridSearchCV.

## 10. Developer UX

The common workflows are short Python module invocations, listed in
[`USAGE.md`](USAGE.md): `python -m scripts.prepare_data`,
`python -m scripts.train_all`, `python -m scripts.tune`,
`python -m scripts.monitor`, `uvicorn app:app --reload`, `pytest`,
`ruff check src tests scripts app.py`. CI calls these directly so the
local and the remote contracts match.

## 11. What is *not* yet in scope (future work)

- **Continual-learning loop** — automated retraining triggered by drift
  alarms (the building blocks in `src/monitoring` exist; the
  orchestrator does not).
- **Data versioning** with DVC or LakeFS (the dataset is committed to
  git for course convenience).
- **Authentication / authorisation** on the API (no secrets to protect
  in the current research setup).
- **Cloud deployment** (Lecture 08): only the local Docker path is
  implemented today.
