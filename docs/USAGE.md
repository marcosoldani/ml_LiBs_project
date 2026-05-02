# Usage guide

This document is a quick command reference. The narrative tour lives in
[`README.md`](../README.md); the design rationale is in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime + pytest + ruff + security scans
# `pip install -r requirements.txt` for the runtime-only set (production)
```

## One-shot preprocessing

```bash
python -m scripts.prepare_data
```

Writes `data/processed/batteries_cleaned_dataset.csv` from
`data/raw/GEIS.mat` via `scipy.io.loadmat`.

## Training benchmarks

```bash
python -m scripts.train_all                # all three tasks
python -m scripts.train_all --tasks 1 3    # subset
```

Each run persists:
- `models/task1/benchmark.json`
- `models/task2/benchmark.json`
- `models/task3/benchmark.json`
- `models/task1/last_run.joblib`, `models/task3/last_run.joblib`

## Hyperparameter tuning

```bash
python -m scripts.tune                     # GridSearchCV for all tasks
python -m scripts.tune --tasks 2
```

Results land in `models/tuning/<task>_best.json` and under
`mlruns/battery-geis-<task>-tuning/`.

## Running the platform

The UI is a **React SPA served by a FastAPI backend** on port 8000.

```bash
# Backend (hot reload)
uvicorn app:app --reload --host 127.0.0.1 --port 8000

# Production-like (no reload, serves frontend/dist if built)
uvicorn app:app --host 0.0.0.0 --port 8000

# React frontend dev server (proxies /api → :8000)
cd frontend && npm install && npm run dev      # http://localhost:5173

# MLflow tracking UI
python -m mlflow ui --backend-store-uri mlruns --port 5000
```

For development, run backend and frontend in two terminals.

## Drift monitoring (Lecture 06)

```bash
# Compare two CSVs
python -m scripts.monitor \
  --reference data/processed/batteries_cleaned_dataset.csv \
  --current   data/processed/recent_campaign.csv

# Compare against the JSONL prediction log written by /api/predictions/log
python -m scripts.monitor \
  --current logs/predictions.jsonl --current-format jsonl
```

Reports are saved to `logs/drift_report.json` (KS + PSI per feature, plus
an overall *drift detected* verdict). The same logic is exposed by the
`GET /api/monitoring/drift` endpoint.

## Python API

```python
from src.data.loader import load_dataset
from src.models.task1_loo import run_leave_one_out
from src.models.task2_classification import run_classification
from src.models.task3_aging import run_aging_interpolation
from src.monitoring import detect_dataset_drift

df = load_dataset()

# Task 1
loo = run_leave_one_out(aging=2, temperature=22.5, df=df)
print(loo.best_model_name, loo.metrics_per_model[loo.best_model_name])

# Task 2 — pick the (Aging, SOC) pair to hold out
clf = run_classification(aging=4, soc=3, df=df)
print(clf.best_model_name, clf.metrics_per_model[clf.best_model_name])

# Task 3
aging = run_aging_interpolation(excluded_aging=2, df=df)
print(aging.per_temperature.head())

# Drift report
report = detect_dataset_drift(df[df.Aging != 4], df[df.Aging == 4])
print(report.share_drifted, report.drift_detected)
```

## Docker

```bash
docker build -t battery-mlops .
docker run --rm -p 8000:8000 battery-mlops
# http://localhost:8000
#  - /        → React SPA
#  - /docs    → Swagger / OpenAPI
#  - /api/*   → backend endpoints

# or compose (mounts data/, models/, logs/ as volumes)
docker compose up --build
```

## Tests & lint

```bash
make test                                                # pytest + coverage gate (≥ 90 %) + cleanup
make test-fast                                           # subset (-m "not slow", < 1 min)
make ci                                                  # full CI pipeline locally
pytest                                                   # raw pytest (no coverage)
pytest --cov=src --cov=app --cov-report=term-missing     # with coverage
ruff check src tests scripts app.py                      # lint
ruff check --fix src tests scripts app.py                # auto-fix
```

CI runs lint + pytest with `--cov-fail-under=90` (currently 95 %) plus
`bandit --severity-level high` and `pip-audit --strict` (both
**blocking**) on every push and pull request, on Python 3.11 and 3.12.
The Docker image build is also validated on PRs.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://localhost:3000` | Comma-separated list of origins allowed by FastAPI's CORS middleware |
| `MLFLOW_TRACKING_URI` | `mlruns/` (file backend) | Override to point at a remote MLflow tracking server |

## Extending the platform

1. **Add a new model.** Append it to `src/models/registry.py`. It will
   automatically participate in the benchmark of every task and appear
   in the UI rankings.
2. **Add a new feature.** Edit `src/data/features.py`. Tests cover the
   column order and no-NaN invariants.
3. **Add a new task.** Create `src/models/task4_<name>.py` mirroring the
   dataclass pattern of the existing three tasks, expose it from
   `app.py`, and add a React page under `frontend/src/pages/`.
4. **Add a drift metric.** Implement it in `src/monitoring/drift.py`
   next to KS and PSI; the dataset-level reporter consumes any function
   that returns ``(score, flag)``.
