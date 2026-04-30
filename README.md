# 🔋 Battery GEIS — MLOps Platform

> **SUPSI — Semester Project** (in collaborazione con Politecnico di Milano)
> Authors: **Davide Corso** & **Marco Soldani**
> Paper di riferimento: *Hysteresis Phenomenon in the Electric Parameters of Lithium-Ion Batteries under Temperature Effects* — gruppo di ricerca del Politecnico di Milano (**IEEE THERMINIC 2025**).

A full **MLOps pipeline** for the analysis of **Galvanostatic Electrochemical Impedance Spectroscopy (GEIS)** measurements on a **10 Ah LiCoO₂** pouch cell, covering data preparation, model training, model serving, UI and CI/CD.

The platform exposes three prediction tasks through a **FastAPI** backend (`app.py`) and a **React + TypeScript** frontend (`frontend/`):

1. **Leave-One-Out** — reconstruct a single missing Nyquist plot from the other 39.
2. **Young-Old Classification** — pick an `(Aging, SOC)` pair, hold its 8 Nyquist curves (one per temperature) out from training, and classify them as young (Aging 0–2) or old (Aging 3–4) from their shape alone.
3. **Leave-One-Aging-Out** — predict an entire aging level (default: Aging 2, the middle one) from the other four.

---

## Table of contents

1. [Background & paper](#background--paper)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Usage](#usage)
5. [The three tasks](#the-three-tasks)
6. [Dataset](#dataset)
7. [Hyperparameter tuning](#hyperparameter-tuning)
8. [Experiment tracking (MLflow)](#experiment-tracking-mlflow)
9. [Monitoring & drift detection](#monitoring--drift-detection)
10. [MLOps practices](#mlops-practices)
11. [Testing & CI](#testing--ci)
12. [Docker](#docker)
13. [Documentation](#documentation)

---

## Background & paper

### The physical problem
Lithium-Ion batteries degrade over time: their **internal impedance grows**
and its frequency response reshapes as the electrochemistry ages. We measure
this with **Galvanostatic Electrochemical Impedance Spectroscopy (GEIS)** —
a small sinusoidal current is injected at many frequencies and the voltage
response yields the complex impedance `Z(f) = Re(Z) + j·Im(Z)`.

Plotting `Re(Z)` on the X axis and `-Im(Z)` on the Y axis (a **Nyquist
diagram**, standard EIS convention) reveals two characteristic features:

- a **high-frequency semicircle** — charge-transfer resistance and
  double-layer capacitance (~RC parallel behaviour);
- a **low-frequency tail** — solid-state diffusion (Warburg element).

As the cell ages, the semicircle **widens** (charge-transfer resistance
grows) and the tail **rotates**. Temperature has a large, non-linear effect
(Arrhenius-like), which is why temperature is a first-class feature in
every model (`1/T` is added explicitly).

### The paper
> *Hysteresis Phenomenon in the Electric Parameters of Lithium-Ion Batteries
> under Temperature Effects* — gruppo di ricerca del Politecnico di Milano,
> **IEEE THERMINIC 2025**.

The paper characterises a 10 Ah LiCoO₂ pouch cell across **5 aging states**
(fresh → aged), **8 temperatures** (20 – 47.5 °C) and **5 states of charge**
(SOC 0 → 4) and shows a measurable hysteresis in the impedance parameters
with respect to thermal cycling.

### Why these three tasks
Our MLOps platform answers three questions a battery engineer actually cares
about:

1. **Can we skip a measurement?** If the lab acquires 39 of 40 `(Aging,
   Temperature)` combinations, can ML fill in the 40th? → **Task 1**.
2. **Can we triage an unseen pair?** Given an Aging level and a SOC
   level the model was never trained on, can we tell from the shape of
   the 8 Nyquist curves of that pair (one per temperature) whether the
   cell is still young (Aging 0–2) or already aged (Aging 3–4)? →
   **Task 2**.
3. **Can we extrapolate through life?** If we only have impedance data at
   four aging states, can we predict the fifth we never measured directly?
   → **Task 3** (the hardest — the model never sees the target aging during
   training).

A "good" prediction in Tasks 1 / 3 means the predicted Nyquist curve has
the **right semicircle size** and the **right low-frequency tail orientation**
(R² per component ≥ 0.95 is our empirical bar). For Task 2 we track
accuracy on the 8 held-out Nyquist curves of the chosen `(Aging, SOC)` pair
(one curve per temperature) plus per-temperature accuracy — AUC/ROC are
not applicable because all 8 curves belong to the same class by
construction.

---

## Project layout

```
TestMLOps_Progetto/
├── app.py                     # FastAPI backend (HTTP layer over src/)
├── frontend/                  # React + Vite + TypeScript SPA
│   ├── src/
│   │   ├── pages/             # Home, Task1, Task2, Task3, Documentation
│   │   ├── components/        # Layout, Plot, MetricGrid, …
│   │   ├── api.ts             # Axios client for /api
│   │   └── types.ts
│   ├── vite.config.ts
│   └── package.json
├── src/                       # Core package (data, models, evaluation, viz)
│   ├── config.py              # YAML config loader (cached)
│   ├── logger.py              # Structured logging
│   ├── constants.py
│   ├── data/
│   │   ├── preprocessing.py   # mat → csv
│   │   ├── loader.py
│   │   └── features.py        # Feature engineering
│   ├── models/
│   │   ├── registry.py
│   │   ├── task1_loo.py
│   │   ├── task2_classification.py
│   │   └── task3_aging.py
│   ├── evaluation/metrics.py
│   ├── visualization/plots.py
│   └── monitoring/            # Drift detection (KS / PSI) + prediction log
├── scripts/                   # CLI entry points
│   ├── prepare_data.py
│   ├── train_all.py
│   ├── tune.py
│   └── monitor.py             # Drift report CLI (Lecture 06)
├── tests/                     # pytest suite (data / features / metrics / models)
├── notebooks/                 # Original experiment notebooks (kept for reference)
├── config/config.yaml         # Central configuration
├── data/
│   ├── raw/GEIS.mat
│   └── processed/batteries_cleaned_dataset.csv
├── models/task{1,2,3}/        # Persisted benchmarks & joblib dumps
├── logs/pipeline.log          # Runtime logs
├── docs/                      # Extended markdown documentation + LaTeX semester report
├── .github/workflows/ci.yml   # GitHub Actions
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml             # Ruff + pytest config
├── Makefile                   # `make test`, `make ci`, `make clean-cov`, …
├── requirements.txt           # Runtime dependencies only
└── requirements-dev.txt       # `-r requirements.txt` + pytest, ruff, bandit, pip-audit
```

## Quick start

> **Python version.** Tested on Python 3.11 and 3.12. CI runs both.

```bash
# 1. Clone, create a virtualenv, install deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime + pytest + ruff + bandit + pip-audit
# (use `pip install -r requirements.txt` for the runtime-only set, e.g. in prod)

# 2. Prepare the dataset (once)
python -m scripts.prepare_data        # → data/processed/batteries_cleaned_dataset.csv

# 3. Train all baselines (optional — the app trains on demand)
python -m scripts.train_all           # → models/task{1,2,3}/benchmark.json

# 4. Run the tests
make test                             # pytest + coverage gate (≥ 90 %) + auto-cleanup

# 5. Launch backend + frontend (two terminals)
uvicorn app:app --reload --port 8000  # backend
cd frontend && npm install && npm run dev   # frontend on :5173

# Production single-binary mode (FastAPI also serves frontend/dist):
cd frontend && npm run build
uvicorn app:app --host 0.0.0.0 --port 8000   # http://localhost:8000
```

## Usage

### Command cheat sheet

| Command | Effect |
|---|---|
| `pip install -r requirements.txt` | Install runtime dependencies only |
| `pip install -r requirements-dev.txt` | Install runtime + dev tooling (pytest, ruff, bandit, pip-audit) |
| `make ci` | Run the exact CI pipeline locally (ruff + pytest + coverage gate + bandit + pip-audit) |
| `python -m scripts.prepare_data` | Regenerate the processed CSV from the raw `.mat` file |
| `python -m scripts.train_all` | Run default benchmarks for tasks 1, 2, 3 (subset: `--tasks 1 3`) |
| `python -m scripts.tune` | Run GridSearchCV and persist `models/tuning/*.json` (subset: `--tasks 2`) |
| `python -m scripts.monitor --current path/to.csv` | Drift detection (Lecture 06) |
| `python -m mlflow ui --backend-store-uri mlruns` | Launch the MLflow UI on `http://localhost:5000` |
| `pytest` / `pytest --cov=src --cov=app --cov-report=term-missing` | Run tests (with optional coverage; CI uses both `--cov=src` and `--cov=app`) |
| `ruff check src tests scripts app.py` | Ruff linting |
| `uvicorn app:app --reload --port 8000` | Launch the FastAPI backend (`http://localhost:8000`) |
| `cd frontend && npm run dev` | Launch the React dev server (`http://localhost:5173`, proxies `/api`) |
| `cd frontend && npm run build` | Compile the React app to `frontend/dist` |
| `docker build -t battery-mlops .` / `docker run -p 8000:8000 battery-mlops` | Docker workflow |

### Python API

```python
from src.data.loader import load_dataset
from src.models.task1_loo import run_leave_one_out
from src.models.task2_classification import run_classification
from src.models.task3_aging import run_aging_interpolation

df = load_dataset()
task1 = run_leave_one_out(aging=2, temperature=22.5, df=df)
task2 = run_classification(aging=4, soc=3, df=df)
task3 = run_aging_interpolation(excluded_aging=2, df=df)

print(task1.best_model_name, task1.metrics_per_model[task1.best_model_name])
```

Each `run_*` function returns a **typed dataclass** that the UI, the tests and the scripts all consume.

## The three tasks

### Task 1 — Leave-One-Out Curve Reconstruction

Given the 40 possible `(Aging, Temperature)` combinations, **one** is held out (random or user-picked). The model is trained on the remaining **39 combinations** and must reconstruct the 5 Nyquist plots of the missing combination.

- **Baselines:** Ridge, Random Forest, Gradient Boosting, KNN, Bagging(Ridge)
- **Target:** `(Z_real, Z_imag)` multi-output regression
- **Selection metric:** MSE

### Task 2 — Young vs Old Classification

The user picks an `(Aging, SOC)` pair: the **8 Nyquist curves** of that
combination (one per temperature, ~49 frequencies each, ~392 rows) are
**removed** from training. The model learns from the remaining ~9 400
rows and must classify those 8 curves as **Young (Aging 0–2)** or
**Old (Aging 3–4)** from their *shape* alone. Aging is the **target**,
not a feature.

- **Baselines:** Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, KNN, SVM (RBF)
- **Hold-out strategy:** the 8 Nyquist curves of the picked `(Aging, SOC)` go to test; everything else trains
- **Metrics:** Accuracy on the 8 held-out curves (global and per-temperature). AUC-ROC is **not applicable** because all 8 curves belong to the same class by construction.

### Task 3 — Leave-One-Aging-Out (Aging Interpolation)

An **entire aging level** is removed from training (default: **Aging 2**, the middle one). The model learns from the remaining four aging states and reconstructs all **40 Nyquist plots** (8 Temperatures × 5 SOC) of the held-out aging.

- **Baselines:** same regressors as Task 1
- **Difficulty:** *High* — no direct observation of the held-out aging
- **Selection metric:** R²

## Dataset

| Column | Type | Description |
|---|---|---|
| `Aging` | int | 0 (fresh) → 4 (aged) |
| `Temperature` | float | 8 levels in °C (20 → 47.5) |
| `SOC` | int | 0 → 4 (approx. 0%, 25%, 50%, 75%, 100%) |
| `Frequency` | float | Hz — log-spaced values (~0.1 Hz → 10 kHz). 49 frequencies on the majority of the 200 curves; 5 curves carry 50 frequencies. ~9 800 rows total. |
| `Z_real` | float | Re(Z) in **mΩ** |
| `Z_imag` | float | Im(Z) in **mΩ** |

**Shape:** 5 × 8 × 5 = **200 Nyquist curves** (~9 800 rows).

## Hyperparameter tuning

Baseline hyperparameters in `src/models/registry.py` are intentionally conservative. To defend them with data, run:

```bash
python -m scripts.tune                 # all three tasks
python -m scripts.tune --tasks 1       # single task
```

Each task uses a **task-coherent cross-validation** strategy (no leakage):

| Task | Splitter used by `scripts.tune` | Grid |
|---|---|---|
| 1 — LOO | `LeaveOneGroupOut` on `(Aging, Temperature)`, **subsampled to the first 10 of the 40 groups** to keep the grid search tractable (~5 min vs ~20). The production benchmark in `scripts.train_all` still uses the full 40-group split. | RF, GB, KNN |
| 2 — Classification | `StratifiedGroupKFold(n_splits=3)` with groups `(Aging, Temperature)` on a **single SOC slice** (default `SOC = 3`). This is *not* the production hold-out (which excludes a whole `(Aging, SOC)` pair); a true 25-fold cross-pair tuning would multiply the grid time by 25 with little additional information. | RF, GB, SVM |
| 3 — Aging interpolation | `LeaveOneGroupOut` on `Aging` (5 groups, no subsampling — the dimension is already small enough). | RF, GB, KNN |

> **Note on Task 2 tuning.** `tune_task2` is intentionally faster than the production hold-out: it tunes the family of estimators on a representative slice (`SOC = 3` is the most informative, as `tab:task2-cross-pair` shows) and then the chosen hyperparameters are applied to whatever `(Aging, SOC)` pair the user picks at inference time. The trade-off (faster tuning vs partial coverage of the hyperparameter landscape) is documented here rather than left implicit in the code.

Best parameters are persisted to `models/tuning/<task>_best.json`. To use them at training time:

```python
from src.models.registry import regression_models
models = regression_models(use_tuned=True, task="task1")
```

The grids are declared under `tuning:` in [`config/config.yaml`](config/config.yaml) so they can be edited without touching code.

## Experiment tracking (MLflow)

Every call to `run_leave_one_out`, `run_classification`, `run_aging_interpolation` and the tuning scripts opens an **MLflow run** under a local file-backed store (`mlruns/`, gitignored). A parent run records the configuration + dataset SHA-256 + git SHA; one nested run per trained model logs parameters, metrics and duration.

```bash
python -m mlflow ui --backend-store-uri mlruns --port 5000
```

Inspect runs via the MLflow UI (`python -m mlflow ui --backend-store-uri mlruns --port 5000` → `http://localhost:5000`). No server is required for logging — the file backend works offline.

Tracking can be disabled (`tracking.enabled: false` in `config.yaml`) without touching any code: the `track_pipeline` / `log_model_run` helpers become no-ops.

## Monitoring & drift detection

Lecture 06 of the SUPSI MLOps course is about *Monitoring & Continual Learning*. The platform implements that with a small dependency-free module:

- **`src/monitoring/drift.py`** — Kolmogorov–Smirnov + Population Stability Index drift checks per numerical feature, with a dataset-level verdict.
- **`src/monitoring/prediction_log.py`** — JSON-Lines append-only store written by `POST /api/predictions/log`; doubles as audit trail and drift source.
- **`scripts/monitor.py`** — CLI that compares a reference distribution (training CSV) against a current one (CSV or JSONL) and writes `logs/drift_report.json`.
- **`GET /api/monitoring/drift`** — same logic exposed as a REST endpoint for the dashboard.

```bash
python -m scripts.monitor --current data/processed/recent_campaign.csv
# or
python -m scripts.monitor \
  --reference data/processed/batteries_cleaned_dataset.csv \
  --current   logs/predictions.jsonl --current-format jsonl
```

The full operational playbook is in [`docs/MONITORING.md`](docs/MONITORING.md).

## MLOps practices

| Concern | Lecture | Implementation |
|---|---|---|
| **Reproducibility** | L01, L02 | Versioned deps in `requirements.txt`, `random_state=42`, deterministic splits, dataset SHA-256 logged per run |
| **Configuration** | L02 | `config/config.yaml` + cached `load_config()` (tasks, tuning grids, tracking all declarative; `task2.young_max_aging` read from config — no hard-codes) |
| **Data versioning** | L02 | Raw `.mat` preserved; processed CSV regenerable; SHA tracked in MLflow tags |
| **Separation of concerns** | L02 | `data / models / evaluation / visualization / tracking / monitoring` packages |
| **Logging** | L03 | `src/logger.py` → rotating `logs/pipeline.log` (10 MB × 5 backups) + stdout |
| **Experiment tracking** | L03 | MLflow file backend (`mlruns/`), browsable via `python -m mlflow ui` |
| **Hyperparameter tuning** | L03 | GridSearchCV with task-coherent CV (`scripts/tune.py`) |
| **Artifact persistence** | L03 | `models/taskN/benchmark.json`, `models/tuning/*.json`, joblib dumps, MLflow artifacts |
| **Testing** | L04 | 81 pytest tests, ≥90 % coverage on `src/` + `app.py` — data, features, metrics, per-task pipelines, every Plotly figure, drift detection, prediction log |
| **CI** | L04 | GitHub Actions on Python 3.11 + 3.12 with ruff, pytest+coverage gate, `bandit`, `pip-audit` and Docker build smoke test |
| **Containerisation** | L02, L05 | Multi-stage Dockerfile (Node build → Python runtime) + docker-compose with mounted volumes |
| **API contract / serving** | L05 | FastAPI auto-generates an OpenAPI schema (Swagger at `/docs`, ReDoc at `/redoc`); CORS env-driven via `CORS_ALLOWED_ORIGINS` |
| **Monitoring** | L06 | KS + PSI drift detection in `src/monitoring/`; `scripts/monitor.py`; `/api/monitoring/drift`; prediction log at `logs/predictions.jsonl` |
| **DevUX** | L02 | Workflow esposti come CLI Python (`python -m scripts.{prepare_data,train_all,tune,monitor}`) + `uvicorn`, `pytest`, `ruff`, MLflow UI |
| **Shared code** | L01, L02 | Notebooks, scripts and the FastAPI bridge all call the **same** `src/` modules |

## Testing & CI

```bash
make test            # full suite + coverage gate (≥ 90 %, currently 95 %) + auto-cleanup of joblib leftovers
make test-fast       # fast subset (-m "not slow", ~70 tests, < 1 min)
make clean-cov       # wipe leftover `.coverage.*` files (joblib n_jobs=-1 stragglers)
pytest               # raw pytest (81 tests, no coverage report)
ruff check src tests scripts app.py         # lint
```

The suite covers data ingestion, feature engineering, metric contracts,
per-task smoke runs, every public Plotly figure (layout, traces,
`-Im(Z)` convention), the MATLAB → CSV preprocessor, the GridSearchCV
tuning pipeline, the drift-detection module (KS, PSI, prediction log),
the entire FastAPI HTTP layer (one test per endpoint, 22 tests in
`test_app_api.py`) and the persisted benchmark contracts of all three
tasks. GitHub Actions runs lint + tests + coverage gate (`≥ 90 %`,
currently 95 %) + `bandit` + `pip-audit` on every push / PR, on
Python 3.11 **and** 3.12, plus a `docker build` smoke test on PRs.

## Docker

```bash
# Build
docker build -t battery-mlops .

# Run (FastAPI + bundled React frontend on port 8000)
docker run --rm -p 8000:8000 battery-mlops

# or with compose
docker compose up --build
```

The `docker-compose.yml` mounts `data/`, `models/` and `logs/` so that a container run can reuse existing artefacts.

## Documentation

- 📚 **In-app docs page** — open the React app and navigate to `📚 Docs` for an interactive tour (overview, data, tasks, architecture, running, API reference, MLOps, references).
- 🧩 **OpenAPI / Swagger** — the FastAPI schema is published at `http://localhost:8000/docs` while the backend is running.
- 📄 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — project architecture + MLOps design decisions per lecture.
- 📄 [`docs/USAGE.md`](docs/USAGE.md) — commands, workflows, Python API examples.
- 📄 [`docs/MONITORING.md`](docs/MONITORING.md) — drift detection playbook.
- 📑 **Semester report** — full LaTeX write-up at [`docs/main.pdf`](docs/main.pdf); sources in [`docs/sources/`](docs/sources/) (`cd docs/sources && latexmk -pdf main.tex`).

---

*This project is a SUPSI semester project — it is meant as a teaching artefact
and a reference implementation of MLOps principles on top of a real research
pipeline.*
