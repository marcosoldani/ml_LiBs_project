# Monitoring playbook

Maps the SUPSI **Lecture 06 — Monitoring & Continual Learning** syllabus to
the components shipped in `src/monitoring/`, `scripts/monitor.py` and the
FastAPI bridge.

## What we monitor

| Concern | Signal | Component |
|---|---|---|
| **Input drift** | Distribution of `Aging`, `Temperature`, `SOC`, `Frequency`, `Z_real`, `Z_imag` between training data and freshly collected campaigns | `src/monitoring/drift.py::detect_dataset_drift` |
| **Prediction drift** | Distribution of model outputs (e.g. `Z_real_pred`, `Z_imag_pred` for Task 1/3 or class probabilities for Task 2) | Same module, fed by `logs/predictions.jsonl` |
| **Audit trail** | Every API call (inputs + outputs + timestamp + extra metadata) | `src/monitoring/prediction_log.py` |

## Drift detection methods

Two complementary tests run side-by-side per feature:

- **Kolmogorov–Smirnov (KS)** — non-parametric two-sample test on the
  empirical CDF. Sensitive to *any* shape change. Threshold: `α = 0.05`.
- **Population Stability Index (PSI)** — bin-based divergence used heavily
  in industrial monitoring. Common interpretation:
  - PSI < 0.10 → no significant change
  - 0.10 ≤ PSI < 0.25 → minor drift
  - PSI ≥ 0.25 → major drift
  Default threshold: `0.25`.

A feature is flagged as drifted if **either** test fires. The dataset-level
verdict is `drift_detected = True` when at least 25 % of the analysed
features individually drift.

## Operational workflows

### Offline (batch) check

```bash
python -m scripts.monitor \
  --reference data/processed/batteries_cleaned_dataset.csv \
  --current   data/processed/recent_campaign.csv
```

Writes `logs/drift_report.json` with the per-feature scores plus the
aggregated verdict. Useful when a new battery characterisation is uploaded.

### Online check against deployed traffic

1. The frontend (or any API consumer) calls
   `POST /api/predictions/log` after each `POST /api/taskN/run`, attaching
   the inputs, outputs and any context (user, environment, …).
2. Records accumulate in `logs/predictions.jsonl`.
3. A daily / weekly job runs

```bash
python -m scripts.monitor \
  --current logs/predictions.jsonl --current-format jsonl
```

   to compute drift between the *training* feature distribution and the
   *deployed* one.

### One-shot demo via the API

```http
GET /api/monitoring/drift
GET /api/monitoring/drift?soc=4
GET /api/monitoring/drift?feature=Z_real
```

Returns the JSON drift report directly to the caller; convenient for the
React dashboard.

## What to do when drift fires

1. **Inspect the report**: which features moved? Is it instrumentation
   noise (e.g. a new battery sample) or a genuine population change?
2. **Snapshot the offending data** to `data/processed/` with a date
   suffix and re-run the training pipeline
   (`python -m scripts.train_all`) to refresh the benchmarks.
3. **Promote the new baseline**: copy `models/taskN/benchmark.json` next
   to the date-stamped one in `models/history/` and update the
   `models/taskN/last_run.joblib`.
4. **Document the incident** in `CHANGELOG.md` under the appropriate
   release.

The full *continual-learning* loop (drift → automated retraining →
redeployment) is left as future work; the current playbook covers the
detection and human-in-the-loop refresh that Lecture 06 asks for.
