"""FastAPI backend for the Battery GEIS MLOps platform.

Exposes the pipelines defined in ``src/`` over HTTP so the React frontend in
``frontend/`` can drive them. The Python ML logic in ``src/`` is reused
verbatim — this module only adds an HTTP layer on top.

Run locally::

    uvicorn app:app --reload --port 8000

The OpenAPI schema is published at ``/docs`` (Swagger) and ``/redoc``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, load_config, models_path
from src.constants import (
    AGING_COLORS,
    AGING_LABELS,
    CLASS_COLORS,
    CLASS_NAMES,
    SOC_LABELS,
    TEMP_COLORS,
)
from src.data.loader import dataset_summary, load_dataset
from src.models.task1_loo import run_leave_one_out
from src.models.task2_classification import run_classification
from src.models.task3_aging import run_aging_interpolation
from src.monitoring import detect_dataset_drift, log_prediction, read_predictions

_DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:4173,http://localhost:3000"
)


def _cors_origins() -> list[str]:
    """Return CORS origins from ``CORS_ALLOWED_ORIGINS`` env var, comma-separated.

    Default keeps the project working out-of-the-box on localhost (Vite dev
    on :5173, preview on :4173, generic dev on :3000). In production, set
    ``CORS_ALLOWED_ORIGINS`` to the public host(s) of the React frontend.
    """
    raw = os.getenv("CORS_ALLOWED_ORIGINS", _DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


# ── App & CORS ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Battery GEIS MLOps API",
    description=(
        "HTTP layer over the Battery GEIS MLOps pipelines (Task 1 LOO, "
        "Task 2 Young/Old classification, Task 3 Aging interpolation). "
        "Wraps the Python modules in ``src/`` for the React frontend."
    ),
    version=load_config().project.version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────────
PAPER_PATH = PROJECT_ROOT / "docs" / "Polimi_THERMINIC_2025.pdf"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas objects to JSON-friendly Python."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return _to_jsonable(df.replace({np.nan: None}).to_dict(orient="records"))


# ── Schemas ─────────────────────────────────────────────────────────────────
class Task1Request(BaseModel):
    aging: int = Field(..., ge=0, le=4, description="Aging level to hold out (0–4)")
    temperature: float = Field(..., description="Temperature in °C to hold out")


class Task2Request(BaseModel):
    aging: int = Field(..., ge=0, le=4, description="Aging level to hold out (0–4)")
    soc: int = Field(..., ge=0, le=4, description="State-of-charge index to hold out (0–4)")


class Task3Request(BaseModel):
    excluded_aging: int = Field(..., ge=0, le=4, description="Aging level to fully hold out")


# ── General endpoints ───────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/project")
def project_info() -> dict[str, Any]:
    cfg = load_config()
    return {
        "name": cfg.project.name,
        "version": cfg.project.version,
        "description": cfg.project.description,
        "authors": list(cfg.project.authors),
        "paper": {
            "title": (
                "Hysteresis Phenomenon in the Electric Parameters of "
                "Lithium-Ion Batteries under Temperature Effects"
            ),
            "venue": "IEEE THERMINIC 2025",
            "doi": "10.1109/THERMINIC65879.2025.11216945",
            "url": "https://doi.org/10.1109/THERMINIC65879.2025.11216945",
            "available": PAPER_PATH.exists(),
        },
        "constants": {
            "aging_labels": AGING_LABELS,
            "aging_colors": AGING_COLORS,
            "soc_labels": SOC_LABELS,
            "temp_colors": {str(k): v for k, v in TEMP_COLORS.items()},
            "class_names": CLASS_NAMES,
            "class_colors": CLASS_COLORS,
        },
    }


@app.get("/api/paper")
def paper_pdf():
    if not PAPER_PATH.exists():
        raise HTTPException(404, "Paper PDF not bundled in this build.")
    return FileResponse(
        PAPER_PATH,
        media_type="application/pdf",
        filename="Polimi_THERMINIC_2025.pdf",
    )


# ── Dataset endpoints ───────────────────────────────────────────────────────
@app.get("/api/dataset/summary")
def get_dataset_summary() -> dict[str, Any]:
    df = load_dataset()
    return _to_jsonable(dataset_summary(df))


@app.get("/api/dataset/options")
def get_dataset_options() -> dict[str, Any]:
    df = load_dataset()
    summary = dataset_summary(df)
    return {
        "agings": summary["agings"],
        "temperatures": summary["temperatures"],
        "socs": summary["socs"],
    }


@app.get("/api/dataset/curves")
def get_dataset_curves(aging: int) -> dict[str, Any]:
    """Nyquist scatter at the given aging level, grouped by temperature."""
    df = load_dataset()
    sub = df[df["Aging"] == aging]
    if sub.empty:
        raise HTTPException(404, f"No rows for Aging={aging}")
    series: list[dict[str, Any]] = []
    for temp in sorted(sub["Temperature"].unique()):
        rows = sub[sub["Temperature"] == temp].sort_values("Frequency", ascending=False)
        series.append(
            {
                "temperature": float(temp),
                "color": TEMP_COLORS.get(float(temp), "#444"),
                "z_real": rows["Z_real"].tolist(),
                "z_imag_neg": (-rows["Z_imag"]).tolist(),
                "soc": rows["SOC"].astype(int).tolist(),
                "frequency": rows["Frequency"].tolist(),
            }
        )
    return {"aging": aging, "series": series}


@app.get("/api/dataset/agg-by-temp")
def get_agg_by_temp() -> dict[str, Any]:
    """Mean Re(Z) per (Aging, Temperature). Used for the Home temperature line plot."""
    df = load_dataset()
    agg = (
        df.groupby(["Aging", "Temperature"])[["Z_real", "Z_imag"]]
        .mean()
        .reset_index()
    )
    series: list[dict[str, Any]] = []
    for aging in sorted(agg["Aging"].unique()):
        rows = agg[agg["Aging"] == aging].sort_values("Temperature")
        series.append(
            {
                "aging": int(aging),
                "label": AGING_LABELS.get(int(aging), f"Aging {aging}"),
                "color": AGING_COLORS.get(int(aging), "#333"),
                "temperature": rows["Temperature"].tolist(),
                "z_real_mean": rows["Z_real"].tolist(),
            }
        )
    return {"series": series}


@app.get("/api/dataset/aging-evolution")
def get_aging_evolution(soc: int, excluded_aging: int) -> dict[str, Any]:
    """Task 3 EDA — impedance curves vs aging at three reference temperatures."""
    df = load_dataset()
    panels: list[dict[str, Any]] = []
    for temp in (20.0, 30.0, 40.0):
        traces: list[dict[str, Any]] = []
        for aging in sorted(df["Aging"].unique()):
            rows = df[
                (df["Aging"] == aging)
                & (df["Temperature"] == temp)
                & (df["SOC"] == soc)
            ].sort_values("Frequency", ascending=False)
            if rows.empty:
                continue
            traces.append(
                {
                    "aging": int(aging),
                    "label": AGING_LABELS.get(int(aging), f"Aging {aging}"),
                    "color": AGING_COLORS.get(int(aging), "#333"),
                    "dashed": int(aging) == int(excluded_aging),
                    "z_real": rows["Z_real"].tolist(),
                    "z_imag_neg": (-rows["Z_imag"]).tolist(),
                }
            )
        panels.append({"temperature": temp, "traces": traces})
    return {"soc": soc, "excluded_aging": excluded_aging, "panels": panels}


# ── Task pipelines ──────────────────────────────────────────────────────────
def _task1_payload(result: Any, focus_soc: int | None = None) -> dict[str, Any]:
    df_test = result.df_test
    socs_in_test = sorted(int(s) for s in df_test["SOC"].unique())
    by_soc: dict[int, dict[str, Any]] = {}
    for soc in socs_in_test:
        mask = df_test["SOC"].values == soc
        rows = df_test.loc[mask].sort_values("Frequency", ascending=False)
        idx = rows.index.values
        by_soc[soc] = {
            "soc": soc,
            "frequency": rows["Frequency"].tolist(),
            "z_real_actual": rows["Z_real"].tolist(),
            "z_imag_neg_actual": (-rows["Z_imag"]).tolist(),
            "z_real_pred": result.best_predictions[idx, 0].tolist(),
            "z_imag_neg_pred": (-result.best_predictions[idx, 1]).tolist(),
        }

    residuals = []
    for soc in socs_in_test:
        mask = df_test["SOC"].values == soc
        err = result.best_predictions[mask] - result.y_test[mask]
        residuals.append(
            {
                "soc": soc,
                "mae_real": float(np.mean(np.abs(err[:, 0]))),
                "mae_imag": float(np.mean(np.abs(err[:, 1]))),
                "max_abs": float(np.max(np.abs(err))),
                "n_points": int(mask.sum()),
            }
        )

    return {
        "task": "task1",
        "excluded_aging": int(result.excluded_aging),
        "excluded_temperature": float(result.excluded_temperature),
        "best_model_name": result.best_model_name,
        "metrics_per_model": _to_jsonable(result.metrics_per_model),
        "training_time_s": _to_jsonable(result.training_time),
        "n_test_points": int(len(result.y_test)),
        "available_socs": socs_in_test,
        "focus_soc": int(focus_soc) if focus_soc is not None else socs_in_test[0],
        "predictions_by_soc": by_soc,
        "residuals_by_soc": residuals,
    }


def _task2_payload(result: Any) -> dict[str, Any]:
    best_pred = result.predictions[result.best_model_name]
    df_test = result.df_test
    y_test = np.asarray(result.y_test)
    true_class = int(result.true_class)

    # One Nyquist panel per temperature: every panel holds the ~49 frequency
    # points of the held-out (Aging, SOC) at that temperature, split into the
    # two predicted-class groups so each group can be coloured.
    panels: list[dict[str, Any]] = []
    for temp in sorted(df_test["Temperature"].unique()):
        mask = (df_test["Temperature"] == temp).values
        sub = df_test.loc[mask].reset_index(drop=True)
        pred_temp = best_pred[mask]
        n_total = int(mask.sum())
        n_correct = int(np.sum(pred_temp == true_class))

        groups: list[dict[str, Any]] = []
        for cls, label, color in zip([0, 1], CLASS_NAMES, CLASS_COLORS, strict=False):
            m = pred_temp == cls
            groups.append(
                {
                    "class": int(cls),
                    "label": label,
                    "color": color,
                    "z_real": sub.loc[m, "Z_real"].tolist(),
                    "z_imag_neg": (-sub.loc[m, "Z_imag"]).tolist(),
                    "frequency": sub.loc[m, "Frequency"].tolist(),
                }
            )

        majority = int(np.bincount(pred_temp, minlength=2).argmax())
        panels.append(
            {
                "temperature": float(temp),
                "n_points": n_total,
                "n_correct": n_correct,
                "accuracy": (n_correct / n_total) if n_total else 0.0,
                "is_correct": majority == true_class,
                "groups": groups,
            }
        )

    return {
        "task": "task2",
        "aging": int(result.aging),
        "soc": int(result.soc),
        "true_class": true_class,
        "true_label": result.true_label,
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
        "best_model_name": result.best_model_name,
        "metrics_per_model": _to_jsonable(result.metrics_per_model),
        "training_time_s": _to_jsonable(result.training_time),
        "n_test_points": int(len(y_test)),
        "panels": panels,
        "per_temperature": _to_jsonable(result.per_temperature.to_dict(orient="records")),
        "feature_importance": (
            _to_jsonable(result.feature_importance)
            if result.feature_importance
            else None
        ),
        "accuracy_full": float(np.mean(best_pred == y_test)),
        "errors_full": int(np.sum(best_pred != y_test)),
        "n_full": int(len(y_test)),
    }


def _task3_payload(result: Any) -> dict[str, Any]:
    err = result.df_test.copy()
    err["err_real"] = err["Z_real_pred"] - err["Z_real"]
    err["err_imag"] = err["Z_imag_pred"] - err["Z_imag"]
    err["err_abs"] = np.sqrt(err["err_real"] ** 2 + err["err_imag"] ** 2)

    # Grid 8 temperatures × 5 SOC for the actual-vs-predicted plot
    temps = sorted(result.df_test["Temperature"].unique().tolist())
    socs = sorted(result.df_test["SOC"].unique().tolist())
    grid: list[dict[str, Any]] = []
    for temp in temps:
        row_panels = []
        for soc in socs:
            sub = result.df_test[
                (result.df_test["Temperature"] == temp)
                & (result.df_test["SOC"] == soc)
            ].sort_values("Frequency", ascending=False)
            if sub.empty:
                row_panels.append(None)
                continue
            row_panels.append(
                {
                    "temperature": float(temp),
                    "soc": int(soc),
                    "color": TEMP_COLORS.get(float(temp), "#555"),
                    "z_real_actual": sub["Z_real"].tolist(),
                    "z_imag_neg_actual": (-sub["Z_imag"]).tolist(),
                    "z_real_pred": sub["Z_real_pred"].tolist(),
                    "z_imag_neg_pred": (-sub["Z_imag_pred"]).tolist(),
                }
            )
        grid.append({"temperature": float(temp), "cells": row_panels})

    err_freq = (
        err[["Frequency", "err_abs", "SOC"]]
        .rename(columns={"err_abs": "err"})
        .to_dict(orient="records")
    )
    return {
        "task": "task3",
        "excluded_aging": int(result.excluded_aging),
        "best_model_name": result.best_model_name,
        "metrics_per_model": _to_jsonable(result.metrics_per_model),
        "training_time_s": _to_jsonable(result.training_time),
        "n_test_points": int(len(result.y_test)),
        "per_temperature": _df_records(result.per_temperature),
        "grid": grid,
        "error_distribution": err["err_abs"].tolist(),
        "error_p95": float(np.percentile(err["err_abs"], 95)),
        "error_vs_freq": _to_jsonable(err_freq),
        "temperatures": [float(t) for t in temps],
        "socs": [int(s) for s in socs],
    }


@app.post("/api/task1/run")
def task1_run(req: Task1Request) -> dict[str, Any]:
    try:
        result = run_leave_one_out(aging=int(req.aging), temperature=float(req.temperature))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _task1_payload(result)


@app.post("/api/task2/run")
def task2_run(req: Task2Request) -> dict[str, Any]:
    try:
        result = run_classification(aging=int(req.aging), soc=int(req.soc))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _task2_payload(result)


@app.post("/api/task3/run")
def task3_run(req: Task3Request) -> dict[str, Any]:
    try:
        result = run_aging_interpolation(excluded_aging=int(req.excluded_aging))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _task3_payload(result)


# ── Persisted benchmarks ────────────────────────────────────────────────────
@app.get("/api/benchmarks/{task}")
def get_benchmark(task: str) -> dict[str, Any]:
    if task not in {"task1", "task2", "task3"}:
        raise HTTPException(404, "Unknown task. Use task1 / task2 / task3.")
    path = models_path() / task / "benchmark.json"
    if not path.exists():
        raise HTTPException(404, f"No persisted benchmark at {path}")
    return json.loads(path.read_text())


# ── Monitoring (Lecture 06) ─────────────────────────────────────────────────
class PredictionLogRequest(BaseModel):
    task: str = Field(..., description="One of task1 / task2 / task3")
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/predictions/log")
def predictions_log(req: PredictionLogRequest) -> dict[str, Any]:
    """Append a prediction event to ``logs/predictions.jsonl`` (audit + drift source)."""
    if req.task not in {"task1", "task2", "task3"}:
        raise HTTPException(400, "task must be task1, task2 or task3")
    target = log_prediction(
        task=req.task,
        inputs=req.inputs,
        outputs=req.outputs,
        extra=req.extra,
    )
    return {"status": "logged", "path": str(target)}


@app.get("/api/predictions/recent")
def predictions_recent(limit: int = 50) -> dict[str, Any]:
    """Return the last ``limit`` records from the prediction log (default: 50)."""
    records = read_predictions()
    return {"count": len(records), "records": records[-limit:]}


@app.get("/api/monitoring/drift")
def monitoring_drift(
    feature: str | None = None,
    soc: int | None = None,
) -> dict[str, Any]:
    """Compare a recent slice of the dataset against the global reference.

    This is a *demo* endpoint: in absence of new measurements, we use the
    rows at the requested ``soc`` (or the highest aging level if none is
    given) as the *current* distribution and the rest as *reference*. It is
    enough to demonstrate drift detection wiring end-to-end.
    """
    df = load_dataset()
    if soc is not None:
        current = df[df["SOC"] == soc]
        reference = df[df["SOC"] != soc]
    else:
        current = df[df["Aging"] == int(df["Aging"].max())]
        reference = df[df["Aging"] != int(df["Aging"].max())]
    if reference.empty or current.empty:
        raise HTTPException(400, "Empty reference or current slice")
    features = [feature] if feature else None
    report = detect_dataset_drift(reference, current, features=features)
    return report.to_dict()


# ── Static frontend (production build, optional) ────────────────────────────
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="frontend",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
