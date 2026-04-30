"""HTTP-layer smoke tests for the FastAPI endpoints in ``app.py``.

These tests exercise the request/response contract — body validation,
key shape of the returned JSON, basic invariants — without re-testing
the modelling logic that is already covered in ``test_models.py``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app import app

    return TestClient(app)


# ── General endpoints ──────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_project_info(client):
    r = client.get("/api/project")
    assert r.status_code == 200
    body = r.json()
    assert {"name", "version", "description", "authors", "paper", "constants"} <= set(body)
    assert isinstance(body["authors"], list)
    assert {"title", "venue", "doi", "url", "available"} <= set(body["paper"])
    assert {"aging_labels", "aging_colors", "soc_labels", "temp_colors",
            "class_names", "class_colors"} <= set(body["constants"])


def test_paper_pdf(client):
    """`/api/paper` either streams the bundled PDF or 404s cleanly.

    The PDF is bundled in the repo by default but is *optional* in
    Docker images that strip large binaries — both outcomes are valid.
    """
    r = client.get("/api/paper")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.headers["content-type"] == "application/pdf"


# ── Dataset endpoints ──────────────────────────────────────────────────────


def test_dataset_summary(client):
    r = client.get("/api/dataset/summary")
    assert r.status_code == 200
    body = r.json()
    assert {"rows", "columns", "agings", "temperatures", "socs",
            "freq_min", "freq_max", "n_combinations", "n_curves"} <= set(body)
    assert body["rows"] > 0


def test_dataset_options(client):
    r = client.get("/api/dataset/options")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"agings", "temperatures", "socs"}
    assert len(body["agings"]) == 5
    assert len(body["socs"]) == 5


def test_dataset_curves_ok(client):
    r = client.get("/api/dataset/curves", params={"aging": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["aging"] == 0
    assert isinstance(body["series"], list) and len(body["series"]) > 0
    s0 = body["series"][0]
    assert {"temperature", "color", "z_real", "z_imag_neg", "soc",
            "frequency"} <= set(s0)


def test_dataset_curves_unknown_aging(client):
    r = client.get("/api/dataset/curves", params={"aging": 99})
    assert r.status_code == 404


def test_dataset_agg_by_temp(client):
    r = client.get("/api/dataset/agg-by-temp")
    assert r.status_code == 200
    body = r.json()
    assert "series" in body
    s = body["series"][0]
    assert {"aging", "label", "color", "temperature", "z_real_mean"} <= set(s)


def test_dataset_aging_evolution(client):
    r = client.get(
        "/api/dataset/aging-evolution",
        params={"soc": 2, "excluded_aging": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["soc"] == 2 and body["excluded_aging"] == 2
    assert len(body["panels"]) == 3  # three reference temperatures


# ── Task pipelines ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_task1_run_endpoint(client):
    """`POST /api/task1/run` returns the LOO payload for an excluded pair."""
    r = client.post(
        "/api/task1/run",
        json={"aging": 2, "temperature": 22.5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "task1"
    assert body["excluded_aging"] == 2
    assert body["excluded_temperature"] == 22.5
    assert body["best_model_name"] in body["metrics_per_model"]
    assert body["n_test_points"] > 0
    assert isinstance(body["available_socs"], list)
    assert isinstance(body["predictions_by_soc"], dict)


def test_task1_run_endpoint_invalid_temperature(client):
    """A temperature not present in the dataset → 400."""
    r = client.post(
        "/api/task1/run",
        json={"aging": 0, "temperature": 999.0},
    )
    assert r.status_code == 400


@pytest.mark.slow
def test_task2_run_endpoint_returns_8_panels(client):
    """`POST /api/task2/run` with `{aging, soc}` → 8 Nyquist panels."""
    r = client.post("/api/task2/run", json={"aging": 4, "soc": 3})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["task"] == "task2"
    assert body["aging"] == 4 and body["soc"] == 3
    assert body["true_label"] in {"Young", "Old"}
    assert body["true_class"] in (0, 1)
    assert body["best_model_name"] in body["metrics_per_model"]

    panels = body["panels"]
    assert len(panels) == 8
    for p in panels:
        assert {"temperature", "n_points", "n_correct", "accuracy",
                "is_correct", "groups"} <= set(p)
        assert 0.0 <= p["accuracy"] <= 1.0
        assert isinstance(p["is_correct"], bool)
        assert p["n_correct"] <= p["n_points"]


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"soc": 2}, 422),                         # pre-rewrite payload
        ({"aging": 9, "soc": 3}, 422),             # aging out of range
        ({"aging": 2, "soc": 9}, 422),             # soc out of range
    ],
    ids=["missing-aging", "aging-out-of-range", "soc-out-of-range"],
)
def test_task2_run_endpoint_rejects_invalid_payload(client, payload, expected):
    """Pydantic validates the request body shape before any modelling runs."""
    r = client.post("/api/task2/run", json=payload)
    assert r.status_code == expected


@pytest.mark.slow
def test_task3_run_endpoint(client):
    """`POST /api/task3/run` returns the aging-interpolation payload."""
    r = client.post("/api/task3/run", json={"excluded_aging": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "task3"
    assert body["excluded_aging"] == 2
    assert body["best_model_name"] in body["metrics_per_model"]
    assert isinstance(body["per_temperature"], list)
    assert isinstance(body["grid"], list)


# ── Benchmarks ─────────────────────────────────────────────────────────────


def test_benchmark_unknown_task(client):
    r = client.get("/api/benchmarks/task9")
    assert r.status_code == 404


@pytest.mark.slow
def test_benchmark_existing_task(client, tmp_path, monkeypatch):
    """`GET /api/benchmarks/task2` returns the persisted benchmark JSON.

    To make the test deterministic (independent of whether
    ``scripts.train_all`` has ever been executed in the working tree),
    we redirect ``models_path()`` to a tempdir and write a freshly
    generated benchmark there before the request.
    """
    from src.models import task2_classification

    monkeypatch.setattr(task2_classification, "models_path", lambda: tmp_path)
    monkeypatch.setattr(task2_classification, "TASK_DIR", tmp_path / "task2")

    import app as app_module

    monkeypatch.setattr(app_module, "models_path", lambda: tmp_path)

    task2_classification.persist_default_benchmark()

    r = client.get("/api/benchmarks/task2")
    assert r.status_code == 200
    body = r.json()
    assert {"aging", "soc", "best_model", "metrics"} <= set(body)


# ── Monitoring + prediction log ────────────────────────────────────────────


def test_monitoring_drift_default(client):
    r = client.get("/api/monitoring/drift")
    assert r.status_code == 200
    body = r.json()
    # `to_dict()` from src.monitoring.drift returns nested feature reports.
    assert isinstance(body, dict)


def test_monitoring_drift_by_soc(client):
    r = client.get("/api/monitoring/drift", params={"soc": 0, "feature": "Z_real"})
    assert r.status_code == 200


def test_monitoring_drift_invalid_slice(client):
    """A SOC that yields an empty slice should 400."""
    r = client.get("/api/monitoring/drift", params={"soc": 99})
    assert r.status_code == 400


def test_predictions_log_and_recent(client, tmp_path, monkeypatch):
    """Append a prediction event then read it back via `/api/predictions/recent`."""
    from src.monitoring import prediction_log as pl

    # Redirect ``logs_path()`` to a tempdir so the audit trail is hermetic.
    # NB: this works because ``prediction_log.py`` imports ``logs_path`` as
    # ``from src.config import logs_path`` (binding a module-local name).
    # If a future refactor switches to ``import src.config`` and references
    # ``src.config.logs_path()`` at call site, this monkeypatch will silently
    # become a no-op — patch ``src.config.logs_path`` instead.
    monkeypatch.setattr(pl, "logs_path", lambda: tmp_path)

    payload = {
        "task": "task2",
        "inputs": {"aging": 4, "soc": 3},
        "outputs": {"prediction": "Old"},
        "extra": {"source": "unit-test"},
    }
    r = client.post("/api/predictions/log", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "logged"

    r2 = client.get("/api/predictions/recent", params={"limit": 5})
    assert r2.status_code == 200
    data = r2.json()
    assert data["count"] >= 1
    last = data["records"][-1]
    assert last["task"] == "task2"


def test_predictions_log_invalid_task(client):
    r = client.post(
        "/api/predictions/log",
        json={"task": "task9", "inputs": {}, "outputs": {}, "extra": {}},
    )
    assert r.status_code == 400
