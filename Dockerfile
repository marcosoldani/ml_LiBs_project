# syntax=docker/dockerfile:1.6
# Multi-stage build: Node compiles the React frontend, Python runs FastAPI
# and serves the static bundle.

# ── Stage 1: build the frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund

COPY frontend ./
RUN npm run build


# ── Stage 2: Python API ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src ./src
COPY app.py ./
COPY scripts ./scripts
COPY config ./config
COPY data ./data
COPY README.md ./

# Bring in the compiled React bundle so FastAPI mounts it at "/"
COPY --from=frontend /frontend/dist ./frontend/dist

RUN mkdir -p logs models outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
  CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
