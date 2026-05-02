.PHONY: test test-fast lint format ci clean clean-cov api frontend

PYTHON ?= python

# Run the full test suite with coverage gate, then sweep any joblib
# worker leftovers. scikit-learn's `n_jobs=-1` spawns subprocesses that
# write `.coverage.HOSTNAME.pidN.UUID` files which pytest-cov does not
# always combine on macOS — the trailing `find -delete` removes them.
test:
	$(PYTHON) -m pytest --cov=src --cov=app --cov-report=term-missing --cov-fail-under=90
	@find . -maxdepth 1 -name ".coverage.*" -delete

# Fast subset (skips `slow` markers) — useful for the dev loop.
test-fast:
	$(PYTHON) -m pytest -m "not slow"

lint:
	$(PYTHON) -m ruff check src tests scripts app.py

# Auto-format the codebase (Ruff replaces black + isort + autoflake).
format:
	$(PYTHON) -m ruff format src tests scripts app.py
	$(PYTHON) -m ruff check --fix src tests scripts app.py

# Reproduce the GitHub Actions pipeline locally — useful before pushing.
# pip-audit is run last, with `-` to keep make from aborting on arm64
# Macs: there the conditional pin selects the SIGILL-safe cryptography
# 41.x line which still carries old advisories. CI (Linux amd64) picks
# the >=46.0.7 line and is fully clean.
ci: lint test
	bandit -r src app.py --severity-level high
	@echo "→ Running pip-audit (on arm64 this will report cryptography"
	@echo "  advisories; CI runs on amd64 where ≥ 46.0.7 is installed.)"
	-pip-audit -r requirements.txt --strict

# Remove orphan coverage data files (joblib workers, parallel mode).
clean-cov:
	@find . -maxdepth 1 -name ".coverage.*" -delete
	@rm -f .coverage

# Wipe build artefacts, caches, coverage, mypy.
clean: clean-cov
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name "*.py[cod]" -delete
	@rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov
	@find . -name ".DS_Store" -delete

api:
	uvicorn app:app --reload --port 8000

frontend:
	cd frontend && npm run dev
