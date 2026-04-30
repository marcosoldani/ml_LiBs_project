# Changelog

Tutte le modifiche degne di nota a questo progetto sono documentate in
questo file. Il formato segue [Keep a Changelog](https://keepachangelog.com/it/1.1.0/).

## [Unreleased]

### Fixed (full code-vs-doc audit)

Cross-check capillare di ogni claim tecnico nei `.md`/`.tex`/`.tsx`
contro il codice reale; allineati gli scostamenti residui:

- **`05_task1_loo.tex` Hyperparameter tuning**: la doc affermava
  `LeaveOneGroupOut` con **39 fold** e **312 fit** totali; il codice
  in `src/models/tuning.py::tune_task1` *sub-sample* a 10 dei 40
  gruppi prima del fit. Aggiornato il testo: 10 fold, 80 fit, con la
  motivazione esplicita ("tenere il grid search sotto i pochi minuti").
- **`06_task2_classification.tex` Hyperparameter tuning**: la doc
  affermava `StratifiedGroupKFold` con **5 fold**; in realtà
  `cv_folds = 3` da `config.yaml`. Inoltre il tuning gira su un
  *singolo* SOC (`task2.default_soc = 3`), non sull'hold-out di
  produzione: chiarita la differenza nella stessa sezione.
- **`07_task3_aging.tex` Hyperparameter tuning**: la doc indicava
  **4 fold LOGO** e una griglia KNN
  `n_neighbors:[3,5,7,10,15]` × `weights:[u,d]` × `p:[1,2]` con
  **20 combinazioni**; il codice usa LOGO sui **5 livelli di Aging** e
  la griglia in `config.yaml` è `n_neighbors:[5,10,15]` × `weights:[u,d]`
  → **6 combinazioni × 5 fold = 30 fit**. Allineato.
- **`Documentation.tsx`** (IT + EN): la sezione "Test e lint"
  mostrava `pytest --cov=src --cov-report=term-missing`; aggiunto
  `--cov=app` per riflettere la realtà del gate CI, e i target
  `make test`, `make test-fast`, `make ci`.

Verifiche eseguite a mano (ground truth = output di `pytest`,
`pytest --collect-only`, `ruff`, lettura diretta dei file):
81 test totali (70 fast + 11 `slow`), coverage 95 %, ruff verde,
bandit HIGH = 0, pip-audit pulito su amd64.

### Fixed (review final round)

- **Ruff verde**: corretti i 16 errori di lint che facevano cadere la
  CI:
  - `app.py:27` import inutilizzati (`classification_report`,
    `confusion_matrix`) rimossi dopo il rewrite di Task 2;
  - `scripts/make_report_figures.py` 7× `B905` (zip senza `strict=`),
    3× `E702` (semicoloni multi-statement), 3× `I001` (ordering import);
  - `tests/conftest.py:47` `SIM105` (try/except/pass →
    `contextlib.suppress`).
- **Doc drift** allineate al codice attuale:
  - README "Quick start" ora installa `requirements-dev.txt`;
  - "Project layout" lista `Makefile` e `requirements-dev.txt`;
  - "Hyperparameter tuning" tabella: chiarito che `tune_task1`
    sub-sample a 10 dei 40 LOGO group e che `tune_task2` usa
    `StratifiedGroupKFold` su un singolo SOC (≠ hold-out di
    produzione);
  - "Cheatsheet" distingue runtime vs dev install + aggiunge `make ci`;
  - schema dataset: corretto da "49 frequenze" a "49 sulla maggioranza
    delle 200 curve, 50 su 5 di esse";
  - `docs/ARCHITECTURE.md` §1 riflette lo split runtime/dev; §6 elenca
    bandit/pip-audit come **blocking** invece di "advisory" e usa
    `--cov=src --cov=app --cov-fail-under=90`;
  - `docs/USAGE.md` Installation usa `requirements-dev.txt`; Tests &
    lint allineato ai gate reali; aggiunti `make test`/`make ci`;
  - `frontend/src/pages/Documentation.tsx`: project layout e tabella
    "Reproducibility" allineate sia in IT sia in EN.
- **`cryptography>=46.0.7`** (era `>=46.0.6`): la 46.0.7 (8 aprile 2026)
  fixa CVE-2026-39892 — il claim "closes every advisory up to mid-2026"
  ora è formalmente corretto.

### Added (review final round)

- `[tool.ruff.format]` in `pyproject.toml` + target `make format`
  (Ruff format unifica black + isort + autoflake).
- `[tool.mypy]` configurato in `pyproject.toml`; `mypy` aggiunto a
  `requirements-dev.txt` (dev tool, non bloccante in CI per ora).
- `pre-commit` come dev dep + `.pre-commit-config.yaml` con hook
  trailing-whitespace, end-of-file-fixer, large-file guard, ruff
  lint+format e bandit. Installazione: `pre-commit install`.
- `make ci` riproduce localmente l'intera pipeline GitHub Actions
  (`ruff check` + `pytest --cov` + `bandit` + `pip-audit`); il passo
  `pip-audit` è "soft-fail" su arm64 con messaggio esplicito (vedi
  motivazione del pin condizionale di `cryptography`).
- `pytest-cov<7.0` (era `<6.0`): pytest-cov 6.x è stato rilasciato a
  fine 2024.

### Fixed (CI / security)

- **CI rotta**: il workflow `.github/workflows/ci.yml` chiamava
  `pip install -r requirements-dev.txt` ma il file non esisteva.
  Creato `requirements-dev.txt` con `-r requirements.txt` + dev tools
  (`pytest`, `pytest-cov`, `httpx`, `ruff`, `bandit`, `pip-audit`);
  rimossa la riga ridondante `pip install pip-audit bandit` dal
  workflow. `requirements.txt` ora contiene solo runtime deps.
- **Security gate non bloccanti**: rimossi i `|| true` da `bandit` e
  `pip-audit`. La CI ora fallisce su qualunque vulnerabilità HIGH
  trovata da bandit (`--severity-level high`) e su qualunque advisory
  trovata da `pip-audit --strict`. Bandit attualmente trova 0 issue
  HIGH; pip-audit su amd64 (la CI) è pulita perché la nuova pin
  condizionale di `cryptography` (`>=46.0.6` fuori da arm64) chiude
  tutte le advisory note.
- **`cryptography` pin condizionale per piattaforma**:
  `>=41.0,<42.0` su arm64/aarch64 (workaround SIGILL su Docker Apple
  Silicon) e **`>=46.0.6`** su x86_64. Il primo conserva la
  compatibilità con la build locale dell'autore; il secondo è la
  riga effettivamente usata dalla CI Linux e dalla maggior parte dei
  deployment, e chiude CVE-2023-50782, CVE-2024-0727,
  GHSA-h4gh-qq45-vh27, CVE-2026-26007, CVE-2026-34073, PYSEC-2024-225.
- `tests/test_models.py::_fast_regression_models`: aggiunto un commento
  esplicito che `"Bagging (Ridge)"` è uno *stub* di velocità (un Ridge
  in `MultiOutputRegressor`) e **non** un vero `BaggingRegressor` —
  serve solo a tenere lo smoke test sotto al secondo. Il vero bagging
  vive in `src/models/registry.py`.

### Fixed

- Pulizia file di build/coverage abbandonati: rimossi 16 file
  `.coverage.MacBook-…pidN.UUID` (per-process di joblib mai combinati),
  3 duplicati `.coverage 2/3/4`, 22 duplicati LaTeX `docs/sources/main 2.*`
  / `main 3.*` / `main 4.*`, 24 `.DS_Store` sparsi.
- `.gitignore` esteso con `.coverage.*` e i pattern di duplicato Finder
  (`* 2.*`, `* 3.*`, `* 4.*`).
- `tests/conftest.py` registra un `atexit` + `pytest_unconfigure` che
  spazzano via i file per-pid lasciati dai worker di scikit-learn
  (`n_jobs=-1`), e annulla la variabile d'ambiente
  `COVERAGE_PROCESS_START` per ridurre il numero di subprocess
  tracciati.
- Aggiunto `Makefile` con target `test` / `test-fast` / `lint` /
  `clean-cov` / `clean` — `make test` esegue pytest con il gate al 90 %
  e poi rimuove eventuali residui che sopravvivono al cleanup interno.

### Added

- `tests/test_app_api.py`: HTTP contract per ogni endpoint FastAPI
  (`/health`, `/api/project`, `/api/paper`, `/api/dataset/*`,
  `/api/task{1,2,3}/run`, `/api/benchmarks/*`, `/api/predictions/*`,
  `/api/monitoring/drift`) — 22 test totali.
- `tests/test_persist_benchmarks.py`: 4 test che esercitano
  `persist_default_benchmark` di Task 1/2/3 e `benchmark_full_grid`
  di Task 2 su un sottoinsieme del dataset.
- `benchmark_full_grid()` ora accetta un parametro opzionale ``df``
  per testabilità (slice ridotto invece di tutto il dataset).

### Changed

- **CI**: il gate di coverage passa da `--cov-fail-under=75` a
  `--cov-fail-under=90`, includendo anche `app.py` nel calcolo
  (`--cov=src --cov=app`).
- Suite test: 58 → **81** test, coverage globale 83 % → **95 %**.
  Gli 11 test che eseguono fit reali sono marcati `@pytest.mark.slow`
  (escludibili con `pytest -m 'not slow'`).
- I tre test sui payload invalidi di `/api/task2/run` sono ora un
  singolo test parametrizzato (3 cases). `benchmark_full_grid()` è la
  funzione canonica chiamata anche dalla figura
  `task2_cross_pair_heatmap.pdf` — niente più duplicazione fra
  `scripts/make_report_figures.py` e
  `src/models/task2_classification.py`.
- Documentazione (README, ARCHITECTURE, Documentation.tsx, capitolo 8
  LaTeX, abstract, intro) allineata al nuovo conteggio test e gate
  coverage.

## [1.2.0] — 2026-04-29

### Changed (BREAKING)

- **Task 2 — `POST /api/task2/run`**: il payload accetta ora
  `{aging, soc}` (entrambi `int` in `[0, 4]`) invece del vecchio
  `{soc}`. La nuova semantica esclude le **8 curve di Nyquist** (una
  per temperatura) della coppia `(Aging, SOC)` scelta dall'utente,
  addestra sui restanti ~9 400 punti e classifica le 8 curve come
  Young/Old. Risposta JSON modificata di conseguenza:
  - rimossi `confusion_matrix`, `classification_report`, `roc_curves`,
    `classification_map` (non applicabili: il test fold è monoclasse
    per costruzione);
  - aggiunti `aging`, `true_class`, `true_label`, `panels[]` (8 entry,
    una per temperatura, ognuna con `temperature`, `n_points`,
    `n_correct`, `accuracy`, `is_correct: bool`, `groups[]`),
    `per_temperature[]`.
- `src/models/task2_classification.run_classification(soc)` →
  `run_classification(aging, soc)`.
- `config.task2.default_aging` aggiunto (default `4`); usato dal nuovo
  benchmark di default insieme a `default_soc`.

### Performance

- `persist_default_benchmark()` ora addestra una sola coppia
  (`task2.default_aging`, `task2.default_soc` da config) invece dell'intero
  prodotto cartesiano $5 \times 5$: il benchmark di Task 2 scende da
  ~12 min a ~7 s. Per la heatmap cross-pair del report è disponibile la
  nuova funzione `benchmark_full_grid()` (chiamata esplicita,
  out-file separato `benchmark_full_grid.json`).

### Added

- `src/models/task2_classification.benchmark_full_grid()` per i 25
  hold-out — non invocato da `scripts.train_all`, ma utile per
  rigenerare la figura `task2_cross_pair_heatmap.pdf`.
- Frontend Task 2: due dropdown (Aging + SOC) e griglia di 8 plot di
  Nyquist colorati per classe predetta.
- Documentazione (online + LaTeX) riformulata: "predizione delle 8
  curve di Nyquist (una per temperatura) della coppia (Aging, SOC)
  esclusa", invece della precedente "classificazione al SOC scelto".
- Figure aggiuntive nel report: `task2_cross_pair_heatmap.pdf`
  (heatmap 5×5 dell'accuracy sui 25 hold-out possibili) e
  `task1_residuals_per_soc.pdf` (boxplot residui per SOC).
- Capitolo LaTeX 06 riscritto con tabella `tab:task2-cross-pair`
  (matrice di accuracy su tutti i $(Aging, SOC)$).
- `tests/test_app_api.py`: smoke test HTTP per `/health` e
  `/api/task2/run` (payload nuovo accettato, vecchio rifiutato 422,
  out-of-range rifiutato 422).

### Fixed

- Dark mode: `.banner`, `.banner strong`, `.banner-success`,
  `.banner-error` ora usano `var(--text)` o override
  `[data-theme='dark']` dedicate (testi non più invisibili su sfondo
  scuro). `.page-title` non usa più gradient con stop hardcoded.
- Pagina acronimi del report: `\setglossarystyle{long}` + niente
  numeri di pagina + `\glsdescwidth` ridotto, spaziature compattate.
- Capitolo 2 (background) snellito: rimossa la trattazione dettagliata
  di GEIS / modello a circuito equivalente, esplicitato che sono di
  competenza del Politecnico.
- Abstract IT/EN allineati alla nuova formulazione Task 2.

### Removed

- Dead i18n keys: `task2.predClass`, `task2.classCol`,
  `task2.randomLabel`.
- Import `joblib` non usato in `src/models/task2_classification.py`.
- Emoji da titoli/sezioni e da etichette del frontend (logo, bandiere
  lingua, header documentazione). Rimangono solo le icone funzionali
  del bottone tema (luna/sole).

## [1.1.0] — Precedente

Stato iniziale del rapporto e della piattaforma MLOps prima della
revisione. Vedere `docs/main.pdf` storico per dettagli.
