import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useT, useLang, type Lang } from '../i18n'

type TabKey = 'overview' | 'paper' | 'data' | 'tasks' | 'arch' | 'run' | 'api' | 'mlops' | 'tracking' | 'refs'

const TAB_KEYS: TabKey[] = ['overview', 'paper', 'data', 'tasks', 'arch', 'run', 'api', 'mlops', 'tracking', 'refs']

const CONTENT: Record<Lang, Record<TabKey, string>> = {
  it: {
    overview: `
## Contesto

Questo progetto è il **Progetto di Semestre SUPSI** di Davide Corso e Marco Soldani,
svolto in collaborazione con il **Politecnico di Milano**.
Il punto di partenza è la metodologia sperimentale del paper

> *"Hysteresis Phenomenon in the Electric Parameters of Lithium-Ion Batteries under
> Temperature Effects"* — S. Barcellona, L. Codecasa, S. Colnago, D. D'Amore (**THERMINIC 2025**)

intorno alla quale abbiamo costruito una **pipeline MLOps end-to-end**: backend
FastAPI, frontend React, training riproducibile, experiment tracking, tuning di
iperparametri, suite di test e CI.

La fisica sottostante è la Galvanostatic Electrochemical Impedance Spectroscopy
(GEIS) su una **pouch LiCoO₂ da 10 Ah**, misurata su **8 temperature**, **5
livelli di SOC** e **5 stati di invecchiamento**, ovvero **200 curve di Nyquist**.

## Obiettivi

1. Ricostruire una curva di Nyquist di una **(Aging, Temperatura)** mancante a partire dalle altre 39 (Leave-One-Out).
2. Classificare come **Young vs Old** le 8 curve di Nyquist (una per temperatura) di una coppia (Aging, SOC) **mai vista in addestramento** (Young-Old).
3. Predire un **intero livello di Aging** (default: Aging 2) a partire dagli altri quattro (Leave-One-Aging-Out).

## Come si incastrano i pezzi

\`\`\`
   config.yaml ── load_config()
        │
        ▼
 data/raw/GEIS.mat ── prepare_data ──▶ data/processed/batteries_cleaned_dataset.csv
                                          │
                                          ▼
                                   load_dataset() ──▶ features.py ──▶ models/task{1,2,3}.py
                                                                          │
                                     tuning.py ─────────────────▶  models/tuning/*.json
                                                                          │
                                                                          ▼
                                                                   tracking.py ──▶ mlruns/
                                                                          │
                                                                          ▼
                                                          app.py (FastAPI)  ◀──▶  frontend/ (React)
\`\`\`
`,
    paper: `
## Il paper in breve

> *S. Barcellona, L. Codecasa, S. Colnago, D. D'Amore — Politecnico di Milano,
> **THERMINIC 2025**, DOI \`10.1109/THERMINIC65879.2025.11216945\`.*

### La domanda
I **parametri RC** del modello a circuito equivalente di una cella Litio-Ione si
comportano allo stesso modo durante carica e scarica? E come cambia questa
asimmetria al variare della temperatura?

### Modello a circuito equivalente
L'impedenza della cella in ogni punto (SOC, T) è fittata con un modello a tre blocchi:

\`\`\`
Z_bat(ω) = R_s + R_mid / [1 + (jω)^β_mid · R_mid · Q_mid] + A_w / (jω)^β_w
\`\`\`

- **R_s** — resistenza ohmica ad alta frequenza.
- **Blocco ZARC** *(R_mid, Q_mid, β_mid)* — ramo R‖CPE generalizzato che cattura
  SEI + capacità di trasferimento di carica / doppio strato.
- **Warburg generalizzato** *(A_w, β_w)* — diffusione di Li-ion nell'elettrodo.

### Cella e strumentazione
- **Cella sotto test:** pouch LiCoO₂ / grafite, **10 Ah** nominali, 2.75–4.2 V.
- **Strumento:** BioLogic SP-150 + booster VMP3B-100, pilotato da EC-Lab.
- **Controllo termico:** celle Peltier sotto la cella, loop PI su sonda Pt100.

### Protocollo di misura
1. **Carica CC-CV** completa a 1C fino a 4.2 V → 100 % SOC.
2. **Sweep GEIS** con eccitazione sinusoidale di 3 A (0.3C), **100 mHz–10 kHz**.
3. Scarica del **12.5 % SOC** a 1C, riposo 1 h, GEIS di nuovo. Si scende fino a 0 %.
4. Si esegue il ciclo **inverso** (in carica).
5. Si ripete a **20, 25, 30, 35, 40, 45 °C**.

### Cosa trova il gruppo del Politecnico
- **R_s e R_mid mostrano isteresi.** I valori in scarica sono sistematicamente più
  alti di quelli in carica; il gap si riduce al crescere della temperatura.
- **A_w** mostra isteresi sopra il 50 % SOC; l'effetto si attenua con la temperatura.
- **Q_mid** mostra isteresi solo sotto il 40 % SOC.
- **β_mid e β_w** — nessuna isteresi chiara.

### Il nostro dataset vs le misure del paper

| Dimensione | Paper (fit eq. 1) | Dataset di progetto |
|---|---|---|
| Livelli di Aging | singolo stato | **5** (0 → 4) |
| Temperature | 6 × passi 5 °C (20 → 45 °C) | **8** (20, 22.5, 25, 27.5, 30, 35, 40, 47.5 °C) |
| Granularità SOC | 9 × passi 12.5 % (0 → 100 %) | **5** (≈ 0, 25, 50, 75, 100 %) |
| Curve di Nyquist totali | ≈ 54 per direzione | **200** (singola direzione) |
`,
    data: `
## Dataset

### Sorgente

- **File grezzo:** \`data/raw/GEIS.mat\` — struttura MATLAB prodotta dal
  potenziostato SP-150 + booster VMP3B-100 controllato da EC-Lab.
- **File processato:** \`data/processed/batteries_cleaned_dataset.csv\`,
  generato con \`python -m scripts.prepare_data\`.

### Schema

| Colonna | Tipo | Descrizione |
|---|---|---|
| \`Aging\` | int | 0–4 (0 = fresh, 4 = invecchiata) |
| \`Temperature\` | float | °C — uno degli 8 valori (20 → 47.5) |
| \`SOC\` | int | Indice di carica 0–4 (≈ 0%, 25%, 50%, 75%, 100%) |
| \`Frequency\` | float | Hz — 49 valori log-spaziati da ~0.1 Hz a 10 kHz |
| \`Z_real\` | float | Re(Z) in **mΩ** |
| \`Z_imag\` | float | Im(Z) in **mΩ** |

### Forma

- **5 aging × 8 temperature × 5 SOC = 200 curve di Nyquist**
- ~9 800 righe in totale (una per campione di frequenza)

### Feature engineering

Due insiemi di feature, definiti in \`src/data/features.py\`:

**Regressione (Leave-One-Out, Leave-One-Aging-Out)** — 9 feature:
\`Aging\`, \`Temperature\`, \`SOC\`, \`Frequency\`, \`log_Freq\`,
\`inv_Temp = 1/(T+273.15)\` (termine Arrhenius) e tre interazioni
\`Aging×Temp\`, \`SOC×Temp\`, \`SOC×Aging\`.

**Classificazione (Young-Old)** — 10 feature, dove l'impedenza diventa
**input** (l'Aging è escluso perché definisce il target):
\`Temperature\`, \`Frequency\`, \`log_Freq\`, \`Z_real\`, \`Z_imag\`,
\`Z_magnitude\`, \`Z_phase\`, \`inv_Temp\`, \`Z_real×Temp\`, \`sqrt_Freq\`.
`,
    tasks: `
## I tre task predittivi

### Leave-One-Out — Ricostruzione di una curva

Date le 40 possibili coppie \`(Aging, Temperatura)\`, **una** viene esclusa.
Il modello viene addestrato sulle 39 rimanenti e deve ricostruire le 5 curve di
Nyquist (una per SOC) della coppia mancante.

- **Input:** \`Aging\`, \`Temperatura\`, \`SOC\`, \`Frequenza\` (+ feature derivate).
- **Target:** \`(Z_real, Z_imag)\` — regressione multi-output.
- **Modelli baseline:** Ridge, Random Forest, Gradient Boosting, KNN, Bagging(Ridge).
- **Metrica di selezione:** MSE (più basso → meglio).

### Young-Old — Classificazione

Scegli un Aging e un SOC: le **8 curve di Nyquist** della coppia
\`(Aging, SOC)\` (una per temperatura, ~49 frequenze ciascuna,
~392 righe in totale) escono dal training. Il modello viene addestrato
sulle restanti ~9 400 righe e deve classificare quelle 8 curve come
**Young** (Aging 0–2) o **Old** (Aging 3–4) leggendo solo la loro *forma*.

- **Input UI:** Aging (0–4) e SOC (0–4).
- **Test set:** le 8 curve di Nyquist della coppia (Aging, SOC) — una
  per temperatura, ognuna mostrata come grafico separato.
- **Train set:** tutte le altre coppie (24 combinazioni Aging×SOC al netto
  della esclusa).
- **Feature:** Re(Z), Im(Z), |Z|, fase, Temperatura, Frequenza e derivate.
  L'**Aging non è una feature** — è il target, quindi è escluso dagli input.
- **Target:** \`Age_class\` binario (Young = 0, Old = 1).
- **Modelli baseline:** Logistic Regression, Random Forest, Gradient Boosting,
  Extra Trees, KNN, SVM (RBF).
- **Metrica di selezione:** Accuracy sulle 8 curve della coppia esclusa.

> **Caveat sulle metriche.** Tutte le 8 curve in test appartengono alla
> stessa classe (è una singola coppia Aging–SOC), quindi AUC-ROC non è
> definita e la confusion matrix collassa a una sola riga. Riportiamo
> Accuracy globale e per-temperatura — è ciò che misura davvero la capacità
> di generalizzare a una coppia mai vista.

### Leave-One-Aging-Out — Aging Interpolation

Un **intero livello di invecchiamento** viene escluso (default: **Aging 2**).
Il modello impara dagli altri quattro livelli e ricostruisce tutte le 40 curve di
Nyquist dell'aging mancante.

- **Input / target:** stessi del Leave-One-Out.
- **Perché è più difficile:** nessuna osservazione diretta dell'aging escluso è mai vista in addestramento.
- **Modelli baseline:** stessi del Leave-One-Out.
- **Metrica di selezione:** R² (più alto → meglio).

> **Caveat sui tree-based.** Random Forest e Gradient Boosting non possono
> estrapolare oltre il supporto di training; KNN e Ridge possono interpolare in
> modo più fluido.
`,
    arch: `
## Layout del progetto

\`\`\`
TestMLOps_Progetto/
├── app.py                     # Backend FastAPI (esposto al frontend React)
├── frontend/                  # Frontend React (Vite + TypeScript)
│   ├── src/
│   │   ├── pages/             # Home, Task1, Task2, Task3, Documentation
│   │   ├── components/        # Layout, Plot, MetricGrid, …
│   │   ├── api.ts             # Client Axios per /api
│   │   └── types.ts
│   └── vite.config.ts
├── src/                       # Package ML (consumato da app.py)
│   ├── config.py              # Caricamento config YAML
│   ├── logger.py
│   ├── constants.py
│   ├── tracking.py            # Helper MLflow (no-op se assente)
│   ├── data/{loader,preprocessing,features}.py
│   ├── models/{registry,tuning,task1_loo,task2_classification,task3_aging}.py
│   ├── evaluation/metrics.py
│   ├── visualization/plots.py
│   └── monitoring/            # Drift detection (KS + PSI) + prediction log
├── scripts/                   # CLI: prepare_data, train_all, tune, monitor
├── tests/                     # Suite pytest
├── notebooks/                 # Notebook esplorativi originali
├── config/config.yaml         # Configurazione centralizzata
├── models/                    # Benchmark persistiti / dump joblib / tuning JSON
├── mlruns/                    # Store MLflow su filesystem
├── docs/                      # Documentazione + report LaTeX (sources/)
├── Dockerfile + docker-compose.yml
├── Makefile                   # make test / make ci / make clean-cov
├── requirements.txt           # solo runtime
├── requirements-dev.txt       # runtime + pytest, ruff, bandit, pip-audit
└── pyproject.toml
\`\`\`

## Design a livelli

1. **Configurazione** — singolo \`config.yaml\` + \`load_config()\` (LRU cached).
2. **Data layer** — preprocessing (\`mat → csv\`) e \`load_dataset()\` cached.
3. **Feature layer** — funzioni pure in \`features.py\`.
4. **Model layer** — un file per task che restituisce un dataclass tipizzato.
5. **Evaluation layer** — \`regression_metrics()\` / \`classification_metrics()\`.
6. **Visualization layer** — figure Plotly in Python e renderer React equivalenti.
7. **API layer** — \`app.py\` espone \`src/\` via HTTP con CORS env-driven.
8. **Frontend** — React + Vite + TypeScript che consuma \`/api\`.
9. **Monitoring layer** — KS + PSI drift detection esposti via REST.

Ogni livello dipende solo da quelli sotto.
`,
    run: `
## Esecuzione locale

### Modalità sviluppo a due processi

\`\`\`bash
# 1) Backend (FastAPI su :8000)
uvicorn app:app --reload --port 8000

# 2) Frontend (Vite su :5173)
cd frontend && npm install && npm run dev
\`\`\`

Il dev server di Vite proxa \`/api/*\` su \`http://127.0.0.1:8000\`, quindi apri
<http://localhost:5173>.

### Modalità single-binary (production-like)

\`\`\`bash
cd frontend && npm run build       # → frontend/dist/
uvicorn app:app --host 0.0.0.0 --port 8000
\`\`\`

Quando \`frontend/dist/index.html\` esiste, FastAPI lo monta su \`/\` e
backend + UI vivono entrambi su <http://localhost:8000>.

### Preprocessing del dataset

\`\`\`bash
python -m scripts.prepare_data
# → data/processed/batteries_cleaned_dataset.csv
\`\`\`

### Training di tutti i baseline

\`\`\`bash
python -m scripts.train_all                 # task 1, 2, 3
python -m scripts.train_all --tasks 1 3     # subset
\`\`\`

### Hyperparameter tuning

\`\`\`bash
python -m scripts.tune                      # GridSearchCV per i tre task
python -m scripts.tune --tasks 1
\`\`\`

### Test e lint

\`\`\`bash
make test                                                # 81 test + gate cov ≥ 90 %
make test-fast                                           # subset (-m "not slow")
make ci                                                  # ruff + pytest + bandit + pip-audit
pytest                                                   # raw pytest (no coverage)
pytest --cov=src --cov=app --cov-report=term-missing     # con coverage
ruff check src tests scripts app.py
\`\`\`

### Docker

\`\`\`bash
docker compose up --build
# → http://localhost:8000
\`\`\`
`,
    api: `
## API HTTP (FastAPI)

L'OpenAPI / Swagger UI è disponibile su <http://localhost:8000/docs>.

### Endpoint

| Metodo | Path | Descrizione |
|---|---|---|
| GET | \`/health\` | Liveness probe |
| GET | \`/api/project\` | Nome, versione, autori, info paper, costanti UI |
| GET | \`/api/paper\` | PDF del paper di riferimento |
| GET | \`/api/dataset/summary\` | Righe, combinazioni, range frequenza, livelli |
| GET | \`/api/dataset/options\` | Valori validi di aging / temperatura / SOC |
| GET | \`/api/dataset/curves?aging=N\` | Scatter di Nyquist all'aging dato |
| GET | \`/api/dataset/agg-by-temp\` | Re(Z) medio per (Aging, Temperatura) |
| GET | \`/api/dataset/aging-evolution?soc=&excluded_aging=\` | EDA Task 3 |
| POST | \`/api/task1/run\` | Body: \`{aging, temperature}\` → risultato LOO |
| POST | \`/api/task2/run\` | Body: \`{aging, soc}\` → classificazione della coppia esclusa |
| POST | \`/api/task3/run\` | Body: \`{excluded_aging}\` → aging interpolation |
| GET | \`/api/benchmarks/{task}\` | \`benchmark.json\` persistito |
| POST | \`/api/predictions/log\` | Append al prediction log JSON-Lines |
| GET | \`/api/predictions/recent?limit=N\` | Ultimi N record del log |
| GET | \`/api/monitoring/drift\` | Report KS + PSI di drift |

## API Python (\`src/\`)

\`\`\`python
from src.data.loader import load_dataset, dataset_summary
from src.data.features import build_regression_features, build_classification_features
from src.models.task1_loo import run_leave_one_out
from src.models.task2_classification import run_classification
from src.models.task3_aging import run_aging_interpolation
from src.evaluation.metrics import regression_metrics, classification_metrics
from src.models.registry import regression_models, classification_models
from src.models.tuning import tune_task1, tune_task2, tune_task3, load_best_params
from src.tracking import track_pipeline, log_model_run
from src.monitoring import detect_dataset_drift, log_prediction
\`\`\`

Il layer HTTP (\`app.py\`) è un wrapper sottile: ogni route mappa una di queste
funzioni e converte il dataclass restituito in un payload JSON.
`,
    mlops: `
## Pratiche MLOps

| Area | Implementazione |
|---|---|
| **Riproducibilità** | \`requirements.txt\` (runtime) + \`requirements-dev.txt\` (test/lint/security) versionati, \`random_state=42\` ovunque, split deterministici, dataset SHA-256 loggato per ogni run |
| **Configurazione** | \`config/config.yaml\` + \`load_config()\` con LRU cache — task, griglie tuning e tracking dichiarativi |
| **Data versioning** | \`.mat\` raw in \`data/raw\`, CSV processato rigenerabile, SHA tracciato nei tag MLflow |
| **Separazione di concerns** | Package \`data/\`, \`models/\`, \`evaluation/\`, \`visualization/\`, \`tracking/\`, \`monitoring/\` |
| **Logging** | \`src/logger.py\` → \`logs/pipeline.log\` con rotazione (10 MB × 5) |
| **Experiment tracking** | MLflow file backend (\`mlruns/\`) con parent/child run, dataset hash + git SHA |
| **Hyperparameter tuning** | \`GridSearchCV\` con splitter coerenti col task, best params persistiti |
| **Artifact persistence** | \`models/taskN/benchmark.json\`, joblib dump, artefatti MLflow |
| **Testing** | Suite pytest (81 test, di cui 11 \`slow\`, ≥90 % coverage gate; 95 % attuale) |
| **CI/CD** | \`.github/workflows/ci.yml\` con lint, test+coverage, bandit, pip-audit, docker build |
| **Containerizzazione** | \`Dockerfile\` multistage + \`docker-compose.yml\` |
| **Monitoring** | Drift detection KS + PSI in \`src/monitoring/\` con CLI ed endpoint REST |
| **Frontend / backend split** | \`app.py\` (FastAPI) + \`frontend/\` (React + Vite) — UI e API deployabili indipendentemente |
| **Notebook ↔ produzione** | Stessi moduli di feature e modello — notebook e app non divergono |
`,
    tracking: `
## Experiment tracking (MLflow)

Ogni chiamata a \`run_leave_one_out\`, \`run_classification\`,
\`run_aging_interpolation\` e ai tre driver di tuning apre un **MLflow parent
run** sotto un file backend locale a \`mlruns/\`. Per ogni modello addestrato si
crea un child run con i suoi parametri, metriche e durata.

| Livello | Payload |
|---|---|
| **Esperimento** | Uno per task, es. \`battery-geis-task1\` |
| **Tag del parent run** | \`task\`, \`git_sha\`, \`data_sha256\` |
| **Parametri parent run** | \`(aging, temperature)\` / \`soc\` / \`excluded_aging\`, fold CV, ecc. |
| **Child run (per modello)** | Iperparametri · metriche + \`duration_s\` |

Per sfogliare i run:

\`\`\`bash
python -m mlflow ui --backend-store-uri mlruns --port 5000
\`\`\`

Imposta \`tracking.enabled: false\` in \`config/config.yaml\` per disabilitare tutto.

## Hyperparameter tuning

Gli iperparametri di default in \`src/models/registry.py\` sono conservativi;
\`src/models/tuning.py\` cerca le griglie dichiarate in \`config/config.yaml\`
con uno **splitter CV coerente col task**:

| Task | Splitter | Stimatori tunati |
|---|---|---|
| Leave-One-Out | \`LeaveOneGroupOut\` su (Aging, Temperatura) | RF, GB, KNN |
| Young-Old | hold-out della coppia (Aging, SOC) scelta dall'utente | RF, GB, SVM (RBF) |
| Leave-One-Aging-Out | \`LeaveOneGroupOut\` su Aging | RF, GB, KNN |

\`\`\`bash
python -m scripts.tune                  # tutti i task
python -m scripts.tune --tasks 1
\`\`\`

Per usare i best params persistiti, passa \`use_tuned=True\` alla factory del
registry (\`regression_models(use_tuned=True, task="task1")\`).

## Monitoring & drift detection

\`src/monitoring/drift.py\` espone due test complementari:

- **Kolmogorov–Smirnov (KS)** per cambiamenti di forma della distribuzione;
- **Population Stability Index (PSI)** per spostamenti di massa fra bin
  (soglie convenzionali: < 0.10 stabile, 0.10–0.25 drift minore, ≥ 0.25 maggiore).

Una feature è dichiarata drifted se almeno uno dei due test fire. Il verdetto a
livello dataset scatta se almeno il 25 % delle feature drifta.

\`\`\`bash
python -m scripts.monitor --current data/processed/recent_campaign.csv
\`\`\`

Il \`prediction_log.py\` aggiunge ogni predizione a \`logs/predictions.jsonl\`
(audit trail + sorgente per il drift su traffico live).
`,
    refs: `
## Riferimenti

**Paper di riferimento**

- S. Barcellona, L. Codecasa, S. Colnago, D. D'Amore (Politecnico di Milano).
  *Hysteresis Phenomenon in the Electric Parameters of Lithium-Ion Batteries
  under Temperature Effects.* **IEEE THERMINIC 2025**,
  DOI \`10.1109/THERMINIC65879.2025.11216945\`.

**Letteratura di supporto**

- Barcellona & Piegari. *Lithium Ion Battery Models and Parameter
  Identification Techniques.* Energies 10 (2017).
- Wang, Li, Chen. *Experimental study of fractional-order models for
  lithium-ion battery…* Applied Energy 278 (2020).
- Barcellona, Colnago, Piegari. *Aging Effect on Lithium-Ion Battery
  Resistance Hysteresis.* IEEE TIA (2023).
- F. J. Massey. *The Kolmogorov-Smirnov Test for Goodness of Fit.* JASA (1951).
- N. Siddiqi. *Intelligent Credit Scoring* (Wiley, 2017) — soglie PSI 0.10/0.25.
- B. Efron. *Bootstrap Methods.* Annals of Statistics (1979).
- Quiñonero-Candela et al. *Dataset Shift in Machine Learning* (MIT Press, 2008).

### Tooling

- **Runtime:** Python 3.11–3.12 · scikit-learn · pandas · numpy · scipy
- **API:** FastAPI · uvicorn · Pydantic
- **Frontend:** React · Vite · TypeScript · Plotly.js
- **MLOps:** MLflow · GridSearchCV · pytest · ruff · bandit · pip-audit
`,
  },
  en: {
    overview: `
## Context

This project is the **SUPSI Semester Project** by Davide Corso and Marco Soldani,
carried out in collaboration with **Politecnico di Milano**.
The starting point is the experimental methodology of the paper

> *"Hysteresis Phenomenon in the Electric Parameters of Lithium-Ion Batteries under
> Temperature Effects"* — S. Barcellona, L. Codecasa, S. Colnago, D. D'Amore (**THERMINIC 2025**)

around which we built a **production-grade MLOps pipeline**: FastAPI backend,
React frontend, reproducible training, experiment tracking, hyperparameter
tuning, test suite and CI.

The underlying physics is Galvanostatic Electrochemical Impedance Spectroscopy
(GEIS) on a **10 Ah LiCoO₂ pouch cell**, measured across **8 temperatures**,
**5 SOC levels** and **5 aging states**, giving **200 Nyquist curves**.

## Goals

1. Reconstruct the Nyquist plot of one missing **(Aging, Temperature)** combination from the other 39 (Leave-One-Out).
2. Classify as **Young vs Old** the 8 Nyquist curves (one per temperature) of an (Aging, SOC) pair the model **never saw during training** (Young-Old).
3. Predict an **entire aging level** (default: Aging 2) from the other four (Leave-One-Aging-Out).

## How the pieces fit together

\`\`\`
   config.yaml ── load_config()
        │
        ▼
 data/raw/GEIS.mat ── prepare_data ──▶ data/processed/batteries_cleaned_dataset.csv
                                          │
                                          ▼
                                   load_dataset() ──▶ features.py ──▶ models/task{1,2,3}.py
                                                                          │
                                     tuning.py ─────────────────▶  models/tuning/*.json
                                                                          │
                                                                          ▼
                                                                   tracking.py ──▶ mlruns/
                                                                          │
                                                                          ▼
                                                          app.py (FastAPI)  ◀──▶  frontend/ (React)
\`\`\`
`,
    paper: `
## The paper in plain English

> *S. Barcellona, L. Codecasa, S. Colnago, D. D'Amore — Politecnico di Milano,
> **THERMINIC 2025**, DOI \`10.1109/THERMINIC65879.2025.11216945\`.*

### The question
Do the **RC parameters** of a Li-ion equivalent-circuit model behave the
same when the cell is charging vs discharging? And how does that
asymmetry change with temperature?

### The equivalent-circuit model
The cell's impedance at each (SOC, T) point is fitted to a three-block model:

\`\`\`
Z_bat(ω) = R_s + R_mid / [1 + (jω)^β_mid · R_mid · Q_mid] + A_w / (jω)^β_w
\`\`\`

- **R_s** — high-frequency ohmic resistance.
- **ZARC block** *(R_mid, Q_mid, β_mid)* — generalised R‖CPE branch capturing
  the SEI layer and charge-transfer / double-layer capacitance.
- **Generalised Warburg** *(A_w, β_w)* — Li-ion diffusion inside the electrode.

### Cell & equipment
- **Device under test:** LiCoO₂ / graphite pouch cell, **10 Ah** nominal, 2.75–4.2 V.
- **Instrument:** BioLogic SP-150 + VMP3B-100 booster, driven by EC-Lab.
- **Thermal control:** Peltier cells under the cell, PI loop on a Pt100 probe.

### Measurement protocol
1. Full **CC-CV charge** at 1C up to 4.2 V → 100 % SOC.
2. **GEIS sweep** with 3 A (0.3C) sinusoidal excitation, **100 mHz–10 kHz**.
3. Discharge **12.5 % SOC** at 1C, rest 1 h, run GEIS again. Repeat down to 0 %.
4. Run the *reverse* cycle (charging direction).
5. Repeat at **20, 25, 30, 35, 40, 45 °C**.

### What the Politecnico group find
- **R_s and R_mid exhibit hysteresis.** Discharge values are systematically
  higher than charge ones; the gap shrinks as temperature rises.
- **A_w** shows hysteresis above 50 % SOC; effect weakens with temperature.
- **Q_mid** shows hysteresis only below 40 % SOC.
- **β_mid and β_w** — no clear hysteresis.

### Our dataset vs the paper's measurements

| Dimension | Paper (eq. 1 fits) | Project dataset |
|---|---|---|
| Aging levels | single state | **5** (0 → 4) |
| Temperatures | 6 × 5 °C steps (20 → 45 °C) | **8** (20, 22.5, 25, 27.5, 30, 35, 40, 47.5 °C) |
| SOC granularity | 9 × 12.5 % steps (0 → 100 %) | **5** (≈ 0, 25, 50, 75, 100 %) |
| Total Nyquist curves | ≈ 54 per current direction | **200** (single direction) |
`,
    data: `
## Dataset

### Source

- **Raw file:** \`data/raw/GEIS.mat\` — MATLAB struct produced by an SP-150
  potentiostat + VMP3B-100 booster, controlled via EC-Lab.
- **Processed file:** \`data/processed/batteries_cleaned_dataset.csv\`,
  generated with \`python -m scripts.prepare_data\`.

### Schema

| Column | Type | Description |
|---|---|---|
| \`Aging\` | int | 0–4 (0 = fresh, 4 = aged) |
| \`Temperature\` | float | °C — one of 8 values (20 → 47.5) |
| \`SOC\` | int | State of charge index 0–4 (≈ 0%, 25%, 50%, 75%, 100%) |
| \`Frequency\` | float | Hz — 49 log-spaced values from ~0.1 Hz to 10 kHz |
| \`Z_real\` | float | Re(Z) in **mΩ** |
| \`Z_imag\` | float | Im(Z) in **mΩ** |

### Shape

- **5 agings × 8 temperatures × 5 SOC = 200 Nyquist curves**
- ~9 800 rows in total (one per frequency sample)

### Feature engineering

Two feature sets, defined in \`src/data/features.py\`:

**Regression (Leave-One-Out, Leave-One-Aging-Out)** — 9 features:
\`Aging\`, \`Temperature\`, \`SOC\`, \`Frequency\`, \`log_Freq\`,
\`inv_Temp = 1/(T+273.15)\` (Arrhenius term), and three interactions
\`Aging×Temp\`, \`SOC×Temp\`, \`SOC×Aging\`.

**Classification (Young-Old)** — 10 features where impedance becomes the
**input** (Aging is excluded because it defines the target):
\`Temperature\`, \`Frequency\`, \`log_Freq\`, \`Z_real\`, \`Z_imag\`,
\`Z_magnitude\`, \`Z_phase\`, \`inv_Temp\`, \`Z_real×Temp\`, \`sqrt_Freq\`.
`,
    tasks: `
## The three prediction tasks

### Leave-One-Out — Curve Reconstruction

Given the 40 possible \`(Aging, Temperature)\` combinations, **one** is held out.
The model is trained on the remaining 39 and must reconstruct the 5 Nyquist
plots (one per SOC) of the missing combination.

- **Input:** \`Aging\`, \`Temperature\`, \`SOC\`, \`Frequency\` (+ derived features).
- **Target:** \`(Z_real, Z_imag)\` — multi-output regression.
- **Baseline models:** Ridge, Random Forest, Gradient Boosting, KNN, Bagging(Ridge).
- **Selection metric:** MSE (lower → better).

### Young-Old — Classification

Pick an Aging and a SOC: the **8 Nyquist curves** of that
\`(Aging, SOC)\` pair (one per temperature, ~49 frequencies each,
~392 rows in total) are removed from training. The model is trained on
the remaining ~9 400 rows and must classify those 8 curves as **Young**
(Aging 0–2) or **Old** (Aging 3–4) from their *shape* alone.

- **UI input:** Aging (0–4) and SOC (0–4).
- **Test set:** the 8 Nyquist curves of the (Aging, SOC) pair — one per
  temperature, each rendered as a separate plot.
- **Train set:** every other pair (24 Aging×SOC combos minus the held-out one).
- **Features:** Re(Z), Im(Z), |Z|, phase, Temperature, Frequency and derived
  terms. **Aging is not a feature** — it is the target, hence excluded from
  the inputs.
- **Target:** binary \`Age_class\` (Young = 0, Old = 1).
- **Baseline models:** Logistic Regression, Random Forest, Gradient Boosting,
  Extra Trees, KNN, SVM (RBF).
- **Selection metric:** Accuracy on the 8 held-out curves.

> **Metric caveat.** All 8 held-out curves belong to the same class (it is
> a single Aging–SOC pair), so AUC-ROC is undefined and the confusion
> matrix collapses to one row. We report global and per-temperature
> Accuracy — that is what actually measures generalisation to an unseen pair.

### Leave-One-Aging-Out — Aging Interpolation

An **entire aging level** is held out (default: **Aging 2**).
The model learns from the other four levels and reconstructs all 40
Nyquist plots of the missing aging.

- **Input / target:** same as Leave-One-Out.
- **Why harder:** no direct observation of the held-out aging is ever seen during training.
- **Baseline models:** same as Leave-One-Out.
- **Selection metric:** R² (higher → better).

> **Tree-based caveat.** Random Forest and Gradient Boosting cannot extrapolate
> beyond training support; KNN and Ridge can interpolate more smoothly.
`,
    arch: `
## Project layout

\`\`\`
TestMLOps_Progetto/
├── app.py                     # FastAPI backend (React-facing)
├── frontend/                  # React (Vite + TypeScript) frontend
│   ├── src/
│   │   ├── pages/             # Home, Task1, Task2, Task3, Documentation
│   │   ├── components/        # Layout, Plot, MetricGrid, …
│   │   ├── api.ts             # Axios client for /api
│   │   └── types.ts
│   └── vite.config.ts
├── src/                       # Core ML package (consumed by app.py)
│   ├── config.py              # YAML config loader with path resolution
│   ├── logger.py
│   ├── constants.py
│   ├── tracking.py            # MLflow helpers (graceful no-op if absent)
│   ├── data/{loader,preprocessing,features}.py
│   ├── models/{registry,tuning,task1_loo,task2_classification,task3_aging}.py
│   ├── evaluation/metrics.py
│   ├── visualization/plots.py
│   └── monitoring/            # Drift detection (KS + PSI) + prediction log
├── scripts/                   # CLI: prepare_data, train_all, tune, monitor
├── tests/                     # pytest suite
├── notebooks/                 # Original experiment notebooks
├── config/config.yaml         # Central configuration
├── models/                    # Persisted benchmarks / joblib dumps / tuning JSON
├── mlruns/                    # MLflow file-backed store
├── docs/                      # Markdown docs + LaTeX semester report (sources/)
├── Dockerfile + docker-compose.yml
├── Makefile                   # make test / make ci / make clean-cov
├── requirements.txt           # runtime only
├── requirements-dev.txt       # runtime + pytest, ruff, bandit, pip-audit
└── pyproject.toml
\`\`\`

## Layered design

1. **Configuration** — single \`config.yaml\` + \`load_config()\` (LRU cached).
2. **Data layer** — preprocessing (\`mat → csv\`) and a cached \`load_dataset()\`.
3. **Feature layer** — pure functions in \`features.py\`.
4. **Model layer** — one file per task returning a typed dataclass.
5. **Evaluation layer** — \`regression_metrics()\` / \`classification_metrics()\`.
6. **Visualization layer** — Plotly figures in Python and equivalent React renderers.
7. **API layer** — \`app.py\` exposes \`src/\` over HTTP with env-driven CORS.
8. **Frontend** — React + Vite + TypeScript consuming \`/api\`.
9. **Monitoring layer** — KS + PSI drift detection exposed via REST.

Each layer only depends on the ones below.
`,
    run: `
## Running locally

### Two-process dev mode

\`\`\`bash
# 1) Backend (FastAPI on :8000)
uvicorn app:app --reload --port 8000

# 2) Frontend (Vite on :5173)
cd frontend && npm install && npm run dev
\`\`\`

The Vite dev server proxies \`/api/*\` to \`http://127.0.0.1:8000\`, so just
open <http://localhost:5173>.

### Single-binary mode (production-like)

\`\`\`bash
cd frontend && npm run build       # → frontend/dist/
uvicorn app:app --host 0.0.0.0 --port 8000
\`\`\`

When \`frontend/dist/index.html\` exists, FastAPI mounts it at \`/\`, so both
the UI and the API are reachable at <http://localhost:8000>.

### One-off data preparation

\`\`\`bash
python -m scripts.prepare_data
# → data/processed/batteries_cleaned_dataset.csv
\`\`\`

### Train all baselines

\`\`\`bash
python -m scripts.train_all                 # tasks 1, 2, 3
python -m scripts.train_all --tasks 1 3     # subset
\`\`\`

### Hyperparameter tuning

\`\`\`bash
python -m scripts.tune                      # GridSearchCV for the three tasks
python -m scripts.tune --tasks 1
\`\`\`

### Tests / lint

\`\`\`bash
make test                                                # 81 tests + ≥ 90 % coverage gate
make test-fast                                           # subset (-m "not slow")
make ci                                                  # ruff + pytest + bandit + pip-audit
pytest                                                   # raw pytest (no coverage)
pytest --cov=src --cov=app --cov-report=term-missing     # with coverage
ruff check src tests scripts app.py
\`\`\`

### Docker

\`\`\`bash
docker compose up --build
# → http://localhost:8000
\`\`\`
`,
    api: `
## HTTP API (FastAPI)

The OpenAPI / Swagger UI is bundled at <http://localhost:8000/docs>.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | \`/health\` | Liveness probe |
| GET | \`/api/project\` | Name, version, authors, paper meta, UI constants |
| GET | \`/api/paper\` | Bundled paper PDF |
| GET | \`/api/dataset/summary\` | Rows, combos, frequency range, levels |
| GET | \`/api/dataset/options\` | Valid aging / temperature / SOC values |
| GET | \`/api/dataset/curves?aging=N\` | Nyquist scatter at the given aging |
| GET | \`/api/dataset/agg-by-temp\` | Mean Re(Z) per (Aging, Temperature) |
| GET | \`/api/dataset/aging-evolution?soc=&excluded_aging=\` | Task 3 EDA |
| POST | \`/api/task1/run\` | Body: \`{aging, temperature}\` → LOO result |
| POST | \`/api/task2/run\` | Body: \`{aging, soc}\` → classification of the held-out pair |
| POST | \`/api/task3/run\` | Body: \`{excluded_aging}\` → aging interpolation |
| GET | \`/api/benchmarks/{task}\` | Persisted \`benchmark.json\` for the task |
| POST | \`/api/predictions/log\` | Append to JSON-Lines prediction log |
| GET | \`/api/predictions/recent?limit=N\` | Last N records of the log |
| GET | \`/api/monitoring/drift\` | KS + PSI drift report |

## Python API (\`src/\`)

\`\`\`python
from src.data.loader import load_dataset, dataset_summary
from src.data.features import build_regression_features, build_classification_features
from src.models.task1_loo import run_leave_one_out
from src.models.task2_classification import run_classification
from src.models.task3_aging import run_aging_interpolation
from src.evaluation.metrics import regression_metrics, classification_metrics
from src.models.registry import regression_models, classification_models
from src.models.tuning import tune_task1, tune_task2, tune_task3, load_best_params
from src.tracking import track_pipeline, log_model_run
from src.monitoring import detect_dataset_drift, log_prediction
\`\`\`

The HTTP layer (\`app.py\`) is a thin wrapper: every route maps to one of these
functions and converts the dataclass result to a JSON-friendly payload.
`,
    mlops: `
## MLOps practices

| Concern | Implementation |
|---|---|
| **Reproducibility** | Versioned \`requirements.txt\` (runtime) + \`requirements-dev.txt\` (test / lint / security), \`random_state=42\` everywhere, deterministic splits, dataset SHA-256 logged per run |
| **Configuration** | Single \`config/config.yaml\` + \`load_config()\` with LRU cache — tasks, tuning grids and tracking backend all declarative |
| **Data versioning** | Raw \`.mat\` under \`data/raw\`, processed CSV regenerable, SHA tracked in MLflow tags |
| **Separation of concerns** | \`data/\`, \`models/\`, \`evaluation/\`, \`visualization/\`, \`tracking/\`, \`monitoring/\` packages |
| **Logging** | \`src/logger.py\` → \`logs/pipeline.log\` with rotation (10 MB × 5) |
| **Experiment tracking** | MLflow file backend (\`mlruns/\`), parent/child runs, dataset hash + git SHA |
| **Hyperparameter tuning** | \`GridSearchCV\` with task-coherent splitters, best params persisted |
| **Artifact persistence** | \`models/taskN/benchmark.json\`, joblib dumps, MLflow artefacts |
| **Testing** | pytest suite (81 tests, of which 11 \`slow\`, ≥90 % coverage gate; 95 % current) |
| **CI/CD** | \`.github/workflows/ci.yml\` with lint, test+coverage, bandit, pip-audit, docker build |
| **Containerisation** | Multi-stage \`Dockerfile\` + \`docker-compose.yml\` |
| **Monitoring** | KS + PSI drift detection in \`src/monitoring/\` with CLI and REST endpoints |
| **Frontend / backend split** | \`app.py\` (FastAPI) + \`frontend/\` (React + Vite) — UI and API independently deployable |
| **Notebook ↔ production** | Shared feature & model modules — notebooks and app never diverge |
`,
    tracking: `
## Experiment tracking (MLflow)

Every call to \`run_leave_one_out\`, \`run_classification\`,
\`run_aging_interpolation\` and the three tuning drivers opens an **MLflow
parent run** under a local file-backed store at \`mlruns/\`. A nested child run
is created per trained model with its parameters, metrics and duration.

| Level | Payload |
|---|---|
| **Experiment** | One per task, e.g. \`battery-geis-task1\` |
| **Parent run tags** | \`task\`, \`git_sha\`, \`data_sha256\` |
| **Parent run params** | \`(aging, temperature)\` / \`soc\` / \`excluded_aging\`, CV folds, etc. |
| **Child run (per model)** | Estimator hyperparameters · metrics + \`duration_s\` |

Browse the runs:

\`\`\`bash
python -m mlflow ui --backend-store-uri mlruns --port 5000
\`\`\`

Set \`tracking.enabled: false\` in \`config/config.yaml\` to disable everything.

## Hyperparameter tuning

Baseline hyperparameters in \`src/models/registry.py\` are conservative;
\`src/models/tuning.py\` searches the grids declared in \`config/config.yaml\`
with a **task-coherent CV splitter**:

| Task | Splitter | Tuned estimators |
|---|---|---|
| Leave-One-Out | \`LeaveOneGroupOut\` on (Aging, Temperature) | RF, GB, KNN |
| Young-Old | hold-out of the user-picked (Aging, SOC) pair | RF, GB, SVM (RBF) |
| Leave-One-Aging-Out | \`LeaveOneGroupOut\` on Aging | RF, GB, KNN |

\`\`\`bash
python -m scripts.tune                  # all three tasks
python -m scripts.tune --tasks 1
\`\`\`

To pick up the persisted best params, pass \`use_tuned=True\` to the registry
factory (\`regression_models(use_tuned=True, task="task1")\`).

## Monitoring & drift detection

\`src/monitoring/drift.py\` exposes two complementary tests:

- **Kolmogorov–Smirnov (KS)** for distribution shape changes;
- **Population Stability Index (PSI)** for bin mass shifts (industry
  thresholds: < 0.10 stable, 0.10–0.25 minor drift, ≥ 0.25 major).

A feature is flagged as drifted when at least one of the two tests fires. The
dataset-level verdict triggers when at least 25 % of features drift.

\`\`\`bash
python -m scripts.monitor --current data/processed/recent_campaign.csv
\`\`\`

The \`prediction_log.py\` appends every prediction to
\`logs/predictions.jsonl\` (audit trail + source for live-traffic drift).
`,
    refs: `
## References

**Primary paper**

- S. Barcellona, L. Codecasa, S. Colnago, D. D'Amore (Politecnico di Milano).
  *Hysteresis Phenomenon in the Electric Parameters of Lithium-Ion Batteries
  under Temperature Effects.* **IEEE THERMINIC 2025**,
  DOI \`10.1109/THERMINIC65879.2025.11216945\`.

**Supporting literature**

- Barcellona & Piegari. *Lithium Ion Battery Models and Parameter
  Identification Techniques.* Energies 10 (2017).
- Wang, Li, Chen. *Experimental study of fractional-order models for
  lithium-ion battery…* Applied Energy 278 (2020).
- Barcellona, Colnago, Piegari. *Aging Effect on Lithium-Ion Battery
  Resistance Hysteresis.* IEEE TIA (2023).
- F. J. Massey. *The Kolmogorov-Smirnov Test for Goodness of Fit.* JASA (1951).
- N. Siddiqi. *Intelligent Credit Scoring* (Wiley, 2017) — PSI thresholds 0.10/0.25.
- B. Efron. *Bootstrap Methods.* Annals of Statistics (1979).
- Quiñonero-Candela et al. *Dataset Shift in Machine Learning* (MIT Press, 2008).

### Tooling

- **Runtime:** Python 3.11–3.12 · scikit-learn · pandas · numpy · scipy
- **API:** FastAPI · uvicorn · Pydantic
- **Frontend:** React · Vite · TypeScript · Plotly.js
- **MLOps:** MLflow · GridSearchCV · pytest · ruff · bandit · pip-audit
`,
  },
}

export default function Documentation() {
  const t = useT()
  const { lang } = useLang()
  const [tabKey, setTabKey] = useState<TabKey>('overview')

  const tabLabel = (k: TabKey) => t.docs.tabs[k]

  return (
    <>
      <h1 className="page-title">{t.docs.title}</h1>
      <div className="page-caption">{t.docs.caption}</div>

      <div className="tabs">
        {TAB_KEYS.map((k) => (
          <button key={k} className={k === tabKey ? 'active' : ''} onClick={() => setTabKey(k)}>
            {tabLabel(k)}
          </button>
        ))}
      </div>

      <div className="card prose">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {CONTENT[lang][tabKey]}
        </ReactMarkdown>
      </div>
    </>
  )
}
