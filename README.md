# Student Feedback Analysis Platform

Batch-only pipeline for multilingual (Russian/Kazakh) student survey data. Derives three latent quality indicators from nine ordinal items via ordinal CFA, trains independent text classifiers (language, sentiment, detail level), runs late-fusion experiments, and exposes everything through a FastAPI backend and Next.js UI.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11 | [python.org](https://www.python.org/downloads/) |
| uv | ≥ 0.4 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| bun | ≥ 1.0 | `curl -fsSL https://bun.sh/install \| bash` |

> **Note:** npm, yarn, and pnpm are not used in this project.

---

## Setup

### 1. Backend

```bash
cd backend
uv sync
```

This creates `.venv/` and installs all Python dependencies (pandas, scikit-learn, semopy, fastapi, etc.).

### 2. Frontend

```bash
# from repo root
bun install
```

### 3. Dataset

Place the dataset at `mnt/data/dataset.csv`. The expected format is the original 18,476-row survey export with 14 columns (Russian column headers).

---

## Demo — end-to-end run

Run the full pipeline on the dataset (from repo root):

```bash
cd backend
uv run python -m src.cli run-full \
  --data ../mnt/data/dataset.csv \
  --config configs/experiment.yaml \
  --seed 42
```

This executes all stages in order:
1. Data ingestion and snapshot
2. Preprocessing (unicode normalisation, PII redaction, feature extraction)
3. Stratified 80/10/10 split
4. Ordinal CFA psychometrics (3 factors, 9 items)
5. Text classification baselines (TF-IDF + char n-gram × 3 tasks)
6. Late-fusion regression (survey-only vs text-only vs combined)
7. Contradiction monitoring
8. Evaluation and reporting

Artifacts are written to `backend/runs/<run_id>/`.

---

## Starting the services

### Backend API (port 8000)

```bash
cd backend
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (port 3000)

```bash
# from repo root
bun dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend proxies `/api/*` requests to `http://localhost:8000`.

---

## Running tests

```bash
cd backend
uv run pytest ../tests/backend/ -v
```

65 tests covering preprocessing, ingestion, splits, run management, psychometrics, text classifiers, evaluation, contradiction detection, and API endpoints.

---

## Individual CLI commands

```bash
cd backend

# Psychometrics only
uv run python -m src.cli run-psychometrics \
  --data ../mnt/data/dataset.csv --config configs/experiment.yaml

# Train a single text classifier
uv run python -m src.cli train-sentiment \
  --data ../mnt/data/dataset.csv --config configs/experiment.yaml

# Other tasks: train-language, train-length
# Fusion
uv run python -m src.cli run-fusion \
  --run-id <run_id> --config configs/experiment.yaml

# Contradiction monitoring
uv run python -m src.cli run-contradiction --run-id <run_id>

# Evaluation
uv run python -m src.cli evaluate --run-id <run_id>

# Reports
uv run python -m src.cli report --run-id <run_id>
```

---

## Project layout

```
.
├── backend/
│   ├── src/
│   │   ├── api/            FastAPI app (runs, metrics, artifacts routes)
│   │   ├── ingest/         Dataset loading and snapshotting
│   │   ├── preprocessing/  Unicode normalisation, PII redaction, features
│   │   ├── psychometrics/  Ordinal CFA, reliability (α, ω)
│   │   ├── splits/         Stratified 80/10/10 splits
│   │   ├── text_tasks/     TF-IDF and char-ngram classifiers
│   │   ├── fusion/         Late-fusion regression experiments
│   │   ├── contradiction/  Deterministic flag detector
│   │   ├── evaluation/     Classification and regression metrics
│   │   ├── reporting/      Markdown evaluation reports and model cards
│   │   ├── utils/          Logging, reproducibility, run manager
│   │   ├── pipeline.py     Full pipeline orchestrator
│   │   └── cli.py          Click CLI entry point
│   ├── configs/
│   │   ├── factor_structure.yaml
│   │   └── experiment.yaml
│   └── pyproject.toml
├── app/                    Next.js frontend (App Router)
│   ├── components/
│   ├── lib/api.ts
│   └── runs/[runId]/       Per-run detail pages
├── shared/schemas/         JSON schemas for run metadata and artifacts
├── tests/
│   ├── backend/            65 pytest tests
│   └── fixtures/           tiny_dataset.csv (50 rows for tests)
├── docs/                   Data dictionary, dataset datasheet
├── mnt/data/               Dataset (gitignored)
├── TODO.md
├── PROGRESS.md
├── TESTS.md
├── DECISIONS.md
├── RUNBOOK.md
└── CHANGELOG.md
```

---

## Disclaimers

- Results are **aggregated** — not for individual-level decision making.
- No causal claims. Sentiment and factor scores are observational.
- Psychometrics are based on self-reported ordinal survey items.
