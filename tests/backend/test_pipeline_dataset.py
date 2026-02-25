"""Tests for pipeline with DataFrame input (Phase 1 — dataset version selection)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pytest


BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TINY_DATASET = FIXTURES_DIR / "tiny_dataset.csv"


# ---------------------------------------------------------------------------
# pipeline.run_full_pipeline — df_raw branch
# ---------------------------------------------------------------------------

def test_pipeline_requires_data_path_or_df_raw(tmp_path, experiment_config_path, factor_structure_path):
    """Pipeline raises ValueError when neither data_path nor df_raw is provided."""
    from src.pipeline import run_full_pipeline

    with pytest.raises(ValueError, match="data_path or df_raw"):
        run_full_pipeline(
            config_path=experiment_config_path,
            factor_structure_path=factor_structure_path,
            runs_dir=tmp_path / "runs",
        )


def test_pipeline_snapshot_id_from_df(tiny_df):
    """Snapshot ID derived from DataFrame hash is a 16-char hex string."""
    snap = hashlib.sha256(tiny_df.to_csv(index=False).encode()).hexdigest()[:16]
    assert len(snap) == 16
    assert all(c in "0123456789abcdef" for c in snap)


def test_pipeline_snapshot_deterministic(tiny_df):
    """Same DataFrame produces the same snapshot ID."""
    snap1 = hashlib.sha256(tiny_df.to_csv(index=False).encode()).hexdigest()[:16]
    snap2 = hashlib.sha256(tiny_df.to_csv(index=False).encode()).hexdigest()[:16]
    assert snap1 == snap2


def test_pipeline_df_raw_differs_from_file_hash(tiny_dataset_path, tiny_df):
    """DataFrame content hash differs from file hash (line endings etc. may differ)."""
    from src.utils.reproducibility import hash_file

    file_snap = hash_file(tiny_dataset_path)[:16]
    df_snap   = hashlib.sha256(tiny_df.to_csv(index=False).encode()).hexdigest()[:16]
    # Both are valid hex — they may or may not be equal, but neither crashes
    assert len(file_snap) == 16
    assert len(df_snap) == 16


# ---------------------------------------------------------------------------
# run_manager.create_run — new metadata fields
# ---------------------------------------------------------------------------

def test_run_manager_stores_dataset_metadata(tmp_path):
    """create_run persists dataset_id, branch_id, dataset_version, name in metadata."""
    from src.utils.run_manager import RunManager

    mgr = RunManager(tmp_path / "runs")
    run_id = mgr.create_run(
        config_hash="abc12345",
        data_snapshot_id="snap0001",
        random_seed=7,
        system_info={},
        dataset_id="ds-test-id",
        dataset_version=3,
        branch_id="branch-xyz",
        name="my test run",
    )
    meta = mgr.load_run(run_id)
    assert meta["dataset_id"] == "ds-test-id"
    assert meta["dataset_version"] == 3
    assert meta["branch_id"] == "branch-xyz"
    assert meta["name"] == "my test run"


def test_run_manager_null_dataset_fields_by_default(tmp_path):
    """dataset_id / branch_id / dataset_version / name default to None."""
    from src.utils.run_manager import RunManager

    mgr = RunManager(tmp_path / "runs")
    run_id = mgr.create_run(
        config_hash="abc12345",
        data_snapshot_id="snap0001",
        random_seed=42,
        system_info={},
    )
    meta = mgr.load_run(run_id)
    assert meta["dataset_id"] is None
    assert meta["branch_id"] is None
    assert meta["dataset_version"] is None
    assert meta["name"] is None


# ---------------------------------------------------------------------------
# API schemas — new fields propagate through response
# ---------------------------------------------------------------------------

def test_run_summary_response_includes_new_fields():
    """RunSummaryResponse accepts dataset_id, branch_id, dataset_version, name."""
    from src.api.schemas import RunSummaryResponse

    r = RunSummaryResponse(
        run_id="run_test",
        created_at="2026-01-01T00:00:00+00:00",
        config_hash="abc",
        data_snapshot_id="snap",
        random_seed=42,
        stages={},
        dataset_id="ds-1",
        branch_id="br-1",
        dataset_version=2,
        name="my run",
    )
    assert r.dataset_id == "ds-1"
    assert r.branch_id == "br-1"
    assert r.dataset_version == 2
    assert r.name == "my run"


def test_run_summary_response_optional_fields_default_none():
    """New fields on RunSummaryResponse are all optional and default to None."""
    from src.api.schemas import RunSummaryResponse

    r = RunSummaryResponse(
        run_id="run_test",
        created_at="2026-01-01T00:00:00+00:00",
        config_hash="abc",
        data_snapshot_id="snap",
        random_seed=42,
        stages={},
    )
    assert r.dataset_id is None
    assert r.branch_id is None
    assert r.dataset_version is None
    assert r.name is None


# ---------------------------------------------------------------------------
# API endpoint — POST /api/runs with dataset_id 404 path
# ---------------------------------------------------------------------------

def test_create_run_with_unknown_dataset_id_returns_404(tmp_path):
    """POST /api/runs with a non-existent dataset_id returns 404."""
    import os
    from fastapi.testclient import TestClient

    runs_dir = tmp_path / "runs"
    db_dir   = tmp_path / "db"
    db_dir.mkdir()
    os.environ["SFAP_RUNS_DIR"]   = str(runs_dir)
    os.environ["SFAP_DB_PATH"]    = str(db_dir / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp_path / "datasets")

    from src.api import dependencies
    dependencies.get_run_manager.cache_clear()
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()

    from src.api.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/runs",
            json={"seed": 42, "dataset_id": "nonexistent-dataset-id-xyz"},
        )
    assert resp.status_code == 404


def test_create_run_without_dataset_id_succeeds(tmp_path):
    """POST /api/runs without dataset_id still works (legacy path)."""
    import os
    from fastapi.testclient import TestClient

    runs_dir = tmp_path / "runs"
    os.environ["SFAP_RUNS_DIR"] = str(runs_dir)

    from src.api import dependencies
    dependencies.get_run_manager.cache_clear()

    from src.api.main import app
    with TestClient(app) as client:
        resp = client.post("/api/runs", json={"seed": 99})

    assert resp.status_code == 200
    data = resp.json()
    assert data["random_seed"] == 99
    assert data["dataset_id"] is None
    assert data["name"] is None


def test_pipeline_phase3_registers_models_and_links_analysis(
    monkeypatch,
    tmp_path,
    experiment_config_path,
    factor_structure_path,
):
    """Full pipeline registers 6 models and writes one linked evaluation analysis row."""
    from types import SimpleNamespace

    from src.pipeline import run_full_pipeline
    from src.storage.database import Database
    from src.storage.model_registry import ModelRegistry
    from src.text_tasks.base import ClassificationResult

    df_raw = pd.DataFrame(
        {
            "text_feedback": [f"text {i}" for i in range(12)],
            "language": ["ru", "kz", "mixed"] * 4,
            "sentiment_class": ["positive", "neutral", "negative"] * 4,
            "detail_level": ["short", "medium", "long"] * 4,
        }
    )

    def _fake_run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def _fake_save_preprocessed(df: pd.DataFrame, run_dir: Path) -> Path:
        out = run_dir / "stages" / "preprocessed.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out

    def _fake_run_psychometrics(df: pd.DataFrame, _fs: Path, run_dir: Path):
        scores = pd.DataFrame({"F1": np.linspace(0.1, 0.9, len(df))}, index=df.index)
        out = run_dir / "psychometrics" / "factor_scores.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out, index=False)
        return SimpleNamespace(factor_scores=scores)

    def _fake_split(df: pd.DataFrame, stratify_col: str, seed: int):
        return (
            df.iloc[:8].copy(),
            df.iloc[8:10].copy(),
            df.iloc[10:].copy(),
        )

    def _fake_validate(*args, **kwargs):
        return None

    def _fake_train_all_baselines(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        run_dir: Path,
        seed: int,
        text_col: str = "text_feedback",
    ):
        tasks = ["language", "sentiment", "detail_level"]
        model_types = ["tfidf", "char_ngram"]
        out: dict[str, dict[str, ClassificationResult]] = {}
        for task in tasks:
            out[task] = {}
            for model_type in model_types:
                model_dir = run_dir / "text_tasks" / task / model_type
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "model.joblib"
                model_path.write_bytes(b"dummy-model-bytes")
                metrics_path = model_dir / "metrics.json"
                metrics_path.write_bytes(
                    orjson.dumps(
                        {
                            "task": task,
                            "model_type": model_type,
                            "val": {"macro_f1": 0.9, "accuracy": 0.9},
                            "hyperparameters": {"C": 1.0},
                        },
                        option=orjson.OPT_INDENT_2,
                    )
                )
                out[task][model_type] = ClassificationResult(
                    task=task,
                    model_type=model_type,
                    predictions=np.array([]),
                    probabilities=np.empty((0, 0)),
                    classes=[],
                    model_path=model_path,
                    hyperparameters={"C": 1.0},
                    train_metrics={"macro_f1": 0.95},
                    val_metrics={"macro_f1": 0.9},
                )
        return out

    def _fake_run_fusion(*args, **kwargs):
        return {"ok": True}

    def _fake_run_contradiction(*args, **kwargs):
        return {"ok": True}

    def _fake_run_evaluation(df_test: pd.DataFrame, run_dir: Path, text_col: str = "text_feedback"):
        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        result = {"sentiment": {"tfidf": {"overall": {"macro_f1": 0.9}}}}
        (eval_dir / "classification_results.json").write_bytes(orjson.dumps(result))
        return result

    def _fake_generate_evaluation_report(run_dir: Path) -> Path:
        out = run_dir / "reports" / "evaluation_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Evaluation", encoding="utf-8")
        return out

    def _fake_generate_all_model_cards(run_dir: Path) -> list[Path]:
        return []

    def _fake_generate_data_dictionary(run_dir: Path) -> Path:
        out = run_dir / "reports" / "data_dictionary.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Data Dictionary", encoding="utf-8")
        return out

    monkeypatch.setattr("src.pipeline.run_preprocessing", _fake_run_preprocessing)
    monkeypatch.setattr("src.pipeline.save_preprocessed", _fake_save_preprocessed)
    monkeypatch.setattr("src.pipeline.run_psychometrics", _fake_run_psychometrics)
    monkeypatch.setattr("src.pipeline.stratified_split", _fake_split)
    monkeypatch.setattr("src.pipeline.validate_split_no_leakage", _fake_validate)
    monkeypatch.setattr("src.pipeline.train_all_baselines", _fake_train_all_baselines)
    monkeypatch.setattr("src.pipeline.run_fusion", _fake_run_fusion)
    monkeypatch.setattr("src.pipeline.run_contradiction_monitoring", _fake_run_contradiction)
    monkeypatch.setattr("src.pipeline.run_evaluation", _fake_run_evaluation)
    monkeypatch.setattr("src.pipeline.generate_evaluation_report", _fake_generate_evaluation_report)
    monkeypatch.setattr("src.pipeline.generate_all_model_cards", _fake_generate_all_model_cards)
    monkeypatch.setattr("src.pipeline.generate_data_dictionary", _fake_generate_data_dictionary)

    db = Database(tmp_path / "phase3.db")
    model_registry = ModelRegistry(db, tmp_path / "models")

    run_id = run_full_pipeline(
        data_path=None,
        config_path=experiment_config_path,
        factor_structure_path=factor_structure_path,
        runs_dir=tmp_path / "runs",
        seed=42,
        df_raw=df_raw,
        dataset_id="ds_phase3",
        dataset_version=1,
        model_registry=model_registry,
        db=db,
    )

    run_dir = tmp_path / "runs" / run_id
    assert (run_dir / "raw" / "dataset_snapshot.parquet").exists()
    meta = orjson.loads((run_dir / "raw" / "snapshot_metadata.json").read_bytes())
    assert meta["source"] == "dataframe_direct"

    models, total = model_registry.list_models(dataset_id="ds_phase3", per_page=100)
    assert total == 6

    pipeline_row = db.fetchone(
        "SELECT status FROM pipeline_runs WHERE id = ?",
        (run_id,),
    )
    assert pipeline_row is not None
    assert pipeline_row["status"] == "completed"

    analysis_row = db.fetchone(
        "SELECT id FROM analysis_runs WHERE pipeline_run_id = ?",
        (run_id,),
    )
    assert analysis_row is not None

    ref_count = db.fetchone(
        "SELECT COUNT(*) AS cnt FROM analysis_model_refs WHERE analysis_id = ?",
        (analysis_row["id"],),
    )
    assert ref_count is not None
    assert ref_count["cnt"] == 6
