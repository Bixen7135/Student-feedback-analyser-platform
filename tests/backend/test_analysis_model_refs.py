"""Tests for analysis_model_refs normalized table and related behaviour.

Phase 2 — Issue 7 (DB as sole source of truth) and Issue 9 (normalized refs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labelled_csv(path: Path, n_per_class: int = 15) -> None:
    rows = []
    for s in ["positive", "neutral", "negative"]:
        for i in range(n_per_class):
            rows.append(
                {
                    "text_feedback": f"Feedback {i} class {s} extra words here long text",
                    "language": "ru" if i % 2 == 0 else "kz",
                    "sentiment_class": s,
                    "detail_level": "short" if i < 5 else ("medium" if i < 10 else "long"),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _register_dummy_model(registry, db, tmp, task="sentiment") -> str:
    """Train and register a model; return model_id."""
    from src.training.runner import run_training

    artifacts_dir = tmp / "training_runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    from src.storage.dataset_manager import DatasetManager
    dm = DatasetManager(db, tmp / "datasets")
    ds_list, _ = dm.list_datasets()
    dataset_id = ds_list[0].id

    result = run_training(
        dataset_id=dataset_id,
        task=task,
        model_type="tfidf",
        dataset_manager=dm,
        model_registry=registry,
        artifacts_dir=artifacts_dir,
        seed=42,
    )
    return result["model_id"], dataset_id


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def storage(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("amr_tests")

    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp / "models")

    from src.api import dependencies
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()

    from src.storage.database import Database
    from src.storage.dataset_manager import DatasetManager
    from src.storage.model_registry import ModelRegistry

    db = Database(tmp / "test.db")
    dm = DatasetManager(db, tmp / "datasets")
    mr = ModelRegistry(db, tmp / "models")

    csv_path = tmp / "sample.csv"
    _make_labelled_csv(csv_path)
    dm.upload_dataset(file_path=csv_path, name="amr_test_ds")

    yield {"db": db, "dataset_manager": dm, "model_registry": mr, "tmp": tmp}

    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR"]:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnalysisModelRefs:
    def test_refs_populated_on_run_analysis(self, storage):
        """analysis_model_refs rows are written when run_analysis completes."""
        from src.analysis.runner import run_analysis

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        model_id, dataset_id = _register_dummy_model(mr, db, tmp, task="sentiment")
        artifacts_dir = tmp / "analysis_runs"
        analysis_id = "amr_test_001"

        run_analysis(
            dataset_id=dataset_id,
            model_ids=[model_id],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=artifacts_dir,
            analysis_id=analysis_id,
            name="AMR Test",
        )

        rows = db.fetchall(
            "SELECT * FROM analysis_model_refs WHERE analysis_id = ?",
            (analysis_id,),
        )
        assert len(rows) == 1
        assert rows[0]["model_id"] == model_id

    def test_refs_populated_on_create_job(self, storage):
        """create_job with db= inserts pending analysis_runs row and model refs."""
        from src.analysis.runner import create_job

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        model_id, dataset_id = _register_dummy_model(mr, db, tmp, task="language")
        job_id = "amr_job_001"

        create_job(
            job_id=job_id,
            dataset_id=dataset_id,
            model_ids=[model_id],
            name="AMR Job",
            description="",
            tags=[],
            dataset_version=None,
            db=db,
        )

        # Check analysis_runs row
        row = db.fetchone("SELECT * FROM analysis_runs WHERE id = ?", (job_id,))
        assert row is not None
        assert row["status"] == "pending"

        # Check model_refs
        refs = db.fetchall(
            "SELECT * FROM analysis_model_refs WHERE analysis_id = ?", (job_id,)
        )
        assert len(refs) == 1
        assert refs[0]["model_id"] == model_id

    def test_list_analyses_model_id_filter_uses_refs(self, storage):
        """list_analyses_from_db model_id filter finds analyses via analysis_model_refs."""
        from src.analysis.runner import run_analysis, list_analyses_from_db

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        model_id, dataset_id = _register_dummy_model(mr, db, tmp, task="detail_level")
        analysis_id = "amr_filter_test_001"

        run_analysis(
            dataset_id=dataset_id,
            model_ids=[model_id],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=tmp / "analysis_runs",
            analysis_id=analysis_id,
            name="Filter Test",
        )

        # Filter by the exact model_id — should return exactly this analysis
        results, total = list_analyses_from_db(db, model_id=model_id)
        ids = [r["id"] for r in results]
        assert analysis_id in ids

    def test_list_analyses_filter_does_not_match_partial_id(self, storage):
        """list_analyses_from_db model_id filter requires exact match, not substring."""
        from src.analysis.runner import list_analyses_from_db

        db = storage["db"]
        partial_id = "amr"  # substring of analysis_id prefix, NOT a real model_id

        results, total = list_analyses_from_db(db, model_id=partial_id)
        # A partial/fake model_id should return no results
        assert len(results) == 0

    def test_refs_cleaned_up_on_delete(self, storage):
        """Deleting an analysis also removes its analysis_model_refs rows."""
        from src.analysis.runner import run_analysis, delete_analysis_from_db

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        model_id, dataset_id = _register_dummy_model(mr, db, tmp, task="sentiment")
        analysis_id = "amr_delete_test_001"

        run_analysis(
            dataset_id=dataset_id,
            model_ids=[model_id],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=tmp / "analysis_runs",
            analysis_id=analysis_id,
            name="Delete Test",
        )

        # Confirm refs exist
        refs_before = db.fetchall(
            "SELECT * FROM analysis_model_refs WHERE analysis_id = ?", (analysis_id,)
        )
        assert len(refs_before) >= 1

        delete_analysis_from_db(db, analysis_id)

        # Refs should be gone
        refs_after = db.fetchall(
            "SELECT * FROM analysis_model_refs WHERE analysis_id = ?", (analysis_id,)
        )
        assert len(refs_after) == 0

    def test_delete_model_blocked_by_normalized_refs(self, storage):
        """delete_model returns deleted=False when analysis_model_refs references the model."""
        from src.analysis.runner import run_analysis

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        model_id, dataset_id = _register_dummy_model(mr, db, tmp, task="sentiment")
        analysis_id = "amr_block_delete_001"

        run_analysis(
            dataset_id=dataset_id,
            model_ids=[model_id],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=tmp / "analysis_runs",
            analysis_id=analysis_id,
            name="Block Delete Test",
        )

        result = mr.delete_model(model_id)
        assert result["deleted"] is False
        assert "analyses" in result.get("dependencies", {})

    def test_delete_model_succeeds_when_no_completed_analyses(self, storage):
        """delete_model succeeds when no non-failed analyses reference the model."""
        from src.analysis.runner import run_analysis, delete_analysis_from_db

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        model_id, dataset_id = _register_dummy_model(mr, db, tmp, task="language")
        analysis_id = "amr_allow_delete_001"

        run_analysis(
            dataset_id=dataset_id,
            model_ids=[model_id],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=tmp / "analysis_runs",
            analysis_id=analysis_id,
            name="Allow Delete Test",
        )

        # Delete the analysis first (removes refs)
        delete_analysis_from_db(db, analysis_id)

        # Now deleting the model should succeed
        result = mr.delete_model(model_id)
        assert result["deleted"] is True

    def test_multiple_models_in_one_analysis(self, storage):
        """All model_ids in an analysis are stored in analysis_model_refs."""
        from src.analysis.runner import run_analysis

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        mid1, dataset_id = _register_dummy_model(mr, db, tmp, task="sentiment")
        mid2, _ = _register_dummy_model(mr, db, tmp, task="language")
        analysis_id = "amr_multi_001"

        run_analysis(
            dataset_id=dataset_id,
            model_ids=[mid1, mid2],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=tmp / "analysis_runs",
            analysis_id=analysis_id,
            name="Multi Model Test",
        )

        refs = db.fetchall(
            "SELECT model_id FROM analysis_model_refs WHERE analysis_id = ?",
            (analysis_id,),
        )
        stored_model_ids = {r["model_id"] for r in refs}
        assert mid1 in stored_model_ids
        assert mid2 in stored_model_ids
