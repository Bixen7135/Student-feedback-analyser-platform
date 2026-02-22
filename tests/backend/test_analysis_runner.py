"""Unit and integration tests for the analysis runner."""
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
    """Write a small CSV with text + label columns."""
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


def _train_model_for_analysis(tmp: Path, dataset_manager, model_registry, task="sentiment"):
    """Train and register a model; return model_id."""
    from src.training.runner import run_training, TrainingConfig

    artifacts_dir = tmp / "training_runs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ds_list, _ = dataset_manager.list_datasets()
    dataset_id = ds_list[0].id

    result = run_training(
        dataset_id=dataset_id,
        task=task,
        model_type="tfidf",
        dataset_manager=dataset_manager,
        model_registry=model_registry,
        artifacts_dir=artifacts_dir,
        seed=42,
    )
    return result["model_id"], dataset_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def storage(tmp_path_factory):
    """Set up isolated DB, DatasetManager, and ModelRegistry."""
    tmp = tmp_path_factory.mktemp("analysis_runner")

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
    dataset_manager = DatasetManager(db, tmp / "datasets")
    model_registry = ModelRegistry(db, tmp / "models")

    # Upload a dataset
    csv_path = tmp / "sample.csv"
    _make_labelled_csv(csv_path)
    dataset_manager.upload_dataset(
        file_path=csv_path,
        name="analysis_test_ds",
    )

    yield {
        "db": db,
        "dataset_manager": dataset_manager,
        "model_registry": model_registry,
        "tmp": tmp,
    }

    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR"]:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Tests: _detect_text_col
# ---------------------------------------------------------------------------


class TestDetectTextCol:
    def test_detects_text_feedback(self):
        from src.analysis.runner import _detect_text_col
        import pandas as pd

        df = pd.DataFrame({"text_feedback": ["a", "b"], "q1": [1, 2]})
        assert _detect_text_col(df) == "text_feedback"

    def test_detects_fallback_object_col(self):
        from src.analysis.runner import _detect_text_col
        import pandas as pd

        df = pd.DataFrame({"score": [1, 2], "review": ["good", "bad"]})
        # "review" matches a candidate
        assert _detect_text_col(df) == "review"

    def test_returns_none_for_all_numeric(self):
        from src.analysis.runner import _detect_text_col
        import pandas as pd

        df = pd.DataFrame({"q1": [1, 2], "q2": [3, 4]})
        assert _detect_text_col(df) is None


# ---------------------------------------------------------------------------
# Tests: run_analysis
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    def test_run_analysis_basic(self, storage, tmp_path):
        """run_analysis runs to completion and saves results.csv + summary.json."""
        from src.analysis.runner import run_analysis

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        # Train a model first
        model_id, dataset_id = _train_model_for_analysis(tmp, dm, mr, task="sentiment")

        artifacts_dir = tmp / "analysis_runs"
        analysis_id = "test_analysis_001"

        result = run_analysis(
            dataset_id=dataset_id,
            model_ids=[model_id],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=artifacts_dir,
            analysis_id=analysis_id,
            name="Test Analysis",
        )

        # Check result structure
        assert result["analysis_id"] == analysis_id
        assert result["dataset_id"] == dataset_id
        assert result["n_rows"] == 45  # 3 classes * 15 rows
        assert len(result["models_applied"]) == 1

        m = result["models_applied"][0]
        assert m["model_id"] == model_id
        assert m["task"] == "sentiment"
        assert m["error"] is None
        assert m["n_predicted"] == 45
        assert m["pred_col"] is not None
        assert m["conf_col"] is not None
        assert len(m["class_distribution"]) >= 2
        # distribution should sum to ~1
        total = sum(m["class_distribution"].values())
        assert abs(total - 1.0) < 0.01

    def test_results_csv_saved(self, storage, tmp_path):
        """results.csv should exist and have prediction columns."""
        from src.analysis.runner import run_analysis, get_results_path

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        model_id, dataset_id = _train_model_for_analysis(tmp, dm, mr, task="language")
        artifacts_dir = tmp / "analysis_runs"
        analysis_id = "test_analysis_002"

        run_analysis(
            dataset_id=dataset_id,
            model_ids=[model_id],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=artifacts_dir,
            analysis_id=analysis_id,
        )

        results_path = get_results_path(artifacts_dir, analysis_id)
        assert results_path.exists()

        df = pd.read_csv(results_path)
        assert len(df) == 45
        # Check original columns preserved
        assert "text_feedback" in df.columns
        assert "sentiment_class" in df.columns
        # Check prediction column added
        pred_cols = [c for c in df.columns if c.endswith("_pred")]
        assert len(pred_cols) >= 1

    def test_summary_json_saved(self, storage):
        """summary.json should exist with correct structure."""
        import orjson
        from src.analysis.runner import get_summary_path

        tmp = storage["tmp"]
        artifacts_dir = tmp / "analysis_runs"
        summary_path = get_summary_path(artifacts_dir, "test_analysis_001")
        assert summary_path.exists()

        summary = orjson.loads(summary_path.read_bytes())
        assert summary["analysis_id"] == "test_analysis_001"
        assert "models_applied" in summary
        assert "n_rows" in summary

    def test_db_record_created(self, storage):
        """Analysis run should be persisted to SQLite."""
        from src.analysis.runner import get_analysis_from_db

        db = storage["db"]
        record = get_analysis_from_db(db, "test_analysis_001")
        assert record is not None
        assert record["id"] == "test_analysis_001"
        assert record["status"] == "completed"
        assert record["dataset_id"] is not None

    def test_unknown_model_gracefully_logged(self, storage):
        """Unknown model_id should produce an error entry, not crash."""
        from src.analysis.runner import run_analysis

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        ds_list, _ = dm.list_datasets()
        dataset_id = ds_list[0].id

        artifacts_dir = tmp / "analysis_runs"
        result = run_analysis(
            dataset_id=dataset_id,
            model_ids=["nonexistent_model_id"],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=artifacts_dir,
            analysis_id="test_analysis_bad_model",
        )
        # Should complete but have no models_applied entries from the bad id
        assert result["n_rows"] > 0
        # The bad model is silently skipped (warning logged)
        applied = result["models_applied"]
        assert len(applied) == 0

    def test_multiple_models_applied(self, storage):
        """Two models (different tasks) both applied to same dataset."""
        from src.analysis.runner import run_analysis

        db = storage["db"]
        dm = storage["dataset_manager"]
        mr = storage["model_registry"]
        tmp = storage["tmp"]

        sentiment_id, dataset_id = _train_model_for_analysis(tmp, dm, mr, task="sentiment")
        language_id, _ = _train_model_for_analysis(tmp, dm, mr, task="language")

        artifacts_dir = tmp / "analysis_runs"
        result = run_analysis(
            dataset_id=dataset_id,
            model_ids=[sentiment_id, language_id],
            dataset_manager=dm,
            model_registry=mr,
            db=db,
            artifacts_dir=artifacts_dir,
            analysis_id="test_analysis_multi",
        )

        assert len(result["models_applied"]) == 2
        tasks = {m["task"] for m in result["models_applied"]}
        assert "sentiment" in tasks
        assert "language" in tasks


# ---------------------------------------------------------------------------
# Tests: load_results_page
# ---------------------------------------------------------------------------


class TestLoadResultsPage:
    def test_basic_pagination(self, storage):
        from src.analysis.runner import load_results_page

        tmp = storage["tmp"]
        artifacts_dir = tmp / "analysis_runs"

        page = load_results_page(artifacts_dir, "test_analysis_001", offset=0, limit=10)
        assert page["total_rows"] == 45
        assert page["offset"] == 0
        assert page["limit"] == 10
        assert len(page["rows"]) == 10
        assert len(page["columns"]) > 0

    def test_offset_pagination(self, storage):
        from src.analysis.runner import load_results_page

        tmp = storage["tmp"]
        artifacts_dir = tmp / "analysis_runs"

        page = load_results_page(artifacts_dir, "test_analysis_001", offset=40, limit=10)
        assert len(page["rows"]) == 5  # only 5 rows left at offset 40

    def test_filter_by_col_val(self, storage):
        from src.analysis.runner import load_results_page

        tmp = storage["tmp"]
        artifacts_dir = tmp / "analysis_runs"

        page = load_results_page(
            artifacts_dir, "test_analysis_001",
            offset=0, limit=100,
            filter_col="language", filter_val="ru",
        )
        assert page["total_rows"] > 0
        # All returned rows should have language == ru
        lang_idx = page["columns"].index("language")
        for row in page["rows"]:
            assert row[lang_idx].lower() == "ru"

    def test_nonexistent_analysis_returns_empty(self, storage):
        from src.analysis.runner import load_results_page

        tmp = storage["tmp"]
        page = load_results_page(tmp / "analysis_runs", "does_not_exist", offset=0, limit=10)
        assert page["total_rows"] == 0
        assert page["rows"] == []


# ---------------------------------------------------------------------------
# Tests: DB helpers
# ---------------------------------------------------------------------------


class TestDbHelpers:
    def test_list_analyses(self, storage):
        from src.analysis.runner import list_analyses_from_db

        db = storage["db"]
        analyses, total = list_analyses_from_db(db)
        assert total >= 3  # from prior tests
        assert all("id" in a for a in analyses)

    def test_list_filter_by_status(self, storage):
        from src.analysis.runner import list_analyses_from_db

        db = storage["db"]
        analyses, total = list_analyses_from_db(db, status="completed")
        for a in analyses:
            assert a["status"] == "completed"

    def test_update_metadata(self, storage):
        from src.analysis.runner import update_analysis_metadata, get_analysis_from_db

        db = storage["db"]
        updated = update_analysis_metadata(
            db, "test_analysis_001",
            name="Updated Name",
            tags=["new-tag"],
            comments="A note.",
        )
        assert updated is not None
        assert updated["name"] == "Updated Name"
        assert "new-tag" in updated["tags"]

        # Verify persisted
        record = get_analysis_from_db(db, "test_analysis_001")
        assert record["name"] == "Updated Name"

    def test_delete_analysis(self, storage):
        from src.analysis.runner import (
            delete_analysis_from_db,
            get_analysis_from_db,
            _upsert_analysis_run,
        )

        db = storage["db"]
        # Create a temporary record
        _upsert_analysis_run(
            db=db,
            analysis_id="temp_to_delete",
            name="To Delete",
            description="",
            tags=[],
            dataset_id=None,
            dataset_version=None,
            model_ids=[],
            status="completed",
        )
        assert get_analysis_from_db(db, "temp_to_delete") is not None

        deleted = delete_analysis_from_db(db, "temp_to_delete")
        assert deleted is True
        assert get_analysis_from_db(db, "temp_to_delete") is None

    def test_delete_nonexistent_returns_false(self, storage):
        from src.analysis.runner import delete_analysis_from_db

        db = storage["db"]
        deleted = delete_analysis_from_db(db, "no_such_id")
        assert deleted is False
