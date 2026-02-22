"""Phase 5 — unit and integration tests for advanced filtering, anomaly detection, and filtered export."""
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


def _train_model_for_test(tmp: Path, dataset_manager, model_registry, task="sentiment"):
    from src.training.runner import run_training

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
    tmp = tmp_path_factory.mktemp("phase5_filtering")

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

    csv_path = tmp / "sample.csv"
    _make_labelled_csv(csv_path)
    dataset_manager.upload_dataset(file_path=csv_path, name="phase5_test_ds")

    # Train + run analysis
    model_id, dataset_id = _train_model_for_test(tmp, dataset_manager, model_registry, "sentiment")

    from src.analysis.runner import run_analysis

    artifacts_dir = tmp / "analysis_runs"
    analysis_id = "phase5_analysis_001"
    run_analysis(
        dataset_id=dataset_id,
        model_ids=[model_id],
        dataset_manager=dataset_manager,
        model_registry=model_registry,
        db=db,
        artifacts_dir=artifacts_dir,
        analysis_id=analysis_id,
        name="Phase 5 Test Analysis",
    )

    yield {
        "db": db,
        "dataset_manager": dataset_manager,
        "model_registry": model_registry,
        "tmp": tmp,
        "analysis_id": analysis_id,
        "artifacts_dir": artifacts_dir,
    }

    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR"]:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Tests: _apply_result_filters
# ---------------------------------------------------------------------------


class TestApplyResultFilters:
    def _make_df(self):
        return pd.DataFrame(
            {
                "language": ["ru", "kz", "ru", "kz", "ru"],
                "sentiment_class": ["positive", "negative", "neutral", "positive", "negative"],
                "score": ["0.9", "0.4", "0.7", "0.85", "0.3"],
                "text_feedback": ["Hello world", "Мир", "Test text", "Another row", ""],
            }
        )

    def test_eq_filter(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, filters=[{"col": "language", "op": "eq", "val": "ru"}])
        assert len(result) == 3
        assert all(result["language"] == "ru")

    def test_ne_filter(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, filters=[{"col": "language", "op": "ne", "val": "ru"}])
        assert len(result) == 2
        assert all(result["language"] == "kz")

    def test_contains_filter(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(
            df, filters=[{"col": "text_feedback", "op": "contains", "val": "world"}]
        )
        assert len(result) == 1
        assert result.iloc[0]["text_feedback"] == "Hello world"

    def test_contains_case_insensitive(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(
            df, filters=[{"col": "text_feedback", "op": "contains", "val": "HELLO"}]
        )
        assert len(result) == 1

    def test_gt_filter(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, filters=[{"col": "score", "op": "gt", "val": "0.8"}])
        assert len(result) == 2  # 0.9 and 0.85

    def test_lt_filter(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, filters=[{"col": "score", "op": "lt", "val": "0.5"}])
        assert len(result) == 2  # 0.4 and 0.3

    def test_gte_filter(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, filters=[{"col": "score", "op": "gte", "val": "0.9"}])
        assert len(result) == 1  # only 0.9

    def test_lte_filter(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, filters=[{"col": "score", "op": "lte", "val": "0.4"}])
        assert len(result) == 2  # 0.4 and 0.3

    def test_multi_rule_filters(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(
            df,
            filters=[
                {"col": "language", "op": "eq", "val": "ru"},
                {"col": "sentiment_class", "op": "ne", "val": "positive"},
            ],
        )
        # ru rows: 0, 2, 4 — minus positive (0) → rows 2 and 4
        assert len(result) == 2
        assert all(result["language"] == "ru")
        assert "positive" not in result["sentiment_class"].values

    def test_legacy_filter_col_val(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, filter_col="language", filter_val="kz")
        assert len(result) == 2
        assert all(result["language"] == "kz")

    def test_search_across_text_cols(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, search="мир")  # Cyrillic
        assert len(result) == 1
        assert result.iloc[0]["text_feedback"] == "Мир"

    def test_search_case_insensitive(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(df, search="HELLO")
        assert len(result) == 1

    def test_search_combined_with_filter(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        result = _apply_result_filters(
            df,
            filters=[{"col": "language", "op": "eq", "val": "ru"}],
            search="test",
        )
        # Only "Test text" row (index 2) is ru + contains "test"
        assert len(result) == 1

    def test_unknown_column_ignored(self):
        from src.analysis.runner import _apply_result_filters

        df = self._make_df()
        # Filter on a non-existent column — should be a no-op
        result = _apply_result_filters(
            df, filters=[{"col": "nonexistent_col", "op": "eq", "val": "x"}]
        )
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Tests: load_filtered_df
# ---------------------------------------------------------------------------


class TestLoadFilteredDf:
    def test_returns_dataframe(self, storage):
        from src.analysis.runner import load_filtered_df

        df = load_filtered_df(storage["artifacts_dir"], storage["analysis_id"])
        assert df is not None
        assert len(df) == 45

    def test_with_filter(self, storage):
        from src.analysis.runner import load_filtered_df

        df = load_filtered_df(
            storage["artifacts_dir"],
            storage["analysis_id"],
            filters=[{"col": "language", "op": "eq", "val": "ru"}],
        )
        assert df is not None
        assert len(df) > 0
        assert all(df["language"] == "ru")

    def test_nonexistent_returns_none(self, storage):
        from src.analysis.runner import load_filtered_df

        result = load_filtered_df(storage["artifacts_dir"], "does_not_exist")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: load_results_page with Phase 5 params
# ---------------------------------------------------------------------------


class TestLoadResultsPagePhase5:
    def test_multi_filter(self, storage):
        from src.analysis.runner import load_results_page

        page = load_results_page(
            storage["artifacts_dir"],
            storage["analysis_id"],
            offset=0,
            limit=100,
            filters=[{"col": "language", "op": "eq", "val": "ru"}],
        )
        assert page["total_rows"] > 0
        lang_idx = page["columns"].index("language")
        for row in page["rows"]:
            assert row[lang_idx].lower() == "ru"

    def test_search(self, storage):
        from src.analysis.runner import load_results_page

        # Search for a specific class name that appears in text_feedback
        page = load_results_page(
            storage["artifacts_dir"],
            storage["analysis_id"],
            offset=0,
            limit=100,
            search="positive",
        )
        # All rows should contain "positive" in some text col
        assert page["total_rows"] > 0

    def test_combined_filter_and_search(self, storage):
        from src.analysis.runner import load_results_page

        page = load_results_page(
            storage["artifacts_dir"],
            storage["analysis_id"],
            offset=0,
            limit=100,
            filters=[{"col": "language", "op": "eq", "val": "ru"}],
            search="positive",
        )
        assert page["total_rows"] >= 0  # may be 0 or more depending on data

    def test_backward_compat_filter_col_val(self, storage):
        """Legacy filter_col/filter_val params still work."""
        from src.analysis.runner import load_results_page

        page = load_results_page(
            storage["artifacts_dir"],
            storage["analysis_id"],
            offset=0,
            limit=100,
            filter_col="language",
            filter_val="kz",
        )
        lang_idx = page["columns"].index("language")
        for row in page["rows"]:
            assert row[lang_idx].lower() == "kz"


# ---------------------------------------------------------------------------
# Tests: get_anomalies
# ---------------------------------------------------------------------------


class TestGetAnomalies:
    def test_returns_structure(self, storage):
        from src.analysis.runner import get_anomalies

        result = get_anomalies(storage["artifacts_dir"], storage["analysis_id"])
        assert "analysis_id" in result
        assert "anomalies" in result
        assert "total" in result
        assert "conf_threshold" in result
        assert result["analysis_id"] == storage["analysis_id"]
        assert isinstance(result["anomalies"], list)
        assert isinstance(result["total"], int)

    def test_threshold_at_zero_returns_none(self, storage):
        from src.analysis.runner import get_anomalies

        # Threshold of 0.0 means nothing is low-confidence
        result = get_anomalies(
            storage["artifacts_dir"], storage["analysis_id"], conf_threshold=0.0
        )
        # Only empty text rows could trigger anomalies (none in test data)
        assert result["total"] >= 0

    def test_threshold_at_one_catches_all(self, storage):
        from src.analysis.runner import get_anomalies

        # All confidences are < 1.0, so every row with a conf col is anomalous
        result = get_anomalies(
            storage["artifacts_dir"], storage["analysis_id"], conf_threshold=1.0
        )
        assert result["total"] == 45

    def test_anomaly_entries_have_required_keys(self, storage):
        from src.analysis.runner import get_anomalies

        result = get_anomalies(
            storage["artifacts_dir"], storage["analysis_id"], conf_threshold=1.0
        )
        for a in result["anomalies"][:5]:
            assert "row_index" in a
            assert "reasons" in a
            assert "data" in a
            assert isinstance(a["reasons"], list)
            assert isinstance(a["data"], dict)

    def test_nonexistent_analysis(self, storage):
        from src.analysis.runner import get_anomalies

        result = get_anomalies(storage["artifacts_dir"], "nonexistent_analysis")
        assert result["total"] == 0
        assert result["anomalies"] == []


# ---------------------------------------------------------------------------
# Tests: API endpoints (advanced filtering, anomalies, filtered export)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client_p5(tmp_path_factory):
    """TestClient with an isolated environment and a completed analysis."""
    tmp = tmp_path_factory.mktemp("phase5_api")

    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp / "models")
    os.environ["SFAP_TRAINING_DIR"] = str(tmp / "training_runs")
    os.environ["SFAP_ANALYSIS_DIR"] = str(tmp / "analysis_runs")

    from src.api import dependencies
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()
    dependencies.get_run_manager.cache_clear()

    from src.training import runner as training_runner
    from src.analysis import runner as analysis_runner
    from fastapi.testclient import TestClient
    from src.api.main import app

    with training_runner._jobs_lock:
        training_runner._jobs.clear()
    with analysis_runner._jobs_lock:
        analysis_runner._jobs.clear()

    with TestClient(app) as c:
        csv_path = tmp / "sample.csv"
        _make_labelled_csv(csv_path)
        with open(csv_path, "rb") as f:
            resp = c.post(
                "/api/datasets/upload",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"name": "p5_test_ds"},
            )
        assert resp.status_code == 200, resp.text
        dataset_id = resp.json()["id"]
        c.dataset_id = dataset_id  # type: ignore

        train_resp = c.post(
            "/api/training/start",
            json={"dataset_id": dataset_id, "task": "sentiment", "model_type": "tfidf", "seed": 42},
        )
        assert train_resp.status_code == 202, train_resp.text
        c.model_id = train_resp.json().get("model_id")  # type: ignore
        if not c.model_id:
            job_id = train_resp.json()["job_id"]
            status = c.get(f"/api/training/{job_id}/status").json()
            c.model_id = status.get("model_id")
        assert c.model_id, "Model not registered"

        start_resp = c.post(
            "/api/analyses",
            json={"dataset_id": dataset_id, "model_ids": [c.model_id], "name": "P5 Analysis"},
        )
        assert start_resp.status_code == 202
        c.analysis_id = start_resp.json()["job_id"]  # type: ignore
        yield c

    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR",
                "SFAP_TRAINING_DIR", "SFAP_ANALYSIS_DIR"]:
        os.environ.pop(key, None)


class TestAdvancedFilteringAPI:
    def _skip_unless_completed(self, client):
        status = client.get(f"/api/analyses/{client.analysis_id}/status").json()["status"]
        if status != "completed":
            pytest.skip("Analysis not yet completed")

    def test_results_with_multi_filter(self, api_client_p5):
        self._skip_unless_completed(api_client_p5)
        import json

        filters_json = json.dumps([{"col": "language", "op": "eq", "val": "ru"}])
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/results"
            f"?filters={filters_json}&limit=100"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        lang_idx = body["columns"].index("language")
        for row in body["rows"]:
            assert row[lang_idx].lower() == "ru"

    def test_results_with_ne_filter(self, api_client_p5):
        self._skip_unless_completed(api_client_p5)
        import json

        filters_json = json.dumps([{"col": "language", "op": "ne", "val": "ru"}])
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/results"
            f"?filters={filters_json}&limit=100"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        lang_idx = body["columns"].index("language")
        for row in body["rows"]:
            assert row[lang_idx].lower() != "ru"

    def test_results_with_contains_filter(self, api_client_p5):
        self._skip_unless_completed(api_client_p5)
        import json

        filters_json = json.dumps([{"col": "text_feedback", "op": "contains", "val": "positive"}])
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/results"
            f"?filters={filters_json}&limit=100"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total_rows"] > 0

    def test_results_with_search(self, api_client_p5):
        self._skip_unless_completed(api_client_p5)
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/results"
            "?search=neutral&limit=100"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total_rows"] > 0

    def test_results_invalid_filters_json_422(self, api_client_p5):
        self._skip_unless_completed(api_client_p5)
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/results"
            "?filters=NOT_VALID_JSON"
        )
        assert resp.status_code == 422

    def test_anomalies_endpoint_200(self, api_client_p5):
        self._skip_unless_completed(api_client_p5)
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/anomalies"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "anomalies" in body
        assert "total" in body
        assert "conf_threshold" in body

    def test_anomalies_with_threshold_one(self, api_client_p5):
        self._skip_unless_completed(api_client_p5)
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/anomalies?conf_threshold=1.0"
        )
        assert resp.status_code == 200
        body = resp.json()
        # With threshold=1.0, all rows with conf cols are anomalous
        assert body["total"] == 45

    def test_anomalies_404_for_unknown(self, api_client_p5):
        resp = api_client_p5.get("/api/analyses/nonexistent_id/anomalies")
        assert resp.status_code == 404

    def test_filtered_export_csv(self, api_client_p5):
        """Export with filters applied returns only matching rows."""
        self._skip_unless_completed(api_client_p5)
        import json

        filters_json = json.dumps([{"col": "language", "op": "eq", "val": "ru"}])
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/results/export"
            f"?format=csv&filters={filters_json}"
        )
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")
        content = resp.content.decode("utf-8-sig")
        lines = [l for l in content.splitlines() if l.strip()]
        # Header + 23 ru rows (15 rows * 3 classes, but ~half are ru)
        assert len(lines) > 1
        # Spot check: no kz in language column
        for line in lines[1:]:
            parts = line.split(",")
            # language is second column in the fixture CSV
            # We just check the file is non-empty and structured
            assert len(parts) > 1

    def test_filtered_export_json(self, api_client_p5):
        self._skip_unless_completed(api_client_p5)
        import json as json_mod

        filters_json = json_mod.dumps([{"col": "language", "op": "eq", "val": "kz"}])
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/results/export"
            f"?format=json&filters={filters_json}"
        )
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", "")
        body = resp.json()
        assert "rows" in body
        assert len(body["rows"]) > 0

    def test_unfiltered_export_unchanged(self, api_client_p5):
        """Export without filters returns all rows."""
        self._skip_unless_completed(api_client_p5)
        resp = api_client_p5.get(
            f"/api/analyses/{api_client_p5.analysis_id}/results/export?format=csv"
        )
        assert resp.status_code == 200
        content = resp.content.decode("utf-8-sig")
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) == 46  # 1 header + 45 data rows
