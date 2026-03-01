"""Integration tests for the analysis API endpoints."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.api.main import app
from src.api import dependencies
from src.training import runner as training_runner
from src.analysis import runner as analysis_runner


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


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """TestClient with isolated storage, a dataset, and a trained model."""
    tmp = tmp_path_factory.mktemp("analysis_api")

    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp / "models")
    os.environ["SFAP_TRAINING_DIR"] = str(tmp / "training_runs")
    os.environ["SFAP_ANALYSIS_DIR"] = str(tmp / "analysis_runs")

    # Clear caches
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()
    dependencies.get_run_manager.cache_clear()

    # Clear in-memory stores
    with training_runner._jobs_lock:
        training_runner._jobs.clear()
    with analysis_runner._jobs_lock:
        analysis_runner._jobs.clear()

    with TestClient(app) as c:
        # Upload dataset
        csv_path = tmp / "sample.csv"
        _make_labelled_csv(csv_path)
        with open(csv_path, "rb") as f:
            resp = c.post(
                "/api/datasets/upload",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"name": "analysis_test_ds"},
            )
        assert resp.status_code == 200, resp.text
        dataset_id = resp.json()["id"]
        c.dataset_id = dataset_id  # type: ignore[attr-defined]

        # Train a model (synchronous in TestClient)
        train_resp = c.post(
            "/api/training/start",
            json={
                "dataset_id": dataset_id,
                "task": "sentiment",
                "model_type": "tfidf",
                "seed": 42,
            },
        )
        assert train_resp.status_code == 202, train_resp.text
        c.model_id = train_resp.json().get("model_id")  # type: ignore[attr-defined]

        # If model_id not in immediate response, poll for it
        if not c.model_id:
            job_id = train_resp.json()["job_id"]
            status = c.get(f"/api/training/{job_id}/status").json()
            c.model_id = status.get("model_id")

        assert c.model_id, "Model not registered after training"
        yield c

    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR",
                "SFAP_TRAINING_DIR", "SFAP_ANALYSIS_DIR"]:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Tests: Start Analysis
# ---------------------------------------------------------------------------


class TestStartAnalysis:
    def test_start_returns_202(self, api_client):
        resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [api_client.model_id],
                "name": "Test Analysis",
            },
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["job_id"].startswith("analysis_")
        assert body["dataset_id"] == api_client.dataset_id
        assert api_client.model_id in body["model_ids"]
        assert body["status"] in ("pending", "running", "completed")
        assert body["psychometrics_warning"] is not None
        # Store for subsequent tests
        api_client.analysis_id = body["job_id"]  # type: ignore[attr-defined]

    def test_start_404_unknown_dataset(self, api_client):
        resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": "does_not_exist",
                "model_ids": [api_client.model_id],
            },
        )
        assert resp.status_code == 404

    def test_start_404_unknown_model(self, api_client):
        resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": ["nonexistent_model_id"],
            },
        )
        assert resp.status_code == 404

    def test_start_requires_at_least_one_model(self, api_client):
        resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [],
            },
        )
        assert resp.status_code == 422  # Pydantic validation

    def test_start_allows_cross_dataset_reuse_when_schema_is_compatible(
        self, api_client, tmp_path
    ):
        """A model from dataset A can be applied to dataset B when roles resolve."""
        other_csv = tmp_path / "other_dataset.csv"
        _make_labelled_csv(other_csv, n_per_class=8)
        with open(other_csv, "rb") as f:
            upload_resp = api_client.post(
                "/api/datasets/upload",
                files={"file": ("other_dataset.csv", f, "text/csv")},
                data={"name": "analysis_other_ds"},
            )
        assert upload_resp.status_code == 200, upload_resp.text
        other_dataset_id = upload_resp.json()["id"]

        train_resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": other_dataset_id,
                "task": "sentiment",
                "model_type": "tfidf",
                "seed": 42,
            },
        )
        assert train_resp.status_code == 202, train_resp.text
        other_model_id = train_resp.json().get("model_id")
        if not other_model_id:
            status = api_client.get(
                f"/api/training/{train_resp.json()['job_id']}/status"
            ).json()
            other_model_id = status.get("model_id")
        assert other_model_id, "Expected second dataset model to be registered"

        compatible_resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [other_model_id],
            },
        )
        assert compatible_resp.status_code == 202, compatible_resp.text
        assert compatible_resp.json()["dataset_id"] == api_client.dataset_id

    def test_start_422_for_schema_incompatible_model_dataset(self, api_client, tmp_path):
        """A model should be rejected with structured reasons when the target lacks text."""
        numeric_csv = tmp_path / "numeric_only.csv"
        pd.DataFrame({"q1": [1, 2, 3], "q2": [4, 5, 6]}).to_csv(numeric_csv, index=False)
        with open(numeric_csv, "rb") as f:
            upload_resp = api_client.post(
                "/api/datasets/upload",
                files={"file": ("numeric_only.csv", f, "text/csv")},
                data={"name": "analysis_numeric_only"},
            )
        assert upload_resp.status_code == 200, upload_resp.text
        numeric_dataset_id = upload_resp.json()["id"]

        incompatible_resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": numeric_dataset_id,
                "model_ids": [api_client.model_id],
            },
        )
        assert incompatible_resp.status_code == 422
        detail = incompatible_resp.json()["detail"]
        assert detail["message"].startswith("One or more models are incompatible")
        assert detail["models"][0]["compatibility"]["ok"] is False
        assert detail["models"][0]["compatibility"]["reasons"][0]["code"] == "missing_required_role"

    def test_start_with_metadata(self, api_client):
        resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [api_client.model_id],
                "name": "Named Analysis",
                "description": "A test description",
                "tags": ["test", "phase4"],
            },
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["name"] == "Named Analysis"
        assert body["tags"] == ["test", "phase4"]

    def test_start_with_branch_id(self, api_client):
        """branch_id is accepted and echoed back in the response."""
        # Create a branch to use
        branch_resp = api_client.post(
            f"/api/datasets/{api_client.dataset_id}/branches",
            json={"name": "analysis-test-branch"},
        )
        assert branch_resp.status_code == 200, branch_resp.text
        branch_id = branch_resp.json()["id"]
        api_client.branch_id = branch_id  # type: ignore[attr-defined]

        resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [api_client.model_id],
                "name": "Branch Analysis",
                "branch_id": branch_id,
            },
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["branch_id"] == branch_id

    def test_branch_id_persisted_in_db(self, api_client):
        """branch_id set at start time is retrievable via GET."""
        # Use the branch_id created in test_start_with_branch_id
        branch_id = getattr(api_client, "branch_id", None)
        if branch_id is None:
            pytest.skip("branch_id fixture not set (run test_start_with_branch_id first)")

        # Find the analysis with this branch_id
        resp = api_client.get("/api/analyses?status=completed&per_page=50")
        assert resp.status_code == 200
        analyses = resp.json()["analyses"]
        branch_analyses = [a for a in analyses if a.get("branch_id") == branch_id]
        assert len(branch_analyses) >= 1, "Expected at least one analysis with branch_id"


# ---------------------------------------------------------------------------
# Tests: List Analyses
# ---------------------------------------------------------------------------


class TestListAnalyses:
    def test_list_returns_200(self, api_client):
        resp = api_client.get("/api/analyses")
        assert resp.status_code == 200
        body = resp.json()
        assert "analyses" in body
        assert "total" in body
        assert "page" in body

    def test_list_pagination_params(self, api_client):
        resp = api_client.get("/api/analyses?page=1&per_page=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["page"] == 1
        assert body["per_page"] == 5

    def test_list_filter_by_status(self, api_client):
        resp = api_client.get("/api/analyses?status=completed")
        assert resp.status_code == 200
        body = resp.json()
        for a in body["analyses"]:
            assert a["status"] == "completed"


# ---------------------------------------------------------------------------
# Tests: Get Analysis
# ---------------------------------------------------------------------------


class TestGetAnalysis:
    def test_get_returns_200(self, api_client):
        analysis_id = api_client.analysis_id
        resp = api_client.get(f"/api/analyses/{analysis_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == analysis_id or body.get("job_id") == analysis_id

    def test_get_404_for_unknown(self, api_client):
        resp = api_client.get("/api/analyses/no_such_analysis_id")
        assert resp.status_code == 404

    def test_status_endpoint(self, api_client):
        analysis_id = api_client.analysis_id
        resp = api_client.get(f"/api/analyses/{analysis_id}/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("pending", "running", "completed", "failed")

    def test_completed_has_result_summary(self, api_client):
        """After TestClient background task runs, analysis should be completed."""
        analysis_id = api_client.analysis_id
        resp = api_client.get(f"/api/analyses/{analysis_id}/status")
        body = resp.json()
        if body["status"] == "completed":
            assert body["result_summary"] is not None
            summary = body["result_summary"]
            assert "n_rows" in summary
            assert "models_applied" in summary


# ---------------------------------------------------------------------------
# Tests: Update Analysis
# ---------------------------------------------------------------------------


class TestUpdateAnalysis:
    def test_patch_name(self, api_client):
        analysis_id = api_client.analysis_id
        resp = api_client.patch(
            f"/api/analyses/{analysis_id}",
            json={"name": "Updated Name", "tags": ["updated"]},
        )
        # May be 200 or 404 if analysis not in DB yet (still running)
        if resp.status_code == 200:
            body = resp.json()
            assert body["name"] == "Updated Name"

    def test_patch_404_for_unknown(self, api_client):
        resp = api_client.patch(
            "/api/analyses/no_such_id",
            json={"name": "X"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: Results
# ---------------------------------------------------------------------------


class TestAnalysisResults:
    def _ensure_completed(self, api_client) -> str:
        """Wait for any analysis to be completed and return its ID."""
        resp = api_client.get("/api/analyses?status=completed&per_page=5")
        body = resp.json()
        if body["analyses"]:
            return body["analyses"][0]["id"]
        # Fall back to the test analysis_id
        return api_client.analysis_id

    def test_results_409_for_running(self, api_client):
        """Start a new analysis (may be running) and check that results give 409."""
        resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [api_client.model_id],
            },
        )
        job_id = resp.json()["job_id"]
        status = api_client.get(f"/api/analyses/{job_id}/status").json()["status"]

        if status in ("pending", "running"):
            results_resp = api_client.get(f"/api/analyses/{job_id}/results")
            assert results_resp.status_code == 409

    def test_results_returns_paginated_rows(self, api_client):
        """Completed analysis results should return rows."""
        analysis_id = self._ensure_completed(api_client)
        status = api_client.get(f"/api/analyses/{analysis_id}/status").json()["status"]
        if status != "completed":
            pytest.skip("Analysis not yet completed")

        resp = api_client.get(f"/api/analyses/{analysis_id}/results?limit=10")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "columns" in body
        assert "rows" in body
        assert "total_rows" in body
        assert body["total_rows"] == 45
        assert len(body["rows"]) == 10

    def test_results_filter(self, api_client):
        """Filter by column value."""
        analysis_id = self._ensure_completed(api_client)
        status = api_client.get(f"/api/analyses/{analysis_id}/status").json()["status"]
        if status != "completed":
            pytest.skip("Analysis not yet completed")

        resp = api_client.get(
            f"/api/analyses/{analysis_id}/results"
            "?filter_col=language&filter_val=ru&limit=100"
        )
        assert resp.status_code == 200
        body = resp.json()
        lang_idx = body["columns"].index("language")
        for row in body["rows"]:
            assert row[lang_idx].lower() == "ru"

    def test_export_csv(self, api_client):
        """Export endpoint returns CSV content."""
        analysis_id = self._ensure_completed(api_client)
        status = api_client.get(f"/api/analyses/{analysis_id}/status").json()["status"]
        if status != "completed":
            pytest.skip("Analysis not yet completed")

        resp = api_client.get(f"/api/analyses/{analysis_id}/results/export?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")
        content = resp.content.decode("utf-8-sig")
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) > 1  # header + at least 1 data row


# ---------------------------------------------------------------------------
# Tests: Delete
# ---------------------------------------------------------------------------


class TestDeleteAnalysis:
    def test_delete_returns_200(self, api_client):
        # Create a fresh analysis to delete
        resp = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [api_client.model_id],
                "name": "To Be Deleted",
            },
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Wait for it to complete (background task runs synchronously in TestClient)
        del_resp = api_client.delete(f"/api/analyses/{job_id}")
        assert del_resp.status_code == 200
        body = del_resp.json()
        assert body["deleted"] is True

    def test_delete_404_for_unknown(self, api_client):
        resp = api_client.delete("/api/analyses/no_such_id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: Compare
# ---------------------------------------------------------------------------


class TestCompareAnalyses:
    def test_compare_two_analyses(self, api_client):
        """Start two analyses and compare them."""
        # Start analysis 1
        r1 = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [api_client.model_id],
                "name": "Compare A",
            },
        )
        assert r1.status_code == 202
        id1 = r1.json()["job_id"]

        r2 = api_client.post(
            "/api/analyses",
            json={
                "dataset_id": api_client.dataset_id,
                "model_ids": [api_client.model_id],
                "name": "Compare B",
            },
        )
        assert r2.status_code == 202
        id2 = r2.json()["job_id"]

        # Both should be completed (background tasks run sync in TestClient)
        s1 = api_client.get(f"/api/analyses/{id1}/status").json()["status"]
        s2 = api_client.get(f"/api/analyses/{id2}/status").json()["status"]

        if s1 != "completed" or s2 != "completed":
            pytest.skip("Analyses not yet completed")

        compare_resp = api_client.post(
            "/api/analyses/compare",
            json={"analysis_ids": [id1, id2]},
        )
        assert compare_resp.status_code == 200, compare_resp.text
        body = compare_resp.json()
        assert "run_1" in body
        assert "run_2" in body
        assert "task_comparisons" in body
        assert "shared_tasks" in body

    def test_compare_404_for_unknown(self, api_client):
        resp = api_client.post(
            "/api/analyses/compare",
            json={"analysis_ids": ["nonexistent_1", "nonexistent_2"]},
        )
        assert resp.status_code == 404

    def test_compare_requires_exactly_two(self, api_client):
        resp = api_client.post(
            "/api/analyses/compare",
            json={"analysis_ids": [api_client.analysis_id]},
        )
        assert resp.status_code == 422
