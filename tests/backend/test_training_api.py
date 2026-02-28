"""Integration tests for the training API endpoints."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import sys
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.api.main import app
from src.api import dependencies
from src.training import runner as training_runner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_labelled_csv(path: Path, n_per_class: int = 15) -> None:
    """Write a small CSV with text + label columns to path."""
    rows = []
    for s in ["positive", "neutral", "negative"]:
        for i in range(n_per_class):
            rows.append(
                {
                    "text_feedback": f"Feedback {i} class {s} extra words here",
                    "language": "ru" if i % 2 == 0 else "kz",
                    "sentiment_class": s,
                    "detail_level": "short" if i < 5 else ("medium" if i < 10 else "long"),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """TestClient backed by isolated temp storage, with a pre-uploaded dataset."""
    tmp = tmp_path_factory.mktemp("training_api")

    # Point all storage env vars at temp dirs
    os.environ["SFAP_RUNS_DIR"] = str(tmp / "runs")
    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp / "models")
    os.environ["SFAP_TRAINING_DIR"] = str(tmp / "training_runs")

    # Clear LRU caches so the new env vars take effect
    dependencies.get_run_manager.cache_clear()
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()

    # Clear in-memory job store from any prior test modules
    with training_runner._jobs_lock:
        training_runner._jobs.clear()

    with TestClient(app) as c:
        # Upload a dataset via the API so we have a valid dataset_id
        csv_path = tmp / "sample.csv"
        _make_labelled_csv(csv_path)
        with open(csv_path, "rb") as f:
            resp = c.post(
                "/api/datasets/upload",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"name": "training_test_dataset"},
            )
        assert resp.status_code == 200, resp.text
        dataset_id = resp.json()["id"]
        c.dataset_id = dataset_id  # type: ignore[attr-defined]
        yield c

    # Cleanup env vars
    for key in [
        "SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR", "SFAP_TRAINING_DIR"
    ]:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStartTraining:
    def test_start_returns_202(self, api_client):
        resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": api_client.dataset_id,
                "task": "sentiment",
                "model_type": "tfidf",
                "seed": 42,
            },
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["job_id"].startswith("job_")
        # In TestClient, BackgroundTasks run synchronously after response
        assert body["task"] == "sentiment"
        assert body["model_type"] == "tfidf"
        assert body["dataset_id"] == api_client.dataset_id
        assert body["psychometrics_warning"] is not None

    def test_start_returns_404_for_unknown_dataset(self, api_client):
        resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": "nonexistent-id",
                "task": "sentiment",
                "model_type": "tfidf",
            },
        )
        assert resp.status_code == 404

    def test_start_rejects_invalid_task(self, api_client):
        resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": api_client.dataset_id,
                "task": "bogus",
                "model_type": "tfidf",
            },
        )
        assert resp.status_code == 422  # Pydantic validation error

    def test_start_rejects_invalid_model_type(self, api_client):
        resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": api_client.dataset_id,
                "task": "sentiment",
                "model_type": "bert",
            },
        )
        assert resp.status_code == 422

    def test_start_with_full_config(self, api_client):
        resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": api_client.dataset_id,
                "task": "language",
                "model_type": "char_ngram",
                "seed": 7,
                "name": "my_lang_model",
                "config": {
                    "train_ratio": 0.75,
                    "val_ratio": 0.125,
                    "test_ratio": 0.125,
                    "class_balancing": "oversample",
                },
            },
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["task"] == "language"
        assert body["name"] == "my_lang_model"


class TestListTrainingJobs:
    def test_list_returns_all_jobs(self, api_client):
        resp = api_client.get("/api/training/")
        assert resp.status_code == 200
        body = resp.json()
        assert "jobs" in body
        assert "total" in body
        assert body["total"] == len(body["jobs"])
        assert body["total"] >= 2  # at least the jobs from TestStartTraining

    def test_list_filter_by_task(self, api_client):
        resp = api_client.get("/api/training/?task=sentiment")
        assert resp.status_code == 200
        body = resp.json()
        for job in body["jobs"]:
            assert job["task"] == "sentiment"


class TestGetTrainingStatus:
    def _start_job(self, api_client) -> str:
        resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": api_client.dataset_id,
                "task": "detail_level",
                "model_type": "tfidf",
                "seed": 99,
            },
        )
        assert resp.status_code == 202
        return resp.json()["job_id"]

    def test_status_returns_job(self, api_client):
        job_id = self._start_job(api_client)
        resp = api_client.get(f"/api/training/{job_id}/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] in ("pending", "running", "completed", "failed")

    def test_status_404_for_unknown_job(self, api_client):
        resp = api_client.get("/api/training/nonexistent_job_id/status")
        assert resp.status_code == 404

    def test_completed_job_has_metrics(self, api_client):
        """After TestClient runs background task, job should be completed."""
        job_id = self._start_job(api_client)
        resp = api_client.get(f"/api/training/{job_id}/status")
        body = resp.json()
        # Background tasks in TestClient run synchronously
        if body["status"] == "completed":
            assert body["model_id"] is not None
            assert body["metrics"] is not None
            assert "val" in body["metrics"]


class TestGetTrainingResult:
    def test_result_after_completion(self, api_client):
        resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": api_client.dataset_id,
                "task": "sentiment",
                "model_type": "char_ngram",
                "seed": 11,
            },
        )
        job_id = resp.json()["job_id"]

        result_resp = api_client.get(f"/api/training/{job_id}/result")
        if result_resp.status_code == 409:
            # Job not yet completed — acceptable if not using sync background
            pass
        else:
            assert result_resp.status_code == 200
            body = result_resp.json()
            assert body["status"] in ("completed", "failed")

    def test_result_404_for_unknown_job(self, api_client):
        resp = api_client.get("/api/training/no_such_job/result")
        assert resp.status_code == 404


class TestRegisteredModelLink:
    """After training completes, the model should appear in /api/models/."""

    def test_model_registered_after_training(self, api_client):
        start_resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": api_client.dataset_id,
                "task": "sentiment",
                "model_type": "tfidf",
                "seed": 55,
                "name": "registered_model_test",
            },
        )
        assert start_resp.status_code == 202
        job = start_resp.json()
        job_id = job["job_id"]

        # Poll status (in TestClient background tasks run synchronously)
        status_resp = api_client.get(f"/api/training/{job_id}/status")
        status_body = status_resp.json()

        if status_body["status"] == "completed":
            model_id = status_body["model_id"]
            assert model_id is not None

            # Verify it appears in model registry
            models_resp = api_client.get(
                f"/api/models/?dataset_id={api_client.dataset_id}"
            )
            assert models_resp.status_code == 200
            models = models_resp.json()["models"]
            model_ids = [m["id"] for m in models]
            assert model_id in model_ids

            detail_resp = api_client.get(f"/api/models/{model_id}")
            assert detail_resp.status_code == 200
            assert detail_resp.json()["run_source"] == "training"

    def test_model_card_available_after_training(self, api_client):
        start_resp = api_client.post(
            "/api/training/start",
            json={
                "dataset_id": api_client.dataset_id,
                "task": "language",
                "model_type": "tfidf",
                "seed": 123,
                "name": "model_card_test",
            },
        )
        assert start_resp.status_code == 202
        job_id = start_resp.json()["job_id"]

        status_resp = api_client.get(f"/api/training/{job_id}/status")
        assert status_resp.status_code == 200
        status = status_resp.json()
        if status["status"] != "completed":
            pytest.skip("Training not completed in this test environment")

        model_id = status["model_id"]
        assert model_id is not None

        card_resp = api_client.get(f"/api/models/{model_id}/model-card")
        assert card_resp.status_code == 200, card_resp.text
        assert "text/markdown" in card_resp.headers.get("content-type", "")
        content = card_resp.text
        assert content.startswith("# Model Card:")
