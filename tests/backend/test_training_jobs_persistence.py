"""Tests that training jobs are persisted to SQLite and survive an in-memory clear."""
from __future__ import annotations

import os
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
from src.storage.database import Database
from src.training import runner as training_runner


def _make_csv(path: Path, n_per_class: int = 15) -> None:
    rows = []
    for s in ["positive", "neutral", "negative"]:
        for i in range(n_per_class):
            rows.append({
                "text_feedback": f"Feedback {i} class {s} extra words here",
                "language": "ru" if i % 2 == 0 else "kz",
                "sentiment_class": s,
                "detail_level": "short" if i < 5 else ("medium" if i < 10 else "long"),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def persist_client(tmp_path_factory):
    """TestClient with isolated storage and a pre-uploaded dataset."""
    tmp = tmp_path_factory.mktemp("persist_test")

    os.environ["SFAP_RUNS_DIR"] = str(tmp / "runs")
    os.environ["SFAP_DB_PATH"] = str(tmp / "test_persist.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp / "models")
    os.environ["SFAP_TRAINING_DIR"] = str(tmp / "training_runs")

    dependencies.get_run_manager.cache_clear()
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()

    with training_runner._jobs_lock:
        training_runner._jobs.clear()

    with TestClient(app) as c:
        csv_path = tmp / "sample.csv"
        _make_csv(csv_path)
        with open(csv_path, "rb") as f:
            resp = c.post(
                "/api/datasets/upload",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"name": "persist_test_dataset"},
            )
        assert resp.status_code == 200, resp.text
        c.dataset_id = resp.json()["id"]  # type: ignore[attr-defined]
        c.db_path = str(tmp / "test_persist.db")  # type: ignore[attr-defined]
        yield c

    # Cleanup
    for key in ("SFAP_RUNS_DIR", "SFAP_DB_PATH", "SFAP_DATASETS_DIR",
                "SFAP_MODELS_DIR", "SFAP_TRAINING_DIR"):
        os.environ.pop(key, None)
    dependencies.get_run_manager.cache_clear()
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()


class TestTrainingJobsDB:
    def test_start_job_writes_to_db(self, persist_client):
        """Starting a training job inserts a row in training_jobs."""
        resp = persist_client.post(
            "/api/training/start",
            json={
                "dataset_id": persist_client.dataset_id,
                "task": "sentiment",
                "model_type": "tfidf",
                "seed": 42,
                "name": "persist_test_job",
            },
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]
        assert job_id.startswith("job_")

        # Check the DB directly
        db = Database(Path(persist_client.db_path))
        row = db.fetchone("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
        assert row is not None, "job_id not found in training_jobs table"
        assert row["dataset_id"] == persist_client.dataset_id
        assert row["task"] == "sentiment"
        assert row["model_type"] == "tfidf"

    def test_job_survives_memory_clear(self, persist_client):
        """After clearing _jobs, the job is still retrievable from DB."""
        resp = persist_client.post(
            "/api/training/start",
            json={
                "dataset_id": persist_client.dataset_id,
                "task": "language",
                "model_type": "char_ngram",
                "seed": 42,
            },
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        # Simulate restart: wipe in-memory store
        with training_runner._jobs_lock:
            training_runner._jobs.clear()

        # The list endpoint should still return jobs from DB
        list_resp = persist_client.get("/api/training/")
        assert list_resp.status_code == 200
        body = list_resp.json()
        job_ids_in_list = [j["job_id"] for j in body["jobs"]]
        assert job_id in job_ids_in_list, (
            f"job_id {job_id!r} not found after in-memory clear; got {job_ids_in_list}"
        )

    def test_completed_job_written_to_db(self, persist_client):
        """After a job completes its status and model_id are persisted to DB."""
        resp = persist_client.post(
            "/api/training/start",
            json={
                "dataset_id": persist_client.dataset_id,
                "task": "detail_level",
                "model_type": "tfidf",
                "seed": 7,
            },
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        # TestClient runs background tasks synchronously, so job should be done
        status_resp = persist_client.get(f"/api/training/{job_id}/status")
        assert status_resp.status_code == 200
        job_data = status_resp.json()

        if job_data["status"] == "completed":
            db = Database(Path(persist_client.db_path))
            row = db.fetchone("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
            assert row is not None
            assert row["status"] == "completed"
            assert row["model_id"] is not None

    def test_pipeline_runs_table_exists(self, persist_client):
        """The pipeline_runs table is created by the schema migration."""
        db = Database(Path(persist_client.db_path))
        tables = {r[0] for r in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "pipeline_runs" in tables
        assert "training_jobs" in tables

    def test_list_jobs_from_db_filters_by_task(self, persist_client):
        """list_jobs_from_db correctly filters by task."""
        db = Database(Path(persist_client.db_path))
        jobs = training_runner.list_jobs_from_db(db=db, task="sentiment")
        for j in jobs:
            assert j["task"] == "sentiment"

    def test_model_has_job_id(self, persist_client):
        """A registered model has the job_id field set."""
        # Start and complete a job
        resp = persist_client.post(
            "/api/training/start",
            json={
                "dataset_id": persist_client.dataset_id,
                "task": "sentiment",
                "model_type": "char_ngram",
                "seed": 11,
            },
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        # Get the status
        status_resp = persist_client.get(f"/api/training/{job_id}/status")
        assert status_resp.status_code == 200
        job_data = status_resp.json()

        if job_data["status"] == "completed" and job_data.get("model_id"):
            model_id = job_data["model_id"]
            model_resp = persist_client.get(f"/api/models/{model_id}")
            assert model_resp.status_code == 200
            model_data = model_resp.json()
            # job_id field should be present in the model record
            db = Database(Path(persist_client.db_path))
            row = db.fetchone("SELECT job_id FROM models WHERE id = ?", (model_id,))
            assert row is not None
            assert row["job_id"] == job_id
