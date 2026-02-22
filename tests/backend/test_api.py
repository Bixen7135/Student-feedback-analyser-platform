"""API integration tests using FastAPI TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Import and configure the app
from src.api.main import app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """TestClient with a temporary runs directory."""
    import os
    runs_dir = tmp_path_factory.mktemp("runs")
    os.environ["SFAP_RUNS_DIR"] = str(runs_dir)
    # Clear lru_cache to pick up new env var
    from src.api import dependencies
    dependencies.get_run_manager.cache_clear()
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_runs_empty(client):
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_create_run(client, tiny_dataset_path):
    resp = client.post("/api/runs", json={"seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["random_seed"] == 42


def test_get_run(client, tiny_dataset_path):
    # Create a run first
    create_resp = client.post("/api/runs", json={"seed": 42})
    run_id = create_resp.json()["run_id"]

    # Get it
    resp = client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == run_id


def test_get_run_not_found(client):
    resp = client.get("/api/runs/nonexistent_run_id")
    assert resp.status_code == 404


def test_list_runs_after_create(client):
    client.post("/api/runs", json={"seed": 1})
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_psychometrics_not_found_before_stage(client):
    create_resp = client.post("/api/runs", json={"seed": 42})
    run_id = create_resp.json()["run_id"]
    resp = client.get(f"/api/runs/{run_id}/metrics/psychometrics")
    assert resp.status_code == 404


def test_artifacts_empty_for_new_run(client):
    create_resp = client.post("/api/runs", json={"seed": 42})
    run_id = create_resp.json()["run_id"]
    resp = client.get(f"/api/runs/{run_id}/artifacts")
    assert resp.status_code == 200
    assert resp.json()["artifacts"] == []


# ---------------------------------------------------------------------------
# Phase 1: dataset_id / branch_id / dataset_version / name fields
# ---------------------------------------------------------------------------

def test_create_run_response_includes_new_fields(client):
    """POST /api/runs response includes all Phase 1 fields (null when not supplied)."""
    resp = client.post("/api/runs", json={"seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert "dataset_id" in data
    assert "branch_id" in data
    assert "dataset_version" in data
    assert "name" in data
    assert data["dataset_id"] is None
    assert data["branch_id"] is None
    assert data["dataset_version"] is None
    assert data["name"] is None


def test_create_run_with_name(client):
    """name is stored and returned in the run summary."""
    resp = client.post("/api/runs", json={"seed": 42, "name": "baseline-v1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "baseline-v1"


def test_get_run_includes_new_fields(client):
    """GET /api/runs/{run_id} also exposes Phase 1 fields."""
    create_resp = client.post("/api/runs", json={"seed": 42, "name": "test-run"})
    run_id = create_resp.json()["run_id"]

    resp = client.get(f"/api/runs/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "test-run"
    assert data["dataset_id"] is None


def test_create_run_unknown_dataset_id_returns_404(client):
    """POST /api/runs with non-existent dataset_id returns 404."""
    resp = client.post(
        "/api/runs",
        json={"seed": 42, "dataset_id": "does-not-exist-xyz"},
    )
    assert resp.status_code == 404
