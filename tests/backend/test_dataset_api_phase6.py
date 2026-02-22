"""Tests for Phase 6 dataset API endpoints: /empty, /cells, /rows."""
from __future__ import annotations

import csv
import io
import os

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import dependencies


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dataset_api_p6")
    os.environ["SFAP_RUNS_DIR"] = str(tmp / "runs")
    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    dependencies.get_run_manager.cache_clear()
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    with TestClient(app) as c:
        yield c
    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR"]:
        os.environ.pop(key, None)


def _upload(client, rows: int = 5) -> str:
    """Upload a small dataset and return its ID."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "score"])
    for i in range(rows):
        writer.writerow([i, f"student_{i}", i * 10])
    csv_bytes = output.getvalue().encode()
    resp = client.post(
        "/api/datasets/upload",
        files={"file": ("data.csv", csv_bytes, "text/csv")},
        data={"name": f"DS_{rows}"},
    )
    assert resp.status_code == 200
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# POST /api/datasets/empty
# ---------------------------------------------------------------------------

def test_create_empty_returns_dataset(client):
    resp = client.post("/api/datasets/empty", json={
        "name": "My Empty",
        "columns": ["col_a", "col_b", "col_c"],
        "description": "test empty",
        "author": "tester",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "My Empty"
    assert data["row_count"] == 0
    assert data["current_version"] == 1
    assert len(data["schema_info"]) == 3


def test_create_empty_rejects_single_column(client):
    resp = client.post("/api/datasets/empty", json={"name": "Bad", "columns": ["only_one"]})
    assert resp.status_code == 400


def test_create_empty_rejects_duplicate_columns(client):
    resp = client.post("/api/datasets/empty", json={"name": "Dup", "columns": ["a", "a"]})
    assert resp.status_code == 400


def test_create_empty_appears_in_list(client):
    client.post("/api/datasets/empty", json={"name": "ListTest", "columns": ["x", "y"]})
    resp = client.get("/api/datasets/")
    assert resp.status_code == 200
    names = [d["name"] for d in resp.json()["datasets"]]
    assert "ListTest" in names


# ---------------------------------------------------------------------------
# PATCH /api/datasets/{id}/cells
# ---------------------------------------------------------------------------

def test_patch_cells_returns_new_version(client):
    ds_id = _upload(client)
    resp = client.patch(f"/api/datasets/{ds_id}/cells", json={
        "changes": [{"row_idx": 0, "col": "name", "value": "PATCHED"}],
        "reason": "fix typo",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == 2
    assert data["reason"] == "fix typo"


def test_patch_cells_bad_col_returns_400(client):
    ds_id = _upload(client)
    resp = client.patch(f"/api/datasets/{ds_id}/cells", json={
        "changes": [{"row_idx": 0, "col": "NONEXISTENT", "value": "x"}],
    })
    assert resp.status_code == 400


def test_patch_cells_bad_row_returns_400(client):
    ds_id = _upload(client, rows=3)
    resp = client.patch(f"/api/datasets/{ds_id}/cells", json={
        "changes": [{"row_idx": 999, "col": "name", "value": "x"}],
    })
    assert resp.status_code == 400


def test_patch_cells_increments_version_count(client):
    ds_id = _upload(client)
    client.patch(f"/api/datasets/{ds_id}/cells", json={
        "changes": [{"row_idx": 0, "col": "score", "value": "99"}],
    })
    client.patch(f"/api/datasets/{ds_id}/cells", json={
        "changes": [{"row_idx": 1, "col": "score", "value": "88"}],
    })
    resp = client.get(f"/api/datasets/{ds_id}/versions")
    assert resp.status_code == 200
    assert len(resp.json()) == 3  # v1 + v2 + v3


# ---------------------------------------------------------------------------
# POST /api/datasets/{id}/rows
# ---------------------------------------------------------------------------

def test_post_rows_returns_new_version(client):
    ds_id = _upload(client, rows=3)
    resp = client.post(f"/api/datasets/{ds_id}/rows", json={
        "rows": [{"id": "99", "name": "extra", "score": "100"}],
        "reason": "add student",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == 2
    assert data["row_count"] == 4


def test_post_rows_preview_shows_new_row(client):
    ds_id = _upload(client, rows=2)
    client.post(f"/api/datasets/{ds_id}/rows", json={
        "rows": [{"id": "X", "name": "New", "score": "50"}],
    })
    resp = client.get(f"/api/datasets/{ds_id}/preview?limit=10")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 3
    # Last row should have "New" in name column
    assert any("New" in str(cell) for cell in rows[-1])


# ---------------------------------------------------------------------------
# DELETE /api/datasets/{id}/rows
# ---------------------------------------------------------------------------

def test_delete_rows_returns_new_version(client):
    ds_id = _upload(client, rows=5)
    resp = client.request(
        "DELETE",
        f"/api/datasets/{ds_id}/rows",
        json={"row_indices": [0, 2], "reason": "remove bad rows"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == 2
    assert data["row_count"] == 3


def test_delete_rows_bad_index_returns_400(client):
    ds_id = _upload(client, rows=3)
    resp = client.request(
        "DELETE",
        f"/api/datasets/{ds_id}/rows",
        json={"row_indices": [999]},
    )
    assert resp.status_code == 400


def test_delete_all_rows_returns_400(client):
    ds_id = _upload(client, rows=2)
    resp = client.request(
        "DELETE",
        f"/api/datasets/{ds_id}/rows",
        json={"row_indices": [0, 1]},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Version history after edits
# ---------------------------------------------------------------------------

def test_versions_list_grows_after_edits(client):
    ds_id = _upload(client, rows=4)
    # v1 already created; add v2, v3
    client.patch(f"/api/datasets/{ds_id}/cells", json={
        "changes": [{"row_idx": 0, "col": "name", "value": "A"}],
    })
    client.post(f"/api/datasets/{ds_id}/rows", json={
        "rows": [{"id": "Z", "name": "Z", "score": "0"}],
    })
    resp = client.get(f"/api/datasets/{ds_id}/versions")
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_old_version_preview_unchanged(client):
    ds_id = _upload(client, rows=3)
    # Edit row 0
    client.patch(f"/api/datasets/{ds_id}/cells", json={
        "changes": [{"row_idx": 0, "col": "name", "value": "CHANGED"}],
    })
    # v1 preview should still show original name
    resp = client.get(f"/api/datasets/{ds_id}/preview?version=1&limit=10")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert rows[0][1] == "student_0"  # column index 1 = "name"
