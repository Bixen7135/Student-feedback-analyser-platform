"""Tests for dataset API endpoints."""
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
    """TestClient with temporary storage directories."""
    tmp = tmp_path_factory.mktemp("dataset_api")
    os.environ["SFAP_RUNS_DIR"] = str(tmp / "runs")
    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    # Clear all lru_cache
    dependencies.get_run_manager.cache_clear()
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    with TestClient(app) as c:
        yield c
    # Cleanup env
    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR"]:
        os.environ.pop(key, None)


def _make_csv_bytes(rows: int = 10) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "score"])
    for i in range(rows):
        writer.writerow([i, f"student_{i}", i * 10])
    return output.getvalue().encode("utf-8")


def test_list_datasets_empty(client):
    resp = client.get("/api/datasets/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["datasets"] == []


def test_upload_dataset(client):
    csv_data = _make_csv_bytes()
    resp = client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", csv_data, "text/csv")},
        data={"name": "Test Dataset", "description": "A test", "author": "tester"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test Dataset"
    assert data["row_count"] == 10
    assert data["current_version"] == 1
    assert len(data["schema_info"]) == 3


def test_upload_rejects_non_csv(client):
    resp = client.post(
        "/api/datasets/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"name": "Bad File"},
    )
    assert resp.status_code == 400


def test_list_after_upload(client):
    resp = client.get("/api/datasets/")
    data = resp.json()
    # At least 1 from previous test_upload_dataset
    assert data["total"] >= 1


def test_get_dataset_detail(client):
    csv_data = _make_csv_bytes()
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("detail.csv", csv_data, "text/csv")},
        data={"name": "Detail DS"},
    )
    ds_id = upload.json()["id"]

    resp = client.get(f"/api/datasets/{ds_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Detail DS"
    assert data["row_count"] == 10


def test_get_dataset_not_found(client):
    resp = client.get("/api/datasets/nonexistent")
    assert resp.status_code == 404


def test_get_dataset_preview(client):
    csv_data = _make_csv_bytes(20)
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("preview.csv", csv_data, "text/csv")},
        data={"name": "Preview DS"},
    )
    ds_id = upload.json()["id"]

    resp = client.get(f"/api/datasets/{ds_id}/preview?offset=0&limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_rows"] == 20
    assert len(data["rows"]) == 5
    assert data["columns"] == ["id", "name", "score"]


def test_get_dataset_schema(client):
    csv_data = _make_csv_bytes()
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("schema.csv", csv_data, "text/csv")},
        data={"name": "Schema DS"},
    )
    ds_id = upload.json()["id"]

    resp = client.get(f"/api/datasets/{ds_id}/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["columns"]) == 3


def test_update_metadata(client):
    csv_data = _make_csv_bytes()
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("update.csv", csv_data, "text/csv")},
        data={"name": "Original Name"},
    )
    ds_id = upload.json()["id"]

    resp = client.patch(
        f"/api/datasets/{ds_id}",
        json={"name": "Renamed", "tags": ["updated"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Renamed"
    assert data["tags"] == ["updated"]


def test_delete_dataset(client):
    csv_data = _make_csv_bytes()
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("delete.csv", csv_data, "text/csv")},
        data={"name": "Delete Me"},
    )
    ds_id = upload.json()["id"]

    resp = client.delete(f"/api/datasets/{ds_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True


def test_get_dataset_versions(client):
    csv_data = _make_csv_bytes()
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("version.csv", csv_data, "text/csv")},
        data={"name": "Versioned DS"},
    )
    ds_id = upload.json()["id"]

    resp = client.get(f"/api/datasets/{ds_id}/versions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["version"] == 1


def test_create_subset(client):
    csv_data = _make_csv_bytes(10)
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("source.csv", csv_data, "text/csv")},
        data={"name": "Source DS"},
    )
    ds_id = upload.json()["id"]

    resp = client.post(
        f"/api/datasets/{ds_id}/subset",
        json={
            "name": "Subset DS",
            "filter_config": {"column_in": {"name": ["student_0", "student_1"]}},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["row_count"] == 2
    assert "subset" in data["tags"]


def test_search_datasets(client):
    csv_data = _make_csv_bytes()
    client.post(
        "/api/datasets/upload",
        files={"file": ("alpha.csv", csv_data, "text/csv")},
        data={"name": "Searchable Alpha"},
    )

    resp = client.get("/api/datasets/?search=Searchable+Alpha")
    data = resp.json()
    assert data["total"] >= 1
    names = [d["name"] for d in data["datasets"]]
    assert "Searchable Alpha" in names
