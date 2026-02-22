"""Phase 5 — tests for saved filter CRUD (unit + API)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ---------------------------------------------------------------------------
# Unit tests: saved_filters module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("saved_filters_unit")
    from src.storage.database import Database

    d = Database(tmp / "test.db")
    yield d
    d.close()


class TestSavedFiltersCRUD:
    def test_create_returns_record(self, db):
        from src.analysis.saved_filters import create_saved_filter

        record = create_saved_filter(
            db=db,
            name="My Filter",
            entity_type="analysis_results",
            filter_config={"filters": [{"col": "language", "op": "eq", "val": "ru"}]},
        )
        assert record["id"].startswith("sf_")
        assert record["name"] == "My Filter"
        assert record["entity_type"] == "analysis_results"
        assert isinstance(record["filter_config"], dict)
        assert "filters" in record["filter_config"]
        assert "created_at" in record

    def test_get_existing(self, db):
        from src.analysis.saved_filters import create_saved_filter, get_saved_filter

        created = create_saved_filter(db, "Get Test", "analysis_results", {"search": "hello"})
        fetched = get_saved_filter(db, created["id"])
        assert fetched is not None
        assert fetched["id"] == created["id"]
        assert fetched["filter_config"] == {"search": "hello"}

    def test_get_nonexistent_returns_none(self, db):
        from src.analysis.saved_filters import get_saved_filter

        assert get_saved_filter(db, "sf_nonexistent") is None

    def test_list_all(self, db):
        from src.analysis.saved_filters import list_saved_filters

        records = list_saved_filters(db)
        assert len(records) >= 2  # at least from previous tests
        assert all("id" in r for r in records)

    def test_list_filter_by_entity_type(self, db):
        from src.analysis.saved_filters import create_saved_filter, list_saved_filters

        create_saved_filter(db, "Dataset Filter", "datasets", {"search": "foo"})
        ds_records = list_saved_filters(db, entity_type="datasets")
        assert len(ds_records) >= 1
        for r in ds_records:
            assert r["entity_type"] == "datasets"

        ar_records = list_saved_filters(db, entity_type="analysis_results")
        for r in ar_records:
            assert r["entity_type"] == "analysis_results"

    def test_update_name(self, db):
        from src.analysis.saved_filters import create_saved_filter, update_saved_filter

        created = create_saved_filter(db, "Old Name", "analysis_results", {})
        updated = update_saved_filter(db, created["id"], name="New Name")
        assert updated is not None
        assert updated["name"] == "New Name"
        # filter_config unchanged
        assert updated["filter_config"] == {}

    def test_update_filter_config(self, db):
        from src.analysis.saved_filters import create_saved_filter, update_saved_filter

        created = create_saved_filter(db, "Config Test", "analysis_results", {"search": "old"})
        new_cfg = {"filters": [{"col": "lang", "op": "eq", "val": "ru"}], "search": "new"}
        updated = update_saved_filter(db, created["id"], filter_config=new_cfg)
        assert updated is not None
        assert updated["filter_config"]["search"] == "new"

    def test_update_nonexistent_returns_none(self, db):
        from src.analysis.saved_filters import update_saved_filter

        result = update_saved_filter(db, "sf_nonexistent", name="X")
        assert result is None

    def test_delete_existing(self, db):
        from src.analysis.saved_filters import create_saved_filter, delete_saved_filter, get_saved_filter

        created = create_saved_filter(db, "To Delete", "analysis_results", {})
        assert get_saved_filter(db, created["id"]) is not None

        result = delete_saved_filter(db, created["id"])
        assert result is True
        assert get_saved_filter(db, created["id"]) is None

    def test_delete_nonexistent_returns_false(self, db):
        from src.analysis.saved_filters import delete_saved_filter

        assert delete_saved_filter(db, "sf_nonexistent") is False

    def test_filter_config_is_deserialized(self, db):
        """filter_config should be a dict, not a raw JSON string."""
        from src.analysis.saved_filters import create_saved_filter, get_saved_filter

        cfg = {"filters": [{"col": "a", "op": "contains", "val": "x"}], "search": "test"}
        created = create_saved_filter(db, "Deser Test", "analysis_results", cfg)
        fetched = get_saved_filter(db, created["id"])
        assert isinstance(fetched["filter_config"], dict)
        assert fetched["filter_config"]["search"] == "test"


# ---------------------------------------------------------------------------
# API integration tests: /api/saved-filters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sf_api_client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("saved_filters_api")

    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp / "models")

    from src.api import dependencies
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()

    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as c:
        yield c

    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR"]:
        os.environ.pop(key, None)


class TestSavedFiltersAPI:
    def test_create_201(self, sf_api_client):
        resp = sf_api_client.post(
            "/api/saved-filters",
            json={
                "name": "API Filter",
                "entity_type": "analysis_results",
                "filter_config": {"filters": [{"col": "language", "op": "eq", "val": "ru"}]},
            },
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["id"].startswith("sf_")
        assert body["name"] == "API Filter"
        sf_api_client.saved_filter_id = body["id"]  # type: ignore

    def test_list_200(self, sf_api_client):
        resp = sf_api_client.get("/api/saved-filters")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) >= 1

    def test_list_filter_by_entity_type(self, sf_api_client):
        # Create one with different entity_type
        sf_api_client.post(
            "/api/saved-filters",
            json={"name": "Dataset F", "entity_type": "datasets", "filter_config": {}},
        )
        resp = sf_api_client.get("/api/saved-filters?entity_type=datasets")
        assert resp.status_code == 200
        body = resp.json()
        for item in body:
            assert item["entity_type"] == "datasets"

    def test_get_200(self, sf_api_client):
        filter_id = sf_api_client.saved_filter_id  # type: ignore
        resp = sf_api_client.get(f"/api/saved-filters/{filter_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == filter_id

    def test_get_404(self, sf_api_client):
        resp = sf_api_client.get("/api/saved-filters/sf_nonexistent")
        assert resp.status_code == 404

    def test_update_200(self, sf_api_client):
        filter_id = sf_api_client.saved_filter_id  # type: ignore
        resp = sf_api_client.put(
            f"/api/saved-filters/{filter_id}",
            json={"name": "Updated Filter Name"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == "Updated Filter Name"

    def test_update_404(self, sf_api_client):
        resp = sf_api_client.put(
            "/api/saved-filters/sf_nonexistent",
            json={"name": "X"},
        )
        assert resp.status_code == 404

    def test_delete_200(self, sf_api_client):
        # Create a fresh filter to delete
        resp = sf_api_client.post(
            "/api/saved-filters",
            json={"name": "To Delete", "entity_type": "analysis_results", "filter_config": {}},
        )
        assert resp.status_code == 201
        filter_id = resp.json()["id"]

        del_resp = sf_api_client.delete(f"/api/saved-filters/{filter_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True

        # Should now return 404
        get_resp = sf_api_client.get(f"/api/saved-filters/{filter_id}")
        assert get_resp.status_code == 404

    def test_delete_404(self, sf_api_client):
        resp = sf_api_client.delete("/api/saved-filters/sf_nonexistent")
        assert resp.status_code == 404

    def test_create_validates_name_required(self, sf_api_client):
        resp = sf_api_client.post(
            "/api/saved-filters",
            json={"entity_type": "analysis_results", "filter_config": {}},
        )
        assert resp.status_code == 422

    def test_filter_config_is_dict_in_response(self, sf_api_client):
        resp = sf_api_client.post(
            "/api/saved-filters",
            json={
                "name": "Dict Test",
                "entity_type": "analysis_results",
                "filter_config": {"search": "hello", "filters": []},
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert isinstance(body["filter_config"], dict)
        assert body["filter_config"]["search"] == "hello"
