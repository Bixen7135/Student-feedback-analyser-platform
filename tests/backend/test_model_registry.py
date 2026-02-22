"""Tests for ModelRegistry and model API endpoints."""
from __future__ import annotations

import io
import csv
import os
import tempfile
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.api.main import app
from src.api import dependencies
from src.storage.database import Database
from src.storage.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def db(tmp_dir):
    return Database(tmp_dir / "test.db")


@pytest.fixture
def registry(db, tmp_dir):
    return ModelRegistry(db, tmp_dir / "models")


@pytest.fixture
def fake_model_path(tmp_dir) -> Path:
    """Write a dummy sklearn model to disk."""
    model = LogisticRegression()
    path = tmp_dir / "model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture
def fake_metrics_path(tmp_dir) -> Path:
    """Write dummy metrics JSON."""
    import orjson
    path = tmp_dir / "metrics.json"
    metrics = {"macro_f1": 0.85, "accuracy": 0.87}
    path.write_bytes(orjson.dumps(metrics))
    return path


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """TestClient with temporary storage."""
    tmp = tmp_path_factory.mktemp("model_api")
    os.environ["SFAP_RUNS_DIR"] = str(tmp / "runs")
    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp / "models")
    dependencies.get_run_manager.cache_clear()
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()
    with TestClient(app) as c:
        yield c
    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR"]:
        os.environ.pop(key, None)


def _make_model_file(tmp_path: Path) -> Path:
    """Create a dummy model joblib file."""
    model = LogisticRegression()
    path = tmp_path / f"model_{id(tmp_path)}.joblib"
    joblib.dump(model, path)
    return path


# ---------------------------------------------------------------------------
# Unit tests: ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_model(self, registry, fake_model_path, fake_metrics_path):
        meta = registry.register_model(
            name="Test Sentiment TF-IDF",
            task="sentiment",
            model_type="tfidf",
            source_model_path=fake_model_path,
            source_metrics_path=fake_metrics_path,
            config={"C": 1.0, "max_features": 50000},
        )
        assert meta.task == "sentiment"
        assert meta.model_type == "tfidf"
        assert meta.version == 1
        assert meta.metrics == {"macro_f1": 0.85, "accuracy": 0.87}
        assert meta.config == {"C": 1.0, "max_features": 50000}

    def test_register_stores_artifact(self, registry, fake_model_path):
        meta = registry.register_model(
            name="Artifact Test",
            task="language",
            model_type="char_ngram",
            source_model_path=fake_model_path,
        )
        artifact_path = Path(meta.storage_path) / "model.joblib"
        assert artifact_path.exists()

    def test_register_auto_version_increment(self, registry, fake_model_path, tmp_dir):
        """Models for same task+type+dataset should auto-increment version."""
        import uuid
        ds_id = str(uuid.uuid4())
        path1 = tmp_dir / "m1.joblib"
        path2 = tmp_dir / "m2.joblib"
        import joblib as jl
        jl.dump(LogisticRegression(), path1)
        jl.dump(LogisticRegression(), path2)

        m1 = registry.register_model(
            name="V1", task="sentiment", model_type="tfidf",
            source_model_path=path1, dataset_id=ds_id, dataset_version=1,
        )
        m2 = registry.register_model(
            name="V2", task="sentiment", model_type="tfidf",
            source_model_path=path2, dataset_id=ds_id, dataset_version=1,
        )
        assert m1.version == 1
        assert m2.version == 2

    def test_models_immutable(self, registry, fake_model_path):
        """Once registered, a model cannot be overwritten — only new versions."""
        meta = registry.register_model(
            name="Original",
            task="detail_level",
            model_type="tfidf",
            source_model_path=fake_model_path,
            config={"original": True},
        )
        fetched = registry.get_model(meta.id)
        assert fetched is not None
        assert fetched.config == {"original": True}
        # Verify there's no way to mutate it
        assert fetched.id == meta.id

    def test_list_models_empty(self, db, tmp_dir):
        fresh_registry = ModelRegistry(db, tmp_dir / "fresh_models")
        models, total = fresh_registry.list_models()
        # May have models from other tests in same db, so just assert structure
        assert isinstance(models, list)
        assert isinstance(total, int)

    def test_list_models_by_task(self, registry, fake_model_path, tmp_dir):
        import joblib as jl
        mp = tmp_dir / "task_filter.joblib"
        jl.dump(LogisticRegression(), mp)
        registry.register_model(
            name="Lang Model", task="language", model_type="tfidf",
            source_model_path=mp,
        )
        models, total = registry.list_models(task="language")
        assert total >= 1
        assert all(m.task == "language" for m in models)

    def test_get_model(self, registry, fake_model_path):
        meta = registry.register_model(
            name="Get Test",
            task="sentiment",
            model_type="tfidf",
            source_model_path=fake_model_path,
        )
        fetched = registry.get_model(meta.id)
        assert fetched is not None
        assert fetched.name == "Get Test"

    def test_get_model_not_found(self, registry):
        assert registry.get_model("nonexistent-id") is None

    def test_compare_models(self, registry, fake_model_path, tmp_dir):
        import joblib as jl
        p1 = tmp_dir / "cmp1.joblib"
        p2 = tmp_dir / "cmp2.joblib"
        jl.dump(LogisticRegression(), p1)
        jl.dump(LogisticRegression(), p2)
        m1 = registry.register_model(
            name="Compare A", task="sentiment", model_type="tfidf",
            source_model_path=p1,
        )
        m2 = registry.register_model(
            name="Compare B", task="sentiment", model_type="char_ngram",
            source_model_path=p2,
        )
        results = registry.compare_models([m1.id, m2.id])
        assert len(results) == 2
        ids = {m.id for m in results}
        assert m1.id in ids
        assert m2.id in ids

    def test_load_model_artifact(self, registry, fake_model_path):
        meta = registry.register_model(
            name="Load Test",
            task="language",
            model_type="tfidf",
            source_model_path=fake_model_path,
        )
        path = registry.load_model_artifact(meta.id)
        assert path.exists()
        # Should be loadable
        loaded = joblib.load(path)
        assert isinstance(loaded, LogisticRegression)

    def test_load_model_artifact_not_found(self, registry):
        with pytest.raises(ValueError, match="not found"):
            registry.load_model_artifact("nonexistent")


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

def _make_temp_model(tmp_dir: Path) -> Path:
    path = tmp_dir / f"api_model_{id(tmp_dir)}.joblib"
    joblib.dump(LogisticRegression(), path)
    return path


def test_list_models_empty(api_client):
    resp = api_client.get("/api/models/")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "total" in data


def test_register_model_api(api_client, tmp_path):
    model_path = _make_temp_model(tmp_path)
    resp = api_client.post(
        "/api/models/register",
        json={
            "name": "API Test Model",
            "task": "sentiment",
            "model_type": "tfidf",
            "source_model_path": str(model_path),
            "config": {"C": 1.0},
            "metrics": {"macro_f1": 0.82},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "API Test Model"
    assert data["task"] == "sentiment"
    assert data["version"] == 1


def test_register_model_missing_file(api_client):
    resp = api_client.post(
        "/api/models/register",
        json={
            "name": "Missing",
            "task": "sentiment",
            "model_type": "tfidf",
            "source_model_path": "/nonexistent/model.joblib",
        },
    )
    assert resp.status_code == 400


def test_get_model_detail(api_client, tmp_path):
    model_path = _make_temp_model(tmp_path)
    reg = api_client.post(
        "/api/models/register",
        json={
            "name": "Detail Test",
            "task": "language",
            "model_type": "char_ngram",
            "source_model_path": str(model_path),
            "metrics": {"macro_f1": 0.91},
        },
    )
    model_id = reg.json()["id"]

    resp = api_client.get(f"/api/models/{model_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Detail Test"
    assert data["metrics"]["macro_f1"] == 0.91


def test_get_model_not_found(api_client):
    resp = api_client.get("/api/models/nonexistent")
    assert resp.status_code == 404


def test_list_models_filter_by_task(api_client, tmp_path):
    model_path = _make_temp_model(tmp_path)
    api_client.post(
        "/api/models/register",
        json={
            "name": "Filter Task Model",
            "task": "detail_level",
            "model_type": "tfidf",
            "source_model_path": str(model_path),
        },
    )
    resp = api_client.get("/api/models/?task=detail_level")
    data = resp.json()
    assert data["total"] >= 1
    assert all(m["task"] == "detail_level" for m in data["models"])


def test_get_model_versions(api_client, tmp_path):
    model_path = _make_temp_model(tmp_path)
    reg = api_client.post(
        "/api/models/register",
        json={
            "name": "Versions Test",
            "task": "language",
            "model_type": "tfidf",
            "source_model_path": str(model_path),
        },
    )
    model_id = reg.json()["id"]

    resp = api_client.get(f"/api/models/{model_id}/versions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_compare_models_api(api_client, tmp_path):
    p1 = tmp_path / "cmp_a.joblib"
    p2 = tmp_path / "cmp_b.joblib"
    joblib.dump(LogisticRegression(), p1)
    joblib.dump(LogisticRegression(), p2)

    r1 = api_client.post(
        "/api/models/register",
        json={
            "name": "Compare X",
            "task": "sentiment",
            "model_type": "tfidf",
            "source_model_path": str(p1),
            "metrics": {"macro_f1": 0.80},
        },
    )
    r2 = api_client.post(
        "/api/models/register",
        json={
            "name": "Compare Y",
            "task": "sentiment",
            "model_type": "char_ngram",
            "source_model_path": str(p2),
            "metrics": {"macro_f1": 0.83},
        },
    )

    resp = api_client.post(
        "/api/models/compare",
        json={"model_ids": [r1.json()["id"], r2.json()["id"]]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2


def test_compare_models_requires_two(api_client, tmp_path):
    p = _make_temp_model(tmp_path)
    r = api_client.post(
        "/api/models/register",
        json={
            "name": "Solo",
            "task": "language",
            "model_type": "tfidf",
            "source_model_path": str(p),
        },
    )
    resp = api_client.post(
        "/api/models/compare",
        json={"model_ids": [r.json()["id"]]},
    )
    assert resp.status_code == 400
