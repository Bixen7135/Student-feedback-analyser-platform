"""Integration tests for the analytics API endpoints."""
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

from src.api import dependencies
from src.api.main import app
from src.analysis import runner as analysis_runner
from src.training import runner as training_runner


def _make_labelled_csv(path: Path, n_per_class: int = 15) -> None:
    rows = []
    for label in ["positive", "neutral", "negative"]:
        for i in range(n_per_class):
            rows.append(
                {
                    "text_feedback": f"Feedback {i} class {label} extra words here",
                    "language": "ru" if i % 2 == 0 else "kz",
                    "sentiment_class": label,
                    "detail_level": "short" if i < 5 else ("medium" if i < 10 else "long"),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def analytics_api_client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analytics_api")

    os.environ["SFAP_DB_PATH"] = str(tmp / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp / "models")
    os.environ["SFAP_TRAINING_DIR"] = str(tmp / "training_runs")
    os.environ["SFAP_ANALYSIS_DIR"] = str(tmp / "analysis_runs")

    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()
    dependencies.get_run_manager.cache_clear()

    with training_runner._jobs_lock:
        training_runner._jobs.clear()
    with analysis_runner._jobs_lock:
        analysis_runner._jobs.clear()

    with TestClient(app) as client:
        csv_path = tmp / "sample.csv"
        _make_labelled_csv(csv_path)
        with open(csv_path, "rb") as file_obj:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("sample.csv", file_obj, "text/csv")},
                data={"name": "analytics_api_ds"},
            )
        assert upload_resp.status_code == 200, upload_resp.text
        dataset_id = upload_resp.json()["id"]
        client.dataset_id = dataset_id  # type: ignore[attr-defined]

        train_resp = client.post(
            "/api/training/start",
            json={
                "dataset_id": dataset_id,
                "task": "sentiment",
                "model_type": "tfidf",
                "seed": 42,
            },
        )
        assert train_resp.status_code == 202, train_resp.text
        model_id = train_resp.json().get("model_id")
        if not model_id:
            job_id = train_resp.json()["job_id"]
            model_id = client.get(f"/api/training/{job_id}/status").json().get("model_id")
        assert model_id
        client.model_id = model_id  # type: ignore[attr-defined]

        analysis_resp = client.post(
            "/api/analyses",
            json={
                "dataset_id": dataset_id,
                "model_ids": [model_id],
                "name": "Analytics API Analysis",
            },
        )
        assert analysis_resp.status_code == 202, analysis_resp.text
        analysis_id = analysis_resp.json()["job_id"]
        client.analysis_id = analysis_id  # type: ignore[attr-defined]

        status_resp = client.get(f"/api/analyses/{analysis_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "completed"

        yield client

    for key in [
        "SFAP_DB_PATH",
        "SFAP_DATASETS_DIR",
        "SFAP_MODELS_DIR",
        "SFAP_TRAINING_DIR",
        "SFAP_ANALYSIS_DIR",
    ]:
        os.environ.pop(key, None)


def test_dataset_descriptive_endpoint(analytics_api_client) -> None:
    resp = analytics_api_client.get(
        f"/api/analytics/datasets/{analytics_api_client.dataset_id}/descriptive"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dataset_id"] == analytics_api_client.dataset_id
    assert body["row_count"] == 45
    assert "language" in body["summary"]["categorical"]
    assert "text_feedback" in body["summary"]["text"]


def test_dataset_correlations_endpoint(analytics_api_client) -> None:
    resp = analytics_api_client.get(
        f"/api/analytics/datasets/{analytics_api_client.dataset_id}/correlations"
        "?columns=language,sentiment_class"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dataset_id"] == analytics_api_client.dataset_id
    assert body["row_count"] == 45
    assert len(body["correlations"]) == 1
    assert body["correlations"][0]["metric"] == "cramers_v"


def test_analysis_descriptive_endpoint_supports_filtering(analytics_api_client) -> None:
    resp = analytics_api_client.get(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/descriptive"
        "?filter_col=language&filter_val=ru"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_id"] == analytics_api_client.analysis_id
    assert body["row_count"] == 24
    assert body["summary"]["categorical"]["language"]["levels"]["ru"]["count"] == 24


def test_analysis_correlations_endpoint(analytics_api_client) -> None:
    resp = analytics_api_client.get(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/correlations"
        "?columns=language,sentiment_class"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_id"] == analytics_api_client.analysis_id
    assert len(body["correlations"]) == 1
    assert body["correlations"][0]["metric"] == "cramers_v"


def test_analysis_diagnostics_endpoint(analytics_api_client) -> None:
    resp = analytics_api_client.get(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/diagnostics"
        "?task=sentiment"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task"] == "sentiment"
    assert body["label_col"] == "sentiment_class"
    assert body["pred_col"].endswith("_pred")
    diagnostics = body["diagnostics"]
    assert diagnostics["n_rows"] == 45
    assert len(diagnostics["labels"]) >= 2
    assert "confusion_matrix" in diagnostics


def test_analysis_diagnostics_unknown_task_returns_422(analytics_api_client) -> None:
    resp = analytics_api_client.get(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/diagnostics"
        "?task=language"
    )
    assert resp.status_code == 422


def test_analysis_embeddings_endpoint(analytics_api_client) -> None:
    resp = analytics_api_client.post(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/embeddings",
        json={},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_id"] == analytics_api_client.analysis_id
    assert body["count"] == 45
    assert "row_idx" in body["columns"]
    assert "x" in body["columns"]
    assert "y" in body["columns"]
    assert len(body["points"]) == 45


def test_analysis_embeddings_endpoint_reuses_cached_artifact(analytics_api_client) -> None:
    first = analytics_api_client.post(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/embeddings",
        json={"reuse_cached": True},
    )
    second = analytics_api_client.post(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/embeddings",
        json={"reuse_cached": True},
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    body_1 = first.json()
    body_2 = second.json()
    assert body_1["artifact_path"] == body_2["artifact_path"]
    assert body_1["metadata"] == body_2["metadata"]
    assert body_1["points"] == body_2["points"]
    assert Path(body_1["artifact_path"]).exists()


def test_analysis_cluster_endpoint(analytics_api_client) -> None:
    resp = analytics_api_client.post(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/cluster",
        json={"method": "kmeans", "k": 3},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_id"] == analytics_api_client.analysis_id
    assert "cluster_counts" in body
    assert len(body["clusters"]) == 45
    assert sum(body["cluster_counts"].values()) == 45


def test_analysis_cluster_endpoint_is_idempotent(analytics_api_client) -> None:
    first = analytics_api_client.post(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/cluster",
        json={"method": "kmeans", "k": 3, "reuse_embeddings": True},
    )
    second = analytics_api_client.post(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/cluster",
        json={"method": "kmeans", "k": 3, "reuse_embeddings": True},
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    body_1 = first.json()
    body_2 = second.json()
    assert body_1["artifact_path"] == body_2["artifact_path"]
    assert body_1["metadata"] == body_2["metadata"]
    assert body_1["cluster_counts"] == body_2["cluster_counts"]
    assert body_1["clusters"] == body_2["clusters"]
    assert Path(body_1["artifact_path"]).exists()


def test_analysis_outliers_endpoint(analytics_api_client) -> None:
    resp = analytics_api_client.post(
        f"/api/analytics/analyses/{analytics_api_client.analysis_id}/outliers",
        json={"method": "isolation_forest", "contamination": 0.1},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_id"] == analytics_api_client.analysis_id
    assert len(body["rows"]) == 45
    assert 0 <= body["outlier_count"] <= 45
    assert "is_outlier" in body["rows"][0]
    assert "outlier_score" in body["rows"][0]


def test_model_importance_endpoint(analytics_api_client) -> None:
    resp = analytics_api_client.get(
        f"/api/models/{analytics_api_client.model_id}/importance?top_n=5"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_id"] == analytics_api_client.model_id
    assert body["top_n"] == 5
    assert len(body["classes"]) >= 2
    first_class = body["classes"][0]
    assert len(body["per_class"][first_class]) == 5
    assert "feature" in body["per_class"][first_class][0]


def test_model_explain_endpoint_with_text(analytics_api_client) -> None:
    resp = analytics_api_client.post(
        f"/api/models/{analytics_api_client.model_id}/explain",
        json={"text": "excellent course and very helpful teacher", "top_n": 5},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_id"] == analytics_api_client.model_id
    assert body["source"]["type"] == "text"
    assert "predicted_class" in body
    assert len(body["top_features"]) <= 5


def test_model_explain_endpoint_with_analysis_row(analytics_api_client) -> None:
    resp = analytics_api_client.post(
        f"/api/models/{analytics_api_client.model_id}/explain",
        json={
            "analysis_id": analytics_api_client.analysis_id,
            "row_idx": 0,
            "top_n": 5,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"]["type"] == "analysis_row"
    assert body["source"]["analysis_id"] == analytics_api_client.analysis_id
    assert body["source"]["row_idx"] == 0
