"""Tests for Phase 3 analytics endpoints: distributions, segment-stats, cross-compare."""
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
                    "text_feedback": f"Feedback {i} class {s} extra words here",
                    "language": "ru" if i % 2 == 0 else "kz",
                    "sentiment_class": s,
                    "detail_level": "short" if i < 5 else ("medium" if i < 10 else "long"),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Fixture: isolated client with trained model + completed analysis
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analytics_client(tmp_path_factory):
    """TestClient with a completed analysis job ready for analytics queries."""
    tmp = tmp_path_factory.mktemp("analytics")

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

    with TestClient(app) as c:
        # Upload dataset
        csv_path = tmp / "sample.csv"
        _make_labelled_csv(csv_path)
        with open(csv_path, "rb") as f:
            resp = c.post(
                "/api/datasets/upload",
                files={"file": ("sample.csv", f, "text/csv")},
                data={"name": "analytics_test_ds"},
            )
        assert resp.status_code == 200, resp.text
        dataset_id = resp.json()["id"]
        c.dataset_id = dataset_id  # type: ignore[attr-defined]

        # Train a model
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
        model_id = train_resp.json().get("model_id")
        if not model_id:
            job_id = train_resp.json()["job_id"]
            model_id = c.get(f"/api/training/{job_id}/status").json().get("model_id")
        assert model_id, "Model not registered"
        c.model_id = model_id  # type: ignore[attr-defined]

        # Start and complete analysis
        analysis_resp = c.post(
            "/api/analyses",
            json={
                "dataset_id": dataset_id,
                "model_ids": [model_id],
                "name": "Analytics Test Analysis",
            },
        )
        assert analysis_resp.status_code == 202, analysis_resp.text
        analysis_id = analysis_resp.json()["job_id"]
        c.analysis_id = analysis_id  # type: ignore[attr-defined]

        # Verify it completed
        status = c.get(f"/api/analyses/{analysis_id}/status").json()
        assert status["status"] == "completed", f"Analysis failed: {status}"

        yield c

    for key in ["SFAP_DB_PATH", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR",
                "SFAP_TRAINING_DIR", "SFAP_ANALYSIS_DIR"]:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Tests: GET /distributions
# ---------------------------------------------------------------------------


class TestDistributions:
    def test_returns_distributions_for_pred_col(self, analytics_client):
        """Distributions endpoint returns value counts for a prediction column."""
        c = analytics_client
        analysis_id = c.analysis_id

        # Get columns first
        results = c.get(f"/api/analyses/{analysis_id}/results").json()
        pred_cols = [col for col in results["columns"] if col.endswith("_pred")]
        assert pred_cols, "No prediction columns found"

        resp = c.get(
            f"/api/analyses/{analysis_id}/distributions",
            params={"columns": pred_cols[0]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "distributions" in body
        assert pred_cols[0] in body["distributions"]
        dist = body["distributions"][pred_cols[0]]
        assert isinstance(dist, dict)
        # Should have 3 sentiment classes
        assert len(dist) > 0
        # Values should be integers
        for v in dist.values():
            assert isinstance(v, int)

    def test_returns_distributions_for_original_column(self, analytics_client):
        """Distributions endpoint also works on non-prediction columns."""
        c = analytics_client
        resp = c.get(
            f"/api/analyses/{c.analysis_id}/distributions",
            params={"columns": "sentiment_class"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "sentiment_class" in body["distributions"]
        dist = body["distributions"]["sentiment_class"]
        assert set(dist.keys()) == {"positive", "neutral", "negative"}

    def test_multiple_columns(self, analytics_client):
        """Multiple columns can be requested comma-separated."""
        c = analytics_client
        resp = c.get(
            f"/api/analyses/{c.analysis_id}/distributions",
            params={"columns": "sentiment_class,language"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "sentiment_class" in body["distributions"]
        assert "language" in body["distributions"]

    def test_unknown_column_skipped(self, analytics_client):
        """Unknown column names are silently skipped."""
        c = analytics_client
        resp = c.get(
            f"/api/analyses/{c.analysis_id}/distributions",
            params={"columns": "sentiment_class,does_not_exist"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "sentiment_class" in body["distributions"]
        assert "does_not_exist" not in body["distributions"]

    def test_404_unknown_analysis(self, analytics_client):
        resp = analytics_client.get(
            "/api/analyses/nonexistent_id/distributions",
            params={"columns": "sentiment_class"},
        )
        assert resp.status_code == 404

    def test_requires_columns_param(self, analytics_client):
        resp = analytics_client.get(
            f"/api/analyses/{analytics_client.analysis_id}/distributions",
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: GET /segment-stats
# ---------------------------------------------------------------------------


class TestSegmentStats:
    def test_basic_segment_stats(self, analytics_client):
        """Segment stats returns count/mean/median/std per group."""
        c = analytics_client

        # Get prediction columns
        results = c.get(f"/api/analyses/{c.analysis_id}/results").json()
        conf_cols = [col for col in results["columns"] if col.endswith("_conf")]
        assert conf_cols, "No confidence columns found"

        resp = c.get(
            f"/api/analyses/{c.analysis_id}/segment-stats",
            params={"group_by": "sentiment_class", "metric_col": conf_cols[0]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "groups" in body
        assert body["group_by"] == "sentiment_class"
        assert body["metric_col"] == conf_cols[0]
        groups = body["groups"]
        assert len(groups) == 3  # positive, neutral, negative

        for g in groups:
            assert "group" in g
            assert "count" in g
            assert "mean" in g
            assert "median" in g
            assert "std" in g
            assert g["count"] > 0
            assert 0.0 <= g["mean"] <= 1.0

    def test_segment_stats_by_language(self, analytics_client):
        """Works with language as group_by."""
        c = analytics_client
        results = c.get(f"/api/analyses/{c.analysis_id}/results").json()
        conf_cols = [col for col in results["columns"] if col.endswith("_conf")]

        resp = c.get(
            f"/api/analyses/{c.analysis_id}/segment-stats",
            params={"group_by": "language", "metric_col": conf_cols[0]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        groups = body["groups"]
        assert len(groups) == 2  # ru, kz

    def test_non_numeric_metric_returns_empty(self, analytics_client):
        """Non-numeric metric col returns empty groups (no error)."""
        c = analytics_client
        resp = c.get(
            f"/api/analyses/{c.analysis_id}/segment-stats",
            params={"group_by": "language", "metric_col": "sentiment_class"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # sentiment_class is categorical, can't compute numeric stats → empty
        assert body["groups"] == []

    def test_missing_column_returns_empty(self, analytics_client):
        """Missing group_by column returns empty groups."""
        c = analytics_client
        results = c.get(f"/api/analyses/{c.analysis_id}/results").json()
        conf_cols = [col for col in results["columns"] if col.endswith("_conf")]
        resp = c.get(
            f"/api/analyses/{c.analysis_id}/segment-stats",
            params={"group_by": "no_such_col", "metric_col": conf_cols[0]},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["groups"] == []

    def test_404_unknown_analysis(self, analytics_client):
        resp = analytics_client.get(
            "/api/analyses/nonexistent_id/segment-stats",
            params={"group_by": "language", "metric_col": "x"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: POST /cross-compare
# ---------------------------------------------------------------------------


class TestCrossCompare:
    def _make_second_analysis(self, c) -> str:
        """Create a second completed analysis for the same dataset."""
        resp = c.post(
            "/api/analyses",
            json={
                "dataset_id": c.dataset_id,
                "model_ids": [c.model_id],
                "name": "Cross-Compare Second",
            },
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]
        # Verify completed
        status = c.get(f"/api/analyses/{job_id}/status").json()
        assert status["status"] == "completed", f"Second analysis failed: {status}"
        return job_id

    def test_cross_compare_two_analyses(self, analytics_client):
        """Cross-compare of two analyses returns distributions + disagreement rates."""
        c = analytics_client
        second_id = self._make_second_analysis(c)

        # Get a pred column
        results = c.get(f"/api/analyses/{c.analysis_id}/results").json()
        pred_cols = [col for col in results["columns"] if col.endswith("_pred")]
        assert pred_cols

        resp = c.post(
            "/api/analyses/cross-compare",
            json={
                "analysis_ids": [c.analysis_id, second_id],
                "columns": pred_cols[:1],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert "analysis_ids" in body
        assert "columns" in body
        assert "per_analysis" in body
        assert "disagreement_rates" in body

        assert set(body["analysis_ids"]) == {c.analysis_id, second_id}
        assert pred_cols[0] in body["columns"]

        # per_analysis has entry for each analysis
        for aid in [c.analysis_id, second_id]:
            assert aid in body["per_analysis"]
            assert pred_cols[0] in body["per_analysis"][aid]

        # Disagreement rates are floats in [0,1]
        for col, rate in body["disagreement_rates"].items():
            assert 0.0 <= rate <= 1.0

    def test_cross_compare_same_analysis_twice(self, analytics_client):
        """Comparing an analysis with itself should give 0% disagreement."""
        c = analytics_client
        second_id = self._make_second_analysis(c)

        results = c.get(f"/api/analyses/{c.analysis_id}/results").json()
        pred_cols = [col for col in results["columns"] if col.endswith("_pred")]

        resp = c.post(
            "/api/analyses/cross-compare",
            json={
                "analysis_ids": [c.analysis_id, c.analysis_id],
                "columns": pred_cols[:1],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Same analysis compared to itself = 0 disagreement
        for rate in body["disagreement_rates"].values():
            assert rate == 0.0

        _ = second_id  # suppress unused warning

    def test_cross_compare_requires_two_analyses(self, analytics_client):
        """Less than 2 analysis_ids is rejected."""
        resp = analytics_client.post(
            "/api/analyses/cross-compare",
            json={
                "analysis_ids": [analytics_client.analysis_id],
                "columns": ["sentiment_class"],
            },
        )
        assert resp.status_code == 422

    def test_cross_compare_unknown_analysis_404(self, analytics_client):
        resp = analytics_client.post(
            "/api/analyses/cross-compare",
            json={
                "analysis_ids": [analytics_client.analysis_id, "nonexistent"],
                "columns": ["sentiment_class"],
            },
        )
        assert resp.status_code == 404

    def test_cross_compare_three_analyses(self, analytics_client):
        """Cross-compare supports more than 2 analyses."""
        c = analytics_client
        second_id = self._make_second_analysis(c)
        third_id = self._make_second_analysis(c)

        results = c.get(f"/api/analyses/{c.analysis_id}/results").json()
        pred_cols = [col for col in results["columns"] if col.endswith("_pred")]

        resp = c.post(
            "/api/analyses/cross-compare",
            json={
                "analysis_ids": [c.analysis_id, second_id, third_id],
                "columns": pred_cols[:1],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["analysis_ids"]) == 3
        assert len(body["per_analysis"]) == 3


# ---------------------------------------------------------------------------
# Tests: runner helper functions directly
# ---------------------------------------------------------------------------


class TestRunnerHelpers:
    def _write_results_csv(self, path: Path) -> None:
        """Write a sample results CSV for unit testing the helpers."""
        rows = []
        for i in range(30):
            rows.append(
                {
                    "text_feedback": f"text {i}",
                    "sentiment_class": ["positive", "neutral", "negative"][i % 3],
                    "language": "ru" if i % 2 == 0 else "kz",
                    "fake_conf": round(0.5 + (i % 5) * 0.1, 2),
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_get_distributions_returns_counts(self, tmp_path):
        analysis_dir = tmp_path / "analysis_abc"
        analysis_dir.mkdir()
        self._write_results_csv(analysis_dir / "results.csv")

        result = analysis_runner.get_distributions(
            artifacts_dir=tmp_path,
            analysis_id="analysis_abc",
            columns=["sentiment_class", "language"],
        )
        assert "distributions" in result
        dists = result["distributions"]
        assert "sentiment_class" in dists
        assert "language" in dists
        # 30 rows, 3 classes, 10 each
        assert dists["sentiment_class"].get("positive") == 10
        assert dists["sentiment_class"].get("neutral") == 10
        assert dists["sentiment_class"].get("negative") == 10
        # 30 rows, 2 languages
        assert dists["language"].get("ru") == 15
        assert dists["language"].get("kz") == 15

    def test_get_distributions_missing_file(self, tmp_path):
        result = analysis_runner.get_distributions(
            artifacts_dir=tmp_path,
            analysis_id="nonexistent",
            columns=["sentiment_class"],
        )
        assert result == {"distributions": {}}

    def test_get_segment_stats_basic(self, tmp_path):
        analysis_dir = tmp_path / "analysis_seg"
        analysis_dir.mkdir()
        self._write_results_csv(analysis_dir / "results.csv")

        result = analysis_runner.get_segment_stats(
            artifacts_dir=tmp_path,
            analysis_id="analysis_seg",
            group_by="language",
            metric_col="fake_conf",
        )
        assert result["group_by"] == "language"
        assert result["metric_col"] == "fake_conf"
        groups = result["groups"]
        assert len(groups) == 2
        group_names = {g["group"] for g in groups}
        assert group_names == {"ru", "kz"}
        for g in groups:
            assert g["count"] == 15
            assert "mean" in g
            assert "median" in g
            assert "std" in g

    def test_get_segment_stats_missing_col(self, tmp_path):
        analysis_dir = tmp_path / "analysis_miss"
        analysis_dir.mkdir()
        self._write_results_csv(analysis_dir / "results.csv")

        result = analysis_runner.get_segment_stats(
            artifacts_dir=tmp_path,
            analysis_id="analysis_miss",
            group_by="does_not_exist",
            metric_col="fake_conf",
        )
        assert result["groups"] == []

    def test_get_cross_compare_disagreements_identical(self, tmp_path):
        """Two identical result files → 0% disagreement."""
        for suffix in ["a", "b"]:
            d = tmp_path / f"analysis_{suffix}"
            d.mkdir()
            self._write_results_csv(d / "results.csv")

        result = analysis_runner.get_cross_compare_disagreements(
            artifacts_dir=tmp_path,
            analysis_ids=["analysis_a", "analysis_b"],
            columns=["sentiment_class"],
        )
        assert result.get("sentiment_class") == 0.0

    def test_get_cross_compare_disagreements_different(self, tmp_path):
        """Two different result files → non-zero disagreement on pred col."""
        for suffix, offset in [("c", 0), ("d", 1)]:
            d = tmp_path / f"analysis_{suffix}"
            d.mkdir()
            rows = [
                {"pred": "positive" if (i + offset) % 2 == 0 else "negative"}
                for i in range(20)
            ]
            pd.DataFrame(rows).to_csv(d / "results.csv", index=False)

        result = analysis_runner.get_cross_compare_disagreements(
            artifacts_dir=tmp_path,
            analysis_ids=["analysis_c", "analysis_d"],
            columns=["pred"],
        )
        assert "pred" in result
        assert 0.0 < result["pred"] <= 1.0
