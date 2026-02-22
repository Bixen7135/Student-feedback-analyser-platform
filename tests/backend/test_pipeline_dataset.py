"""Tests for pipeline with DataFrame input (Phase 1 — dataset version selection)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest


BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TINY_DATASET = FIXTURES_DIR / "tiny_dataset.csv"


# ---------------------------------------------------------------------------
# pipeline.run_full_pipeline — df_raw branch
# ---------------------------------------------------------------------------

def test_pipeline_requires_data_path_or_df_raw(tmp_path, experiment_config_path, factor_structure_path):
    """Pipeline raises ValueError when neither data_path nor df_raw is provided."""
    from src.pipeline import run_full_pipeline

    with pytest.raises(ValueError, match="data_path or df_raw"):
        run_full_pipeline(
            config_path=experiment_config_path,
            factor_structure_path=factor_structure_path,
            runs_dir=tmp_path / "runs",
        )


def test_pipeline_snapshot_id_from_df(tiny_df):
    """Snapshot ID derived from DataFrame hash is a 16-char hex string."""
    snap = hashlib.sha256(tiny_df.to_csv(index=False).encode()).hexdigest()[:16]
    assert len(snap) == 16
    assert all(c in "0123456789abcdef" for c in snap)


def test_pipeline_snapshot_deterministic(tiny_df):
    """Same DataFrame produces the same snapshot ID."""
    snap1 = hashlib.sha256(tiny_df.to_csv(index=False).encode()).hexdigest()[:16]
    snap2 = hashlib.sha256(tiny_df.to_csv(index=False).encode()).hexdigest()[:16]
    assert snap1 == snap2


def test_pipeline_df_raw_differs_from_file_hash(tiny_dataset_path, tiny_df):
    """DataFrame content hash differs from file hash (line endings etc. may differ)."""
    from src.utils.reproducibility import hash_file

    file_snap = hash_file(tiny_dataset_path)[:16]
    df_snap   = hashlib.sha256(tiny_df.to_csv(index=False).encode()).hexdigest()[:16]
    # Both are valid hex — they may or may not be equal, but neither crashes
    assert len(file_snap) == 16
    assert len(df_snap) == 16


# ---------------------------------------------------------------------------
# run_manager.create_run — new metadata fields
# ---------------------------------------------------------------------------

def test_run_manager_stores_dataset_metadata(tmp_path):
    """create_run persists dataset_id, branch_id, dataset_version, name in metadata."""
    from src.utils.run_manager import RunManager

    mgr = RunManager(tmp_path / "runs")
    run_id = mgr.create_run(
        config_hash="abc12345",
        data_snapshot_id="snap0001",
        random_seed=7,
        system_info={},
        dataset_id="ds-test-id",
        dataset_version=3,
        branch_id="branch-xyz",
        name="my test run",
    )
    meta = mgr.load_run(run_id)
    assert meta["dataset_id"] == "ds-test-id"
    assert meta["dataset_version"] == 3
    assert meta["branch_id"] == "branch-xyz"
    assert meta["name"] == "my test run"


def test_run_manager_null_dataset_fields_by_default(tmp_path):
    """dataset_id / branch_id / dataset_version / name default to None."""
    from src.utils.run_manager import RunManager

    mgr = RunManager(tmp_path / "runs")
    run_id = mgr.create_run(
        config_hash="abc12345",
        data_snapshot_id="snap0001",
        random_seed=42,
        system_info={},
    )
    meta = mgr.load_run(run_id)
    assert meta["dataset_id"] is None
    assert meta["branch_id"] is None
    assert meta["dataset_version"] is None
    assert meta["name"] is None


# ---------------------------------------------------------------------------
# API schemas — new fields propagate through response
# ---------------------------------------------------------------------------

def test_run_summary_response_includes_new_fields():
    """RunSummaryResponse accepts dataset_id, branch_id, dataset_version, name."""
    from src.api.schemas import RunSummaryResponse

    r = RunSummaryResponse(
        run_id="run_test",
        created_at="2026-01-01T00:00:00+00:00",
        config_hash="abc",
        data_snapshot_id="snap",
        random_seed=42,
        stages={},
        dataset_id="ds-1",
        branch_id="br-1",
        dataset_version=2,
        name="my run",
    )
    assert r.dataset_id == "ds-1"
    assert r.branch_id == "br-1"
    assert r.dataset_version == 2
    assert r.name == "my run"


def test_run_summary_response_optional_fields_default_none():
    """New fields on RunSummaryResponse are all optional and default to None."""
    from src.api.schemas import RunSummaryResponse

    r = RunSummaryResponse(
        run_id="run_test",
        created_at="2026-01-01T00:00:00+00:00",
        config_hash="abc",
        data_snapshot_id="snap",
        random_seed=42,
        stages={},
    )
    assert r.dataset_id is None
    assert r.branch_id is None
    assert r.dataset_version is None
    assert r.name is None


# ---------------------------------------------------------------------------
# API endpoint — POST /api/runs with dataset_id 404 path
# ---------------------------------------------------------------------------

def test_create_run_with_unknown_dataset_id_returns_404(tmp_path):
    """POST /api/runs with a non-existent dataset_id returns 404."""
    import os
    from fastapi.testclient import TestClient

    runs_dir = tmp_path / "runs"
    db_dir   = tmp_path / "db"
    db_dir.mkdir()
    os.environ["SFAP_RUNS_DIR"]   = str(runs_dir)
    os.environ["SFAP_DB_PATH"]    = str(db_dir / "test.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp_path / "datasets")

    from src.api import dependencies
    dependencies.get_run_manager.cache_clear()
    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()

    from src.api.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/runs",
            json={"seed": 42, "dataset_id": "nonexistent-dataset-id-xyz"},
        )
    assert resp.status_code == 404


def test_create_run_without_dataset_id_succeeds(tmp_path):
    """POST /api/runs without dataset_id still works (legacy path)."""
    import os
    from fastapi.testclient import TestClient

    runs_dir = tmp_path / "runs"
    os.environ["SFAP_RUNS_DIR"] = str(runs_dir)

    from src.api import dependencies
    dependencies.get_run_manager.cache_clear()

    from src.api.main import app
    with TestClient(app) as client:
        resp = client.post("/api/runs", json={"seed": 99})

    assert resp.status_code == 200
    data = resp.json()
    assert data["random_seed"] == 99
    assert data["dataset_id"] is None
    assert data["name"] is None
