"""Tests for GET /api/summary."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import orjson
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.api.main import app
from src.api import dependencies


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture()
def summary_client(tmp_path):
    db_path = tmp_path / "summary.db"
    runs_dir = tmp_path / "runs"
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"

    os.environ["SFAP_DB_PATH"] = str(db_path)
    os.environ["SFAP_RUNS_DIR"] = str(runs_dir)
    os.environ["SFAP_DATASETS_DIR"] = str(datasets_dir)
    os.environ["SFAP_MODELS_DIR"] = str(models_dir)

    dependencies._get_db.cache_clear()
    dependencies.get_run_manager.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()

    now = _utcnow()
    dataset_id = f"ds_{uuid4().hex[:8]}"
    version_id = f"ver_{uuid4().hex[:8]}"
    branch_id = f"br_{uuid4().hex[:8]}"
    model_id = f"model_{uuid4().hex[:8]}"
    analysis_id = f"analysis_{uuid4().hex[:8]}"
    run_id = f"run_{uuid4().hex[:8]}"

    db = dependencies.get_db()
    schema_info = orjson.dumps(
        [{"name": "item_1"}, {"name": "item_2"}, {"name": "text_feedback"}]
    ).decode()
    column_roles = orjson.dumps(
        {"q1": "item_1", "q2": "item_2", "text_feedback": "text"}
    ).decode()

    db.execute(
        """INSERT INTO datasets
           (id, name, description, tags, author, created_at, current_version, schema_info,
            row_count, file_size_bytes, sha256, status, default_branch_id)
           VALUES (?, ?, '', '[]', '', ?, 1, ?, 120, 1, 'abc', 'active', ?)""",
        (dataset_id, "summary_ds", now, schema_info, branch_id),
    )
    db.execute(
        """INSERT INTO dataset_branches
           (id, dataset_id, name, description, base_version_id, head_version_id, author, created_at, is_default, is_deleted)
           VALUES (?, ?, 'main', '', NULL, ?, '', ?, 1, 0)""",
        (branch_id, dataset_id, version_id, now),
    )
    db.execute(
        """INSERT INTO dataset_versions
           (id, dataset_id, version, created_at, author, reason, sha256,
            row_count, file_size_bytes, storage_path, branch_id, column_roles, is_deleted)
           VALUES (?, ?, 1, ?, '', 'initial', 'abc', 120, 1, ?, ?, ?, 0)""",
        (version_id, dataset_id, now, str(tmp_path / "data.csv"), branch_id, column_roles),
    )
    db.execute(
        """INSERT INTO models
           (id, name, task, model_type, version, dataset_id, dataset_version,
            config, metrics, created_at, status, storage_path, run_id)
           VALUES (?, 'summary_model', 'sentiment', 'tfidf', 1, ?, 1,
                   '{}', '{}', ?, 'active', ?, ?)""",
        (model_id, dataset_id, now, str(models_dir / "summary_model.pkl"), run_id),
    )
    db.execute(
        """INSERT INTO analysis_runs
           (id, created_at, status, dataset_id, model_ids)
           VALUES (?, ?, 'completed', ?, ?)""",
        (analysis_id, now, dataset_id, orjson.dumps([model_id]).decode()),
    )
    db.execute(
        """INSERT INTO pipeline_runs
           (id, dataset_id, dataset_version, config_hash, data_snapshot_id, random_seed,
            status, created_at, completed_at, git_commit, system_info)
           VALUES (?, ?, 1, 'cfg', 'snap', 42, 'completed', ?, ?, 'deadbee', '{}')""",
        (run_id, dataset_id, now, now),
    )
    db.commit()

    run_dir = runs_dir / run_id / "psychometrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cfa_summary.json").write_bytes(
        orjson.dumps({"factor_names": ["program_quality", "resources", "digital"]})
    )

    with TestClient(app) as c:
        yield c

    for key in ["SFAP_DB_PATH", "SFAP_RUNS_DIR", "SFAP_DATASETS_DIR", "SFAP_MODELS_DIR"]:
        os.environ.pop(key, None)


def test_summary_endpoint_returns_dynamic_counts(summary_client):
    resp = summary_client.get("/api/summary")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total_datasets"] == 1
    assert data["total_models"] == 1
    assert data["total_analyses"] == 1
    assert data["total_responses"] == 120
    assert data["n_survey_items"] == 2
    assert data["n_latent_factors"] == 3
