"""Tests for version-aware /api/datasets/{id}/column-roles."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import orjson
import pandas as pd
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
def versioned_client(tmp_path):
    os.environ["SFAP_DB_PATH"] = str(tmp_path / "versioned.db")
    os.environ["SFAP_DATASETS_DIR"] = str(tmp_path / "datasets")
    os.environ["SFAP_MODELS_DIR"] = str(tmp_path / "models")
    os.environ["SFAP_TRAINING_DIR"] = str(tmp_path / "training")
    os.environ["SFAP_ANALYSIS_DIR"] = str(tmp_path / "analysis")

    dependencies._get_db.cache_clear()
    dependencies.get_dataset_manager.cache_clear()
    dependencies.get_model_registry.cache_clear()
    dependencies.get_run_manager.cache_clear()

    with TestClient(app) as c:
        yield c

    for key in [
        "SFAP_DB_PATH",
        "SFAP_DATASETS_DIR",
        "SFAP_MODELS_DIR",
        "SFAP_TRAINING_DIR",
        "SFAP_ANALYSIS_DIR",
    ]:
        os.environ.pop(key, None)


def test_column_roles_accepts_version_id_and_version(versioned_client, tmp_path):
    csv_path = tmp_path / "roles.csv"
    pd.DataFrame(
        [
            {
                "text_feedback": "good class",
                "sentiment_class": "positive",
                "language": "en",
                "item_1": "5",
            }
        ]
    ).to_csv(csv_path, index=False)

    with open(csv_path, "rb") as f:
        upload_resp = versioned_client.post(
            "/api/datasets/upload",
            files={"file": ("roles.csv", f, "text/csv")},
            data={"name": "roles_test"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    dataset_id = upload_resp.json()["id"]

    versions_resp = versioned_client.get(f"/api/datasets/{dataset_id}/versions")
    assert versions_resp.status_code == 200, versions_resp.text
    versions = versions_resp.json()
    assert len(versions) == 1
    v1 = versions[0]

    db = dependencies.get_db()
    v1_row = db.fetchone(
        "SELECT storage_path FROM dataset_versions WHERE id = ?", (v1["id"],)
    )
    assert v1_row is not None

    v2_id = f"ver_{uuid4().hex[:8]}"
    custom_roles = {"feedback_text": "text", "tone_label": "sentiment", "q1": "item_1"}
    now = _utcnow()
    db.execute(
        """INSERT INTO dataset_versions
           (id, dataset_id, version, created_at, author, reason, sha256,
            row_count, file_size_bytes, storage_path, branch_id, column_roles, is_deleted)
           VALUES (?, ?, 2, ?, '', 'manual test version', 'def', 1, 1, ?, ?, ?, 0)""",
        (
            v2_id,
            dataset_id,
            now,
            v1_row["storage_path"],
            v1["branch_id"],
            orjson.dumps(custom_roles).decode(),
        ),
    )
    db.execute(
        "UPDATE datasets SET current_version = 2 WHERE id = ?",
        (dataset_id,),
    )
    db.commit()

    by_id_resp = versioned_client.get(
        f"/api/datasets/{dataset_id}/column-roles?version_id={v1['id']}"
    )
    assert by_id_resp.status_code == 200, by_id_resp.text
    roles_v1 = by_id_resp.json()["column_roles"]
    assert roles_v1 != custom_roles

    by_number_resp = versioned_client.get(
        f"/api/datasets/{dataset_id}/column-roles?version=2"
    )
    assert by_number_resp.status_code == 200, by_number_resp.text
    assert by_number_resp.json()["column_roles"] == custom_roles

    current_resp = versioned_client.get(f"/api/datasets/{dataset_id}/column-roles")
    assert current_resp.status_code == 200, current_resp.text
    assert current_resp.json()["column_roles"] == custom_roles

