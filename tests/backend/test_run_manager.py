"""Unit tests for RunManager."""
from __future__ import annotations

import pytest

from src.utils.run_manager import RunManager


def test_create_run_generates_unique_id(tmp_path):
    mgr = RunManager(tmp_path / "runs")
    id1 = mgr.create_run("abc123", "snap1", 42, {})
    id2 = mgr.create_run("abc123", "snap1", 42, {})
    assert id1 != id2


def test_create_run_creates_directory(tmp_path):
    mgr = RunManager(tmp_path / "runs")
    run_id = mgr.create_run("abc123", "snap1", 42, {})
    assert (tmp_path / "runs" / run_id).exists()
    assert (tmp_path / "runs" / run_id / "metadata.json").exists()


def test_update_stage_persists(tmp_path):
    mgr = RunManager(tmp_path / "runs")
    run_id = mgr.create_run("abc123", "snap1", 42, {})
    mgr.start_stage(run_id, "psychometrics")
    meta = mgr.load_run(run_id)
    assert meta["stages"]["psychometrics"]["status"] == "running"


def test_complete_stage_records_duration(tmp_path):
    import time
    mgr = RunManager(tmp_path / "runs")
    run_id = mgr.create_run("abc123", "snap1", 42, {})
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc).isoformat()
    time.sleep(0.05)
    mgr.complete_stage(run_id, "psychometrics", started)
    meta = mgr.load_run(run_id)
    assert meta["stages"]["psychometrics"]["status"] == "completed"
    assert meta["stages"]["psychometrics"]["duration_seconds"] > 0


def test_artifact_manifest_written(tmp_path):
    from pathlib import Path
    mgr = RunManager(tmp_path / "runs")
    run_id = mgr.create_run("abc123", "snap1", 42, {})
    fake_artifact = tmp_path / "test_file.csv"
    fake_artifact.write_text("a,b\n1,2")
    mgr.register_artifact(run_id, "test_artifact", fake_artifact, "data", "test_stage")
    manifest = mgr.load_manifest(run_id)
    assert len(manifest["artifacts"]) == 1
    assert manifest["artifacts"][0]["name"] == "test_artifact"


def test_list_runs_returns_all(tmp_path):
    mgr = RunManager(tmp_path / "runs")
    id1 = mgr.create_run("abc", "s1", 42, {})
    id2 = mgr.create_run("def", "s2", 43, {})
    runs = mgr.list_runs()
    run_ids = [r["run_id"] for r in runs]
    assert id1 in run_ids
    assert id2 in run_ids
