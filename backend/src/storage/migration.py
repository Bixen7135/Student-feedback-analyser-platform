"""First-startup migration — backfill existing runs and register default dataset."""
from __future__ import annotations

from pathlib import Path

from src.storage.database import Database
from src.storage.dataset_manager import DatasetManager
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_migration(db: Database, datasets_dir: Path, runs_dir: Path, default_data_path: Path | None = None) -> None:
    """Run all migrations. Safe to call multiple times (idempotent)."""
    _register_default_dataset(db, datasets_dir, default_data_path)
    _backfill_runs(db, runs_dir)


def _register_default_dataset(db: Database, datasets_dir: Path, data_path: Path | None) -> None:
    """Register the default dataset (mnt/data/dataset.csv) if not already registered."""
    if data_path is None or not data_path.exists():
        log.info("migration_skip_default_dataset", reason="file not found")
        return

    # Check if already registered by sha256
    import hashlib
    h = hashlib.sha256()
    with open(data_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    sha = h.hexdigest()

    existing = db.fetchone("SELECT id FROM datasets WHERE sha256 = ?", (sha,))
    if existing:
        log.info("migration_default_dataset_exists", id=existing["id"])
        return

    mgr = DatasetManager(db, datasets_dir)
    meta = mgr.upload_dataset(
        file_path=data_path,
        name="Student Feedback Survey (Default)",
        description="Original student feedback dataset with 18,476 responses in Russian and Kazakh.",
        tags=["default", "survey", "ru", "kz"],
        author="system",
    )
    log.info("migration_default_dataset_registered", id=meta.id, rows=meta.row_count)


def _backfill_runs(db: Database, runs_dir: Path) -> None:
    """Scan existing runs/ and backfill the analysis_runs table."""
    import orjson

    if not runs_dir.exists():
        return

    for run_dir in sorted(runs_dir.iterdir()):
        meta_path = run_dir / "metadata.json"
        if not run_dir.is_dir() or not meta_path.exists():
            continue

        run_id = run_dir.name
        existing = db.fetchone("SELECT id FROM analysis_runs WHERE run_id = ?", (run_id,))
        if existing:
            continue

        meta = orjson.loads(meta_path.read_bytes())
        created_at = meta.get("created_at", "")

        # Derive status from stages
        stages = meta.get("stages", {})
        if any(s.get("status") == "failed" for s in stages.values()):
            status = "failed"
        elif all(s.get("status") == "completed" for s in stages.values()) and stages:
            status = "completed"
        elif any(s.get("status") == "running" for s in stages.values()):
            status = "running"
        else:
            status = "pending"

        import uuid
        db.execute(
            """INSERT INTO analysis_runs
            (id, name, description, tags, comments, dataset_id, dataset_version,
             model_ids, created_at, status, run_id, result_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), f"Legacy run {run_id}", "", "[]", "",
             None, None, "[]", created_at, status, run_id, "{}"),
        )

    db.commit()
    log.info("migration_backfill_runs_complete")
