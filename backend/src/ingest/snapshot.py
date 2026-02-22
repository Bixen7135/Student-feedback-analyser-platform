"""Dataset snapshotting — copy raw data to run directory with SHA256 verification."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import orjson

from src.utils.reproducibility import hash_file
from src.utils.logging import get_logger

log = get_logger(__name__)


def create_snapshot(source_path: Path, run_dir: Path) -> str:
    """
    Copy the raw dataset into runs/<run_id>/raw/ and return its SHA256 hash.
    The hash serves as the data_snapshot_id for reproducibility tracking.
    """
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest = raw_dir / source_path.name
    shutil.copy2(source_path, dest)

    snapshot_id = hash_file(dest)
    metadata = {
        "source_path": str(source_path),
        "snapshot_path": str(dest),
        "sha256": snapshot_id,
        "snapshot_created_at": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": dest.stat().st_size,
    }
    (raw_dir / "snapshot_metadata.json").write_bytes(
        orjson.dumps(metadata, option=orjson.OPT_INDENT_2)
    )
    log.info("snapshot_created", snapshot_id=snapshot_id, dest=str(dest))
    return snapshot_id
