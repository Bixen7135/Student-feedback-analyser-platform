"""Run lifecycle management — creation, status tracking, artifact manifests."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from src.utils.logging import get_logger

log = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class RunManager:
    """Manages run directories, metadata, stage statuses, and artifact manifests."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run creation
    # ------------------------------------------------------------------

    def create_run(
        self,
        config_hash: str,
        data_snapshot_id: str,
        random_seed: int,
        system_info: dict[str, Any],
    ) -> str:
        """Create a new run directory and write initial metadata. Returns run_id."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        run_id = f"run_{ts}_{config_hash[:8]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        for sub in ["raw", "stages", "psychometrics", "splits", "text_tasks", "fusion",
                    "contradiction", "evaluation", "reports"]:
            (run_dir / sub).mkdir(exist_ok=True)

        metadata: dict[str, Any] = {
            "run_id": run_id,
            "created_at": _utcnow(),
            "config_hash": config_hash,
            "data_snapshot_id": data_snapshot_id,
            "git_commit": _get_git_commit(),
            "random_seed": random_seed,
            "system_info": system_info,
            "stages": {},
        }
        self._write_json(run_dir / "metadata.json", metadata)
        self._write_json(run_dir / "artifact_manifest.json", {
            "run_id": run_id,
            "created_at": _utcnow(),
            "artifacts": [],
        })
        log.info("run_created", run_id=run_id, run_dir=str(run_dir))
        return run_id

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def start_stage(self, run_id: str, stage: str) -> None:
        """Mark a stage as running."""
        self._update_stage(run_id, stage, {
            "name": stage,
            "status": "running",
            "started_at": _utcnow(),
            "completed_at": None,
            "duration_seconds": None,
            "error": None,
        })

    def complete_stage(self, run_id: str, stage: str, started_at: str) -> None:
        """Mark a stage as completed and compute duration."""
        started = datetime.fromisoformat(started_at)
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        self._update_stage(run_id, stage, {
            "name": stage,
            "status": "completed",
            "started_at": started_at,
            "completed_at": _utcnow(),
            "duration_seconds": round(duration, 2),
            "error": None,
        })

    def fail_stage(self, run_id: str, stage: str, error: str) -> None:
        """Mark a stage as failed."""
        self._update_stage(run_id, stage, {
            "name": stage,
            "status": "failed",
            "started_at": None,
            "completed_at": _utcnow(),
            "duration_seconds": None,
            "error": error,
        })

    def _update_stage(self, run_id: str, stage: str, status_data: dict) -> None:
        run_dir = self._run_dir(run_id)
        # Write stage-specific file
        stage_file = run_dir / "stages" / f"{stage}.json"
        self._write_json(stage_file, status_data)
        # Update metadata stages map
        metadata = self._load_json(run_dir / "metadata.json")
        metadata["stages"][stage] = status_data
        self._write_json(run_dir / "metadata.json", metadata)
        log.info("stage_updated", run_id=run_id, stage=stage, status=status_data["status"])

    # ------------------------------------------------------------------
    # Artifact manifest
    # ------------------------------------------------------------------

    def register_artifact(
        self,
        run_id: str,
        name: str,
        path: Path,
        artifact_type: str,
        stage: str,
    ) -> None:
        """Add an artifact to the run's manifest."""
        run_dir = self._run_dir(run_id)
        manifest = self._load_json(run_dir / "artifact_manifest.json")
        try:
            rel_path = str(path.relative_to(run_dir))
        except ValueError:
            rel_path = str(path)
        size = path.stat().st_size if path.exists() else None
        manifest["artifacts"].append({
            "name": name,
            "path": rel_path,
            "type": artifact_type,
            "stage": stage,
            "size_bytes": size,
            "sha256": None,
            "created_at": _utcnow(),
        })
        self._write_json(run_dir / "artifact_manifest.json", manifest)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def load_run(self, run_id: str) -> dict[str, Any]:
        return self._load_json(self._run_dir(run_id) / "metadata.json")

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        return self._load_json(self._run_dir(run_id) / "artifact_manifest.json")

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for d in sorted(self.runs_dir.iterdir()):
            meta_path = d / "metadata.json"
            if d.is_dir() and meta_path.exists():
                runs.append(self._load_json(meta_path))
        return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)

    def run_exists(self, run_id: str) -> bool:
        return (self.runs_dir / run_id / "metadata.json").exists()

    def delete_run(self, run_id: str) -> None:
        """Permanently remove a run directory and all its artifacts."""
        import shutil
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        shutil.rmtree(run_dir)
        log.info("run_deleted", run_id=run_id)

    def get_run_dir(self, run_id: str) -> Path:
        return self._run_dir(run_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    @staticmethod
    def _load_json(path: Path) -> Any:
        return orjson.loads(path.read_bytes())
