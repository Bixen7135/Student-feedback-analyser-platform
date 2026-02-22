"""Artifact and report download API routes."""
from __future__ import annotations

from pathlib import Path

import orjson  # type: ignore
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.api.dependencies import get_run_manager
from src.api.schemas import ArtifactManifestResponse, ArtifactResponse
from src.utils.run_manager import RunManager

router = APIRouter(prefix="/api/runs/{run_id}/artifacts", tags=["artifacts"])


@router.get("/", response_model=ArtifactManifestResponse)
async def list_artifacts(run_id: str, mgr: RunManager = Depends(get_run_manager)):
    """List all artifacts for a run."""
    if not mgr.run_exists(run_id):
        raise HTTPException(404, f"Run not found: {run_id}")
    manifest = mgr.load_manifest(run_id)
    artifacts = [ArtifactResponse(**a) for a in manifest.get("artifacts", [])]
    return ArtifactManifestResponse(run_id=run_id, artifacts=artifacts)


@router.get("/reports/{report_name}")
async def download_report(
    run_id: str,
    report_name: str,
    mgr: RunManager = Depends(get_run_manager),
):
    """Download a report file (markdown) from a run."""
    if not mgr.run_exists(run_id):
        raise HTTPException(404, f"Run not found: {run_id}")
    rdir = mgr.get_run_dir(run_id)

    # Sanitize filename
    safe_name = Path(report_name).name
    report_path = rdir / "reports" / safe_name

    if not report_path.exists():
        # Try model cards subdirectory
        report_path = rdir / "reports" / "model_cards" / safe_name

    if not report_path.exists():
        raise HTTPException(404, f"Report not found: {report_name}")

    return FileResponse(
        path=str(report_path),
        filename=safe_name,
        media_type="text/markdown",
    )


@router.get("/{artifact_path:path}")
async def download_artifact(
    run_id: str,
    artifact_path: str,
    mgr: RunManager = Depends(get_run_manager),
):
    """Download any artifact file from a run by its relative path."""
    if not mgr.run_exists(run_id):
        raise HTTPException(404, f"Run not found: {run_id}")
    rdir = mgr.get_run_dir(run_id)

    # Security: prevent path traversal
    try:
        full_path = (rdir / artifact_path).resolve()
        full_path.relative_to(rdir.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")

    if not full_path.exists():
        raise HTTPException(404, f"Artifact not found: {artifact_path}")

    return FileResponse(path=str(full_path), filename=full_path.name)
