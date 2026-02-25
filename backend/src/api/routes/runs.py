"""Run management API routes."""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import orjson  # type: ignore
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.dependencies import get_run_manager, get_dataset_manager, get_db, get_model_registry
from src.api.schemas import (
    CreateRunRequest,
    RunDetailResponse,
    RunSummaryResponse,
    StageStatusResponse,
)
from src.utils.run_manager import RunManager

router = APIRouter(prefix="/api/runs", tags=["runs"])

# ---------------------------------------------------------------------------
# Absolute paths resolved from this file's location so they work regardless
# of which directory the server is started from.
# File layout: backend/src/api/routes/runs.py
#   parents[0] = backend/src/api/routes/
#   parents[1] = backend/src/api/
#   parents[2] = backend/src/
#   parents[3] = backend/          ← _BACKEND_DIR
#   parents[4] = <repo root>       ← _REPO_ROOT
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parents[3]
_REPO_ROOT   = _BACKEND_DIR.parent

_DEFAULT_DATA_PATH   = _REPO_ROOT   / "mnt/data/dataset.csv"
_DEFAULT_CONFIG_PATH = _BACKEND_DIR / "configs/experiment.yaml"
_DEFAULT_FACTOR_PATH = _BACKEND_DIR / "configs/factor_structure.yaml"


def _resolve_data_path(override: str | None = None) -> Path:
    """Return the dataset path: CLI override → env var → absolute default."""
    if override:
        return Path(override)
    env = os.environ.get("SFAP_DATA_PATH")
    return Path(env) if env else _DEFAULT_DATA_PATH


def _stage_to_response(stage_data: dict) -> StageStatusResponse:
    return StageStatusResponse(**{k: stage_data.get(k) for k in StageStatusResponse.model_fields})


def _run_to_summary(meta: dict) -> RunSummaryResponse:
    stages = {k: _stage_to_response(v) for k, v in meta.get("stages", {}).items()}
    return RunSummaryResponse(
        run_id=meta["run_id"],
        created_at=meta["created_at"],
        config_hash=meta.get("config_hash", ""),
        data_snapshot_id=meta.get("data_snapshot_id", ""),
        random_seed=meta.get("random_seed", 42),
        stages=stages,
        dataset_id=meta.get("dataset_id"),
        branch_id=meta.get("branch_id"),
        dataset_version=meta.get("dataset_version"),
        name=meta.get("name"),
    )


def _run_to_detail(meta: dict) -> RunDetailResponse:
    summary = _run_to_summary(meta)
    return RunDetailResponse(
        **summary.model_dump(),
        git_commit=meta.get("git_commit"),
        system_info=meta.get("system_info"),
    )


@router.get("/", response_model=list[RunSummaryResponse])
async def list_runs(mgr: RunManager = Depends(get_run_manager)):
    """List all runs in reverse chronological order."""
    runs = mgr.list_runs()
    return [_run_to_summary(r) for r in runs]


@router.post("/", response_model=RunSummaryResponse)
async def create_run(
    request: CreateRunRequest,
    background_tasks: BackgroundTasks,
    mgr: RunManager = Depends(get_run_manager),
    db=Depends(get_db),
):
    """Create a new run and optionally launch the full pipeline in background."""
    import hashlib
    from src.config import get_system_info
    from src.utils.reproducibility import hash_file

    config_path = Path(request.config_path) if request.config_path else _DEFAULT_CONFIG_PATH
    config_hash = hash_file(config_path)[:16] if config_path.exists() else "unknown"

    if request.dataset_id:
        # Load DF from DatasetManager and derive snapshot_id from content hash
        dm = get_dataset_manager()
        try:
            df = dm.get_dataframe(
                request.dataset_id,
                version=request.dataset_version,
                branch_id=request.branch_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {exc}")
        data_snapshot_id = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
    else:
        data_path        = _resolve_data_path(request.data_path)
        data_snapshot_id = hash_file(data_path)[:16] if data_path.exists() else "unknown"

    run_id = mgr.create_run(
        config_hash=config_hash,
        data_snapshot_id=data_snapshot_id,
        random_seed=request.seed,
        system_info=get_system_info(),
        dataset_id=request.dataset_id,
        dataset_version=request.dataset_version,
        branch_id=request.branch_id,
        name=request.name,
        db=db,
    )
    meta = mgr.load_run(run_id)
    return _run_to_summary(meta)


@router.get("/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str, mgr: RunManager = Depends(get_run_manager)):
    """Get full detail for a specific run."""
    if not mgr.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    meta = mgr.load_run(run_id)
    return _run_to_detail(meta)


@router.post("/{run_id}/stages/{stage_name}/start")
async def start_stage(
    run_id: str,
    stage_name: str,
    background_tasks: BackgroundTasks,
    mgr: RunManager = Depends(get_run_manager),
    db=Depends(get_db),
    model_registry=Depends(get_model_registry),
):
    """Start a pipeline stage for a run (runs synchronously via background task)."""
    if not mgr.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    valid_stages = [
        "ingest_preprocess", "psychometrics", "splits",
        "text_tasks", "fusion", "contradiction", "evaluation", "reporting", "run_full",
    ]
    if stage_name not in valid_stages:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {stage_name}")

    # For run_full, launch in background
    if stage_name == "run_full":
        background_tasks.add_task(_run_full_background, run_id, mgr, db, model_registry)
        return {"status": "started", "run_id": run_id, "stage": stage_name}

    return {"status": "accepted", "run_id": run_id, "stage": stage_name}


@router.delete("/{run_id}", status_code=204)
async def delete_run(run_id: str, mgr: RunManager = Depends(get_run_manager)):
    """Delete a run and all its artifacts. Refuses if the run is currently running."""
    if not mgr.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    meta = mgr.load_run(run_id)
    if any(s.get("status") == "running" for s in meta.get("stages", {}).values()):
        raise HTTPException(status_code=409, detail="Cannot delete a run that is currently running.")
    mgr.delete_run(run_id)


@router.get("/{run_id}/stages/{stage_name}/status")
async def get_stage_status(
    run_id: str,
    stage_name: str,
    mgr: RunManager = Depends(get_run_manager),
):
    """Get status of a specific stage."""
    if not mgr.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    meta = mgr.load_run(run_id)
    stage = meta.get("stages", {}).get(stage_name)
    if not stage:
        return {"name": stage_name, "status": "pending"}
    return stage


def _run_full_background(run_id: str, mgr: RunManager, db, model_registry) -> None:
    """Background task to run the full pipeline for an existing run.

    Must be a plain (non-async) function so FastAPI runs it in a thread-pool
    worker rather than in the event loop.  Running synchronous pipeline code
    (pandas / sklearn) inside an async function would block the entire event
    loop and prevent the server from responding to polling requests.
    """
    from src.pipeline import run_full_pipeline

    try:
        meta = mgr.load_run(run_id)
        seed = meta.get("random_seed", 42)

        config_path = _DEFAULT_CONFIG_PATH
        factor_path = _DEFAULT_FACTOR_PATH

        dataset_id      = meta.get("dataset_id")
        dataset_version = meta.get("dataset_version")
        branch_id       = meta.get("branch_id")

        if dataset_id:
            # Load DF from the versioned dataset store
            dm = get_dataset_manager()
            df = dm.get_dataframe(dataset_id, version=dataset_version, branch_id=branch_id)
            run_full_pipeline(
                data_path=None,
                config_path=config_path,
                factor_structure_path=factor_path,
                runs_dir=mgr.runs_dir,
                seed=seed,
                existing_run_id=run_id,
                df_raw=df,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                branch_id=branch_id,
                model_registry=model_registry,
                db=db,
            )
        else:
            data_path = _resolve_data_path()
            run_full_pipeline(
                data_path=data_path,
                config_path=config_path,
                factor_structure_path=factor_path,
                runs_dir=mgr.runs_dir,
                seed=seed,
                existing_run_id=run_id,
                model_registry=model_registry,
                db=db,
            )
    except Exception:
        traceback.print_exc()
        # If the pipeline crashed before starting any stage, record the failure
        # on ingest_preprocess so the frontend's allDone() detects it and stops
        # polling instead of waiting forever.
        try:
            meta = mgr.load_run(run_id)
            if not meta.get("stages"):
                mgr.fail_stage(run_id, "ingest_preprocess", traceback.format_exc())
        except Exception:
            pass  # best-effort; original traceback already printed above
