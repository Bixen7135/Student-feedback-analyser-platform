"""Training API routes — launch and monitor classifier training jobs."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_dataset_manager, get_model_registry
from src.training import runner as training_runner
from src.training.config import TrainingConfig

router = APIRouter(prefix="/api/training", tags=["training"])

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent


def _get_artifacts_dir() -> Path:
    base = Path(
        os.environ.get(
            "SFAP_TRAINING_DIR", str(_BACKEND_DIR / "training_runs")
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class TrainingConfigRequest(BaseModel):
    train_ratio: float = Field(0.80, ge=0.5, le=0.95)
    val_ratio: float = Field(0.10, ge=0.02, le=0.30)
    test_ratio: float = Field(0.10, ge=0.02, le=0.30)
    class_balancing: str = Field(
        "class_weight",
        pattern="^(none|oversample|class_weight)$",
    )
    max_features: int | None = Field(None, gt=0)
    C: float | None = Field(None, gt=0)
    max_iter: int | None = Field(None, gt=0)
    text_col: str | None = None
    label_col: str | None = None


class StartTrainingRequest(BaseModel):
    dataset_id: str
    task: str = Field(..., pattern="^(language|sentiment|detail_level)$")
    model_type: str = Field(..., pattern="^(tfidf|char_ngram)$")
    config: TrainingConfigRequest | None = None
    dataset_version: int | None = None
    branch_id: str | None = None
    seed: int = Field(42, ge=0)
    name: str | None = None
    base_model_id: str | None = None


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    dataset_id: str
    dataset_version: int | None
    branch_id: str | None
    task: str
    model_type: str
    seed: int
    name: str | None
    base_model_id: str | None = None
    started_at: str | None
    completed_at: str | None
    error: str | None
    model_id: str | None
    model_name: str | None
    model_version: int | None
    metrics: dict | None
    config: dict | None


class TrainingListResponse(BaseModel):
    jobs: list[TrainingJobResponse]
    total: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start", response_model=TrainingJobResponse, status_code=202)
async def start_training(
    body: StartTrainingRequest,
    background_tasks: BackgroundTasks,
    dataset_manager=Depends(get_dataset_manager),
    model_registry=Depends(get_model_registry),
):
    """Launch a training job in the background.

    Returns 202 Accepted immediately; poll GET /api/training/{job_id}/status
    to track progress.
    """
    ds = dataset_manager.get_dataset(body.dataset_id)
    if ds is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {body.dataset_id}"
        )

    job_id = f"job_{uuid.uuid4().hex[:16]}"

    config: TrainingConfig | None = None
    if body.config:
        cfg = body.config
        config = TrainingConfig(
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            class_balancing=cfg.class_balancing,
            max_features=cfg.max_features,
            C=cfg.C,
            max_iter=cfg.max_iter,
            text_col=cfg.text_col,
            label_col=cfg.label_col,
        )

    job = training_runner.create_job(
        job_id=job_id,
        dataset_id=body.dataset_id,
        task=body.task,
        model_type=body.model_type,
        dataset_version=body.dataset_version,
        branch_id=body.branch_id,
        seed=body.seed,
        name=body.name,
        base_model_id=body.base_model_id,
    )

    background_tasks.add_task(
        training_runner.run_job_background,
        job_id=job_id,
        dataset_id=body.dataset_id,
        task=body.task,
        model_type=body.model_type,
        dataset_manager=dataset_manager,
        model_registry=model_registry,
        artifacts_dir=_get_artifacts_dir(),
        config=config,
        dataset_version=body.dataset_version,
        branch_id=body.branch_id,
        seed=body.seed,
        name=body.name,
        base_model_id=body.base_model_id,
    )

    return TrainingJobResponse(**job)


@router.get("/", response_model=TrainingListResponse)
async def list_training_jobs(
    task: str | None = Query(None),
    status: str | None = Query(None),
):
    """List all training jobs (most recent first)."""
    jobs = training_runner.list_jobs()
    if task:
        jobs = [j for j in jobs if j["task"] == task]
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    return TrainingListResponse(
        jobs=[TrainingJobResponse(**j) for j in jobs],
        total=len(jobs),
    )


@router.get("/{job_id}/status", response_model=TrainingJobResponse)
async def get_training_status(job_id: str):
    """Poll the status of a training job."""
    job = training_runner.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Training job not found: {job_id}"
        )
    return TrainingJobResponse(**job)


@router.get("/{job_id}/result", response_model=TrainingJobResponse)
async def get_training_result(job_id: str):
    """Get training result including metrics and registered model info.

    Returns 409 if the job has not yet completed or failed.
    """
    job = training_runner.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Training job not found: {job_id}"
        )
    if job["status"] not in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job not yet finished. Current status: {job['status']}",
        )
    return TrainingJobResponse(**job)
