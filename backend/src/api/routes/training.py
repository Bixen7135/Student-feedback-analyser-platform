"""Training API routes — launch and monitor classifier training jobs."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_dataset_manager, get_model_registry, get_db
from src.training import runner as training_runner
from src.training.config import TrainingConfig
from src.training.contract import (
    CLASSIFICATION_LOSS,
    TRAINING_PARAMETER_TABLE,
    VALID_MODEL_TYPES,
)

router = APIRouter(prefix="/api/training", tags=["training"])

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
_DEFAULT_DATA_PATH = _REPO_ROOT / "mnt/data/dataset.csv"


def _get_artifacts_dir() -> Path:
    base = Path(
        os.environ.get(
            "SFAP_TRAINING_DIR", str(_BACKEND_DIR / "training_runs")
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_data_path(override: str | None = None) -> Path:
    """Return the dataset path: request override -> env var -> repo default."""
    if override:
        return Path(override)
    env = os.environ.get("SFAP_DATA_PATH")
    return Path(env) if env else _DEFAULT_DATA_PATH


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
    pretrained_model: str | None = None
    max_seq_length: int | None = Field(None, gt=0)
    batch_size: int | None = Field(None, gt=0)
    epochs: int | None = Field(None, gt=0)
    learning_rate: float | None = Field(None, gt=0)
    weight_decay: float | None = Field(None, ge=0)
    warmup_ratio: float | None = Field(None, ge=0, le=1)
    gradient_accumulation_steps: int | None = Field(None, gt=0)
    head_hidden_units: int | None = Field(None, gt=0)
    dropout: float | None = Field(None, ge=0, lt=1)
    activation: str | None = Field(None, pattern="^(relu|gelu|tanh)$")
    text_col: str | None = None
    label_col: str | None = None


class StartTrainingRequest(BaseModel):
    dataset_id: str | None = None
    data_path: str | None = None
    task: str = Field(..., pattern="^(language|sentiment|detail_level)$")
    model_type: str = Field(..., pattern="^(tfidf|char_ngram|xlm_roberta)$")
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
    psychometrics_warning: str | None = None


class TrainingListResponse(BaseModel):
    jobs: list[TrainingJobResponse]
    total: int


class TrainingContractResponse(BaseModel):
    model_types: list[str]
    classification_loss: str
    parameters: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/contract", response_model=TrainingContractResponse)
async def get_training_contract():
    """Return the canonical training parameter contract table."""
    return TrainingContractResponse(
        model_types=sorted(VALID_MODEL_TYPES),
        classification_loss=CLASSIFICATION_LOSS,
        parameters=[dict(row) for row in TRAINING_PARAMETER_TABLE],
    )


@router.post("/start", response_model=TrainingJobResponse, status_code=202)
async def start_training(
    body: StartTrainingRequest,
    background_tasks: BackgroundTasks,
    dataset_manager=Depends(get_dataset_manager),
    model_registry=Depends(get_model_registry),
    db=Depends(get_db),
):
    """Launch a training job in the background.

    Returns 202 Accepted immediately; poll GET /api/training/{job_id}/status
    to track progress.
    """
    data_path: Path | None = None
    dataset_ref: str
    if body.dataset_id:
        ds = dataset_manager.get_dataset(body.dataset_id)
        if ds is None:
            raise HTTPException(
                status_code=404, detail=f"Dataset not found: {body.dataset_id}"
            )
        dataset_ref = body.dataset_id
        has_psychometrics = db.fetchone(
            "SELECT id FROM pipeline_runs WHERE dataset_id = ? AND status = 'completed' LIMIT 1",
            (body.dataset_id,),
        )
        psychometrics_warning = None if has_psychometrics else (
            "No completed pipeline run found for this dataset. "
            "Psychometrics has not been validated. See Scientific Spec."
        )
    else:
        data_path = _resolve_data_path(body.data_path)
        if not data_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Training dataset not found: {data_path}"
            )
        dataset_ref = str(data_path)
        psychometrics_warning = (
            "Training is using a CSV file directly. "
            "No dataset record or psychometrics history is linked."
        )

    job_id = f"job_{uuid.uuid4().hex[:16]}"

    config_payload = body.config.model_dump(exclude_none=True) if body.config else {}
    config = TrainingConfig(
        **config_payload,
        base_model_id=body.base_model_id,
    )
    try:
        serialized_config = config.to_dict(model_type=body.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job = training_runner.create_job(
        job_id=job_id,
        dataset_id=dataset_ref,
        task=body.task,
        model_type=body.model_type,
        dataset_version=body.dataset_version if body.dataset_id else None,
        branch_id=body.branch_id if body.dataset_id else None,
        seed=body.seed,
        name=body.name,
        base_model_id=body.base_model_id,
        config=serialized_config,
        db=db,
    )

    background_tasks.add_task(
        training_runner.run_job_background,
        job_id=job_id,
        dataset_id=body.dataset_id,
        data_path=str(data_path) if data_path else None,
        task=body.task,
        model_type=body.model_type,
        dataset_manager=dataset_manager,
        model_registry=model_registry,
        artifacts_dir=_get_artifacts_dir(),
        config=config,
        dataset_version=body.dataset_version if body.dataset_id else None,
        branch_id=body.branch_id if body.dataset_id else None,
        seed=body.seed,
        name=body.name,
        base_model_id=body.base_model_id,
        db=db,
    )

    return TrainingJobResponse(**job, psychometrics_warning=psychometrics_warning)


@router.get("/", response_model=TrainingListResponse)
async def list_training_jobs(
    task: str | None = Query(None),
    status: str | None = Query(None),
    db=Depends(get_db),
):
    """List all training jobs (most recent first, DB-backed)."""
    jobs = training_runner.list_jobs_from_db(db=db, task=task, status=status)
    return TrainingListResponse(
        jobs=[TrainingJobResponse(**j) for j in jobs],
        total=len(jobs),
    )


@router.get("/{job_id}/status", response_model=TrainingJobResponse)
async def get_training_status(job_id: str, db=Depends(get_db)):
    """Poll the status of a training job."""
    job = training_runner.get_job(job_id)
    if job is None:
        # Fall back to DB for jobs that survived a restart
        db_jobs = training_runner.list_jobs_from_db(db=db)
        job = next((j for j in db_jobs if j["job_id"] == job_id), None)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Training job not found: {job_id}"
        )
    return TrainingJobResponse(**job)


@router.get("/{job_id}/result", response_model=TrainingJobResponse)
async def get_training_result(job_id: str, db=Depends(get_db)):
    """Get training result including metrics and registered model info.

    Returns 409 if the job has not yet completed or failed.
    """
    job = training_runner.get_job(job_id)
    if job is None:
        db_jobs = training_runner.list_jobs_from_db(db=db)
        job = next((j for j in db_jobs if j["job_id"] == job_id), None)
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
