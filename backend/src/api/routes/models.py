"""Model registry API routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_model_registry
from src.storage.model_registry import ModelRegistry

router = APIRouter(prefix="/api/models", tags=["models"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ModelResponse(BaseModel):
    id: str
    name: str
    task: str
    model_type: str
    version: int
    dataset_id: str | None = None
    dataset_version: int | None = None
    config: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    created_at: str
    status: str
    storage_path: str
    run_id: str | None = None
    base_model_id: str | None = None


class ModelLineageResponse(BaseModel):
    model_id: str
    chain: list[ModelResponse]


class ModelUpdateRequest(BaseModel):
    name: str | None = None


class ModelDeleteResponse(BaseModel):
    deleted: bool
    reason: str | None = None
    dependencies: dict[str, int] | None = None
    model_id: str | None = None


class ModelListResponse(BaseModel):
    models: list[ModelResponse]
    total: int
    page: int
    per_page: int


class RegisterModelRequest(BaseModel):
    name: str
    task: str
    model_type: str
    source_model_path: str
    source_metrics_path: str | None = None
    dataset_id: str | None = None
    dataset_version: int | None = None
    config: dict = Field(default_factory=dict)
    metrics: dict | None = None
    run_id: str | None = None


class CompareModelsRequest(BaseModel):
    model_ids: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=ModelListResponse)
async def list_models(
    task: str | None = Query(None),
    model_type: str | None = Query(None),
    dataset_id: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """List models with filters and pagination."""
    models, total = registry.list_models(
        task=task,
        model_type=model_type,
        dataset_id=dataset_id,
        sort=sort,
        order=order,
        page=page,
        per_page=per_page,
    )
    return ModelListResponse(
        models=[ModelResponse(**m.model_dump()) for m in models],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Get full detail for a model."""
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return ModelResponse(**model.model_dump())


@router.get("/{model_id}/model-card")
async def get_model_card(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Download the generated markdown model card for a registry model."""
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    card_path = (
        Path(model.storage_path).parent
        / "reports"
        / "model_cards"
        / f"{model.task}_{model.model_type}_model_card.md"
    )
    if not card_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model card not found for model: {model_id}",
        )

    return FileResponse(
        card_path,
        media_type="text/markdown",
        filename=card_path.name,
    )


@router.get("/{model_id}/versions", response_model=list[ModelResponse])
async def get_model_versions(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Get all versions of a model (same task+type+dataset)."""
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    versions = registry.get_model_versions(
        task=model.task,
        model_type=model.model_type,
        dataset_id=model.dataset_id,
    )
    return [ModelResponse(**v.model_dump()) for v in versions]


@router.get("/{model_id}/lineage", response_model=ModelLineageResponse)
async def get_model_lineage(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Return the ancestry chain for a fine-tuned model.

    Returns models from current back to root (model with no base_model_id).
    Useful for understanding the fine-tuning history and comparing metrics
    across generations.
    """
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    chain = registry.get_lineage(model_id)
    return ModelLineageResponse(
        model_id=model_id,
        chain=[ModelResponse(**m.model_dump()) for m in chain],
    )


@router.post("/register", response_model=ModelResponse)
async def register_model(
    req: RegisterModelRequest,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Register a trained model in the registry."""
    source_path = Path(req.source_model_path)
    if not source_path.exists():
        raise HTTPException(status_code=400, detail=f"Model file not found: {req.source_model_path}")

    metrics_path = Path(req.source_metrics_path) if req.source_metrics_path else None

    meta = registry.register_model(
        name=req.name,
        task=req.task,
        model_type=req.model_type,
        source_model_path=source_path,
        source_metrics_path=metrics_path,
        dataset_id=req.dataset_id,
        dataset_version=req.dataset_version,
        config=req.config,
        metrics=req.metrics,
        run_id=req.run_id,
    )
    return ModelResponse(**meta.model_dump())


@router.post("/compare", response_model=list[ModelResponse])
async def compare_models(
    req: CompareModelsRequest,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Get multiple models for side-by-side comparison."""
    if len(req.model_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models to compare")
    models = registry.compare_models(req.model_ids)
    return [ModelResponse(**m.model_dump()) for m in models]


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model_metadata(
    model_id: str,
    update: ModelUpdateRequest,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Update model metadata (name)."""
    model = registry.update_metadata(model_id, name=update.name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return ModelResponse(**model.model_dump())


@router.delete("/{model_id}", response_model=ModelDeleteResponse)
async def delete_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Delete a model (soft delete/archived). Returns dependency warnings if applicable."""
    try:
        result = registry.delete_model(model_id)
        return ModelDeleteResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
