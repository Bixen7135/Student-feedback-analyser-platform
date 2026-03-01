"""Model registry API routes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.analysis import runner as analysis_runner
from src.analytics.explain import explain_text_instance
from src.analytics.feature_importance import get_global_feature_importance
from src.api.dependencies import get_dataset_manager, get_db, get_model_registry
from src.inference.engine import check_compatibility
from src.inference.signature import canonicalize_signature
from src.schema import DatasetSchemaSnapshot, normalize_column_name, resolve_roles
from src.storage.dataset_manager import DatasetManager
from src.storage.model_registry import ModelRegistry
from src.storage.models import ModelMeta

router = APIRouter(prefix="/api/models", tags=["models"])

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent


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
    input_signature: dict = Field(default_factory=dict)
    preprocess_spec: dict = Field(default_factory=dict)
    training_profile: dict = Field(default_factory=dict)
    run_source: Literal["pipeline", "training", "unknown"] = "unknown"


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
    input_signature: dict | None = None
    preprocess_spec: dict | None = None
    training_profile: dict | None = None


class CompareModelsRequest(BaseModel):
    model_ids: list[str]


class ModelCompatibilityResponse(BaseModel):
    ok: bool
    reasons: list[dict] = Field(default_factory=list)
    resolved_columns: dict = Field(default_factory=dict)
    required_roles: list[str] = Field(default_factory=list)
    preprocess_spec_id: str | None = None
    label_schema: dict = Field(default_factory=dict)
    schema_columns: list[str] = Field(default_factory=list)
    text_col_used: str | None = None
    label_col_used: str | None = None


class ExplainModelRequest(BaseModel):
    text: str | None = None
    analysis_id: str | None = None
    row_idx: int | None = None
    top_n: int = 10


def _derive_run_source(model: ModelMeta) -> Literal["pipeline", "training", "unknown"]:
    """Classify the origin of the model from persisted lineage fields."""
    if model.job_id:
        return "training"
    if not model.run_id:
        return "unknown"
    if model.run_id.startswith("run_"):
        return "pipeline"
    if model.run_id.startswith("training_"):
        return "training"
    return "unknown"


def _to_model_response(model: ModelMeta) -> ModelResponse:
    return ModelResponse(
        **model.model_dump(exclude={"job_id"}),
        run_source=_derive_run_source(model),
    )


def _get_analysis_artifacts_dir() -> Path:
    base = Path(
        os.environ.get(
            "SFAP_ANALYSIS_DIR", str(_BACKEND_DIR / "analysis_runs")
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/", response_model=ModelListResponse)
async def list_models(
    task: str | None = Query(None),
    model_type: str | None = Query(None),
    dataset_id: str | None = Query(None),
    run_id: str | None = Query(None),
    include_archived: bool = Query(False),
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
        run_id=run_id,
        include_archived=include_archived,
        sort=sort,
        order=order,
        page=page,
        per_page=per_page,
    )
    return ModelListResponse(
        models=[_to_model_response(m) for m in models],
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
    return _to_model_response(model)


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
    return [_to_model_response(v) for v in versions]


@router.get("/{model_id}/compatibility", response_model=ModelCompatibilityResponse)
async def get_model_compatibility(
    model_id: str,
    dataset_id: str = Query(...),
    dataset_version: int | None = Query(None),
    branch_id: str | None = Query(None),
    registry: ModelRegistry = Depends(get_model_registry),
    dataset_manager: DatasetManager = Depends(get_dataset_manager),
):
    """Evaluate whether a model can run against a target dataset/version."""
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    try:
        df = dataset_manager.get_dataframe(
            dataset_id=dataset_id,
            version=dataset_version,
            branch_id=branch_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    resolved_columns = resolve_roles(df=df, column_roles={}, overrides={})
    dataset_schema = DatasetSchemaSnapshot(
        columns=tuple(str(col) for col in df.columns),
        normalized_columns=tuple(normalize_column_name(col) for col in df.columns),
        column_roles={},
    )
    report = check_compatibility(
        model_meta=model,
        dataset_schema=dataset_schema,
        resolved_columns=resolved_columns,
    )
    return ModelCompatibilityResponse(**report)


@router.get("/{model_id}/importance", response_model=dict)
async def get_model_importance(
    model_id: str,
    top_n: int = Query(20, ge=1, le=100),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Return global feature importance for supported linear text models."""
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    try:
        return get_global_feature_importance(model, top_n=top_n)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/{model_id}/explain", response_model=dict)
async def explain_model(
    model_id: str,
    req: ExplainModelRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    db=Depends(get_db),
):
    """Return a local explanation for one free-text input or an analysis row."""
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    text = req.text
    source: dict[str, Any] = {"type": "text"}
    if text is None:
        if not req.analysis_id or req.row_idx is None:
            raise HTTPException(
                status_code=422,
                detail="Provide either {text} or {analysis_id, row_idx}.",
            )
        analysis = analysis_runner.get_analysis_from_db(db, req.analysis_id)
        if analysis is None:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found: {req.analysis_id}",
            )
        results_path = analysis_runner.get_results_path(_get_analysis_artifacts_dir(), req.analysis_id)
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Results file not found.")
        df = analysis_runner.load_filtered_df(
            artifacts_dir=_get_analysis_artifacts_dir(),
            analysis_id=req.analysis_id,
        )
        if df is None:
            raise HTTPException(status_code=404, detail="Results file not found.")
        if req.row_idx < 0 or req.row_idx >= len(df):
            raise HTTPException(status_code=422, detail="row_idx is out of range.")

        summary = analysis.get("result_summary") or {}
        signature = canonicalize_signature(model)
        preferred_cols = [
            str(signature["text"].get("source_column") or ""),
            str(summary.get("text_col") or ""),
            "text_feedback",
            "text",
        ]
        text_col = next((col for col in preferred_cols if col and col in df.columns), None)
        if text_col is None:
            raise HTTPException(
                status_code=422,
                detail="No text column found in the referenced analysis results.",
            )
        text = str(df.iloc[req.row_idx][text_col])
        source = {
            "type": "analysis_row",
            "analysis_id": req.analysis_id,
            "row_idx": req.row_idx,
            "text_col": text_col,
        }

    try:
        explanation = explain_text_instance(model, text=str(text or ""), top_n=req.top_n)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        **explanation,
        "source": source,
    }


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
        chain=[_to_model_response(m) for m in chain],
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
        input_signature=req.input_signature,
        preprocess_spec=req.preprocess_spec,
        training_profile=req.training_profile,
    )
    return _to_model_response(meta)


@router.post("/compare", response_model=list[ModelResponse])
async def compare_models(
    req: CompareModelsRequest,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Get multiple models for side-by-side comparison."""
    if len(req.model_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models to compare")
    models = registry.compare_models(req.model_ids)
    return [_to_model_response(m) for m in models]


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
    return _to_model_response(model)


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
