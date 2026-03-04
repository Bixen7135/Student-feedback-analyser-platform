"""Analytics API routes for descriptive statistics and diagnostics."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.analysis import runner as analysis_runner
from src.analytics.clustering import cluster_points
from src.analytics.correlations import mixed_pairwise_correlations
from src.analytics.descriptive import descriptive_summary
from src.analytics.diagnostics import classification_diagnostics
from src.analytics.embeddings_service import compute_or_load_embeddings
from src.analytics.outliers import detect_outliers
from src.api.dependencies import get_dataset_manager, get_db, get_model_registry
from src.inference.signature import canonicalize_signature
from src.storage.dataset_manager import DatasetManager
from src.storage.model_registry import ModelRegistry

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent


class EmbeddingsRequest(BaseModel):
    model_id: str | None = None
    reuse_cached: bool = True
    max_features: int = 512


class ClusterRequest(BaseModel):
    method: str = "kmeans"
    k: int = 3
    eps: float = 0.5
    min_samples: int = 5
    model_id: str | None = None
    reuse_embeddings: bool = True


class OutliersRequest(BaseModel):
    method: str = "isolation_forest"
    contamination: float = 0.1
    n_neighbors: int = 5
    model_id: str | None = None
    reuse_embeddings: bool = True


def _get_artifacts_dir() -> Path:
    base = Path(
        os.environ.get(
            "SFAP_ANALYSIS_DIR", str(_BACKEND_DIR / "analysis_runs")
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _get_analysis_or_job(db: Any, analysis_id: str) -> dict[str, Any]:
    analysis = analysis_runner.get_analysis_from_db(db, analysis_id)
    if analysis is not None:
        return analysis

    job = analysis_runner.get_job(analysis_id)
    if job is not None:
        return job

    raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")


def _parse_filters(filters: str | None) -> list[dict] | None:
    if not filters:
        return None
    try:
        parsed = orjson.loads(filters)
        if not isinstance(parsed, list):
            raise ValueError("filters must be a JSON array")
        return parsed
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid filters JSON: {exc}")


def _apply_result_filters(
    df: pd.DataFrame,
    *,
    filter_col: str | None = None,
    filter_val: str | None = None,
    filters: list[dict] | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    if filters:
        for rule in filters:
            col = str(rule.get("col", "")).strip()
            op = str(rule.get("op", "eq")).strip()
            val = str(rule.get("val", ""))
            if not col or col not in df.columns:
                continue
            col_str = df[col].fillna("").astype(str)
            if op == "eq":
                df = df[col_str.str.lower() == val.lower()]
            elif op == "ne":
                df = df[col_str.str.lower() != val.lower()]
            elif op == "contains":
                df = df[
                    col_str.str.lower().str.contains(
                        val.lower(), na=False, regex=False
                    )
                ]
            elif op in {"gt", "lt", "gte", "lte"}:
                try:
                    num_col = pd.to_numeric(df[col], errors="coerce")
                    num_val = float(val)
                    if op == "gt":
                        df = df[num_col > num_val]
                    elif op == "lt":
                        df = df[num_col < num_val]
                    elif op == "gte":
                        df = df[num_col >= num_val]
                    elif op == "lte":
                        df = df[num_col <= num_val]
                except (TypeError, ValueError):
                    continue
    elif filter_col and filter_val and filter_col in df.columns:
        df = df[df[filter_col].fillna("").astype(str).str.lower() == filter_val.lower()]

    if search and search.strip():
        text_cols = [
            col
            for col in df.columns
            if df[col].dtype == object or df[col].dtype.kind in ("O", "U", "S")
        ]
        if text_cols:
            mask = pd.Series(False, index=df.index)
            needle = search.lower()
            for col in text_cols:
                mask = mask | df[col].fillna("").astype(str).str.lower().str.contains(
                    needle,
                    na=False,
                    regex=False,
                )
            df = df[mask]
    return df


def _load_analysis_df(
    *,
    analysis_id: str,
    db: Any,
    filter_col: str | None = None,
    filter_val: str | None = None,
    filters: list[dict] | None = None,
    search: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis = _get_analysis_or_job(db, analysis_id)
    if analysis.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis not completed. Current status: {analysis.get('status')}",
        )

    df = analysis_runner.load_filtered_df(
        artifacts_dir=_get_artifacts_dir(),
        analysis_id=analysis_id,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=filters,
        search=search,
    )
    if df is None:
        raise HTTPException(status_code=404, detail="Results file not found.")
    return df, analysis


def _resolve_analysis_text_col(analysis: dict[str, Any], df: pd.DataFrame) -> str:
    summary = analysis.get("result_summary") or {}
    text_col = summary.get("text_col")
    if isinstance(text_col, str) and text_col in df.columns:
        return text_col
    for candidate in ("text_feedback", "text", "feedback", "comment", "review", "response"):
        if candidate in df.columns:
            return candidate
    raise HTTPException(
        status_code=422,
        detail="A text column is required to compute embeddings for this analysis.",
    )


def _resolve_embeddings_model(
    analysis: dict[str, Any],
    model_registry: ModelRegistry,
    model_id: str | None,
) -> Any:
    if model_id:
        model_meta = model_registry.get_model(model_id)
        if model_meta is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        return model_meta

    summary = analysis.get("result_summary") or {}
    for model_info in summary.get("models_applied") or []:
        if model_info.get("error"):
            continue
        candidate_id = model_info.get("model_id")
        if not candidate_id:
            continue
        model_meta = model_registry.get_model(candidate_id)
        if model_meta is not None:
            return model_meta
    return None


def _load_points_df(analysis_id: str) -> pd.DataFrame:
    points_path = _get_artifacts_dir() / analysis_id / "analytics" / "embeddings" / "points.csv"
    if not points_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Embeddings have not been computed for this analysis.",
        )
    return pd.read_csv(points_path, keep_default_na=False)


def _resolve_analysis_model(
    *,
    analysis: dict[str, Any],
    task: str,
    model_registry: ModelRegistry,
) -> tuple[dict[str, Any], str]:
    summary = analysis.get("result_summary") or {}
    models_applied = summary.get("models_applied") or []
    for model_info in models_applied:
        if model_info.get("task") != task:
            continue
        if model_info.get("error"):
            continue
        model_id = model_info.get("model_id")
        if not model_id:
            continue
        model_meta = model_registry.get_model(model_id)
        if model_meta is None:
            continue
        signature = canonicalize_signature(model_meta)
        label_col = str(signature["label_schema"].get("column") or "").strip()
        return model_info, label_col

    raise HTTPException(
        status_code=422,
        detail=f"No successful model output found for task '{task}' in analysis {analysis['id']}.",
    )


def _compute_embeddings_payload(
    *,
    df: pd.DataFrame,
    text_col: str,
    analysis_id: str,
    model_meta: Any = None,
    reuse_cached: bool = True,
    max_features: int = 512,
) -> dict[str, Any]:
    try:
        return compute_or_load_embeddings(
            df=df,
            text_col=text_col,
            analysis_dir=_get_artifacts_dir() / analysis_id,
            model_meta=model_meta,
            reuse_cached=reuse_cached,
            max_features=max_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/datasets/{dataset_id}/descriptive", response_model=dict)
async def get_dataset_descriptive(
    dataset_id: str,
    dataset_version: int | None = Query(None),
    branch_id: str | None = Query(None),
    filter_col: str | None = Query(None),
    filter_val: str | None = Query(None),
    filters: str | None = Query(None, description="JSON array: [{col,op,val}]"),
    search: str | None = Query(None),
    dataset_manager: DatasetManager = Depends(get_dataset_manager),
):
    """Return descriptive statistics for a dataset version or branch."""
    try:
        df = dataset_manager.get_dataframe(
            dataset_id=dataset_id,
            version=dataset_version,
            branch_id=branch_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    filtered_df = _apply_result_filters(
        df,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=_parse_filters(filters),
        search=search,
    )
    return {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "branch_id": branch_id,
        "row_count": int(len(filtered_df)),
        "summary": descriptive_summary(filtered_df),
    }


@router.get("/datasets/{dataset_id}/correlations", response_model=dict)
async def get_dataset_correlations(
    dataset_id: str,
    dataset_version: int | None = Query(None),
    branch_id: str | None = Query(None),
    columns: str | None = Query(None, description="Comma-separated column names"),
    filter_col: str | None = Query(None),
    filter_val: str | None = Query(None),
    filters: str | None = Query(None, description="JSON array: [{col,op,val}]"),
    search: str | None = Query(None),
    dataset_manager: DatasetManager = Depends(get_dataset_manager),
):
    """Return pairwise correlations/associations for a dataset version or branch."""
    try:
        df = dataset_manager.get_dataframe(
            dataset_id=dataset_id,
            version=dataset_version,
            branch_id=branch_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    filtered_df = _apply_result_filters(
        df,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=_parse_filters(filters),
        search=search,
    )
    selected_columns = None
    if columns:
        selected_columns = [col.strip() for col in columns.split(",") if col.strip()]
    correlations = mixed_pairwise_correlations(filtered_df, columns=selected_columns)
    return {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "branch_id": branch_id,
        "row_count": int(len(filtered_df)),
        "correlations": correlations,
    }


@router.get("/analyses/{analysis_id}/descriptive", response_model=dict)
async def get_analysis_descriptive(
    analysis_id: str,
    filter_col: str | None = Query(None),
    filter_val: str | None = Query(None),
    filters: str | None = Query(None, description="JSON array: [{col,op,val}]"),
    search: str | None = Query(None),
    db=Depends(get_db),
):
    """Return descriptive statistics for filtered analysis results."""
    df, _ = _load_analysis_df(
        analysis_id=analysis_id,
        db=db,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=_parse_filters(filters),
        search=search,
    )
    return {
        "analysis_id": analysis_id,
        "row_count": int(len(df)),
        "summary": descriptive_summary(df),
    }


@router.get("/analyses/{analysis_id}/correlations", response_model=dict)
async def get_analysis_correlations(
    analysis_id: str,
    columns: str | None = Query(None, description="Comma-separated column names"),
    filter_col: str | None = Query(None),
    filter_val: str | None = Query(None),
    filters: str | None = Query(None, description="JSON array: [{col,op,val}]"),
    search: str | None = Query(None),
    db=Depends(get_db),
):
    """Return pairwise correlations/associations for filtered analysis results."""
    df, _ = _load_analysis_df(
        analysis_id=analysis_id,
        db=db,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=_parse_filters(filters),
        search=search,
    )
    selected_columns = None
    if columns:
        selected_columns = [col.strip() for col in columns.split(",") if col.strip()]
    correlations = mixed_pairwise_correlations(df, columns=selected_columns)
    return {
        "analysis_id": analysis_id,
        "row_count": int(len(df)),
        "correlations": correlations,
    }


@router.get("/analyses/{analysis_id}/diagnostics", response_model=dict)
async def get_analysis_diagnostics(
    analysis_id: str,
    task: str = Query(...),
    filter_col: str | None = Query(None),
    filter_val: str | None = Query(None),
    filters: str | None = Query(None, description="JSON array: [{col,op,val}]"),
    search: str | None = Query(None),
    db=Depends(get_db),
    model_registry: ModelRegistry = Depends(get_model_registry),
):
    """Return task-specific diagnostics from analysis results when labels exist."""
    df, analysis = _load_analysis_df(
        analysis_id=analysis_id,
        db=db,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=_parse_filters(filters),
        search=search,
    )
    model_info, label_col = _resolve_analysis_model(
        analysis=analysis,
        task=task,
        model_registry=model_registry,
    )

    pred_col = model_info.get("pred_col")
    conf_col = model_info.get("conf_col")
    if not pred_col or pred_col not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction column is unavailable for task '{task}'.",
        )
    if not label_col or label_col not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Ground-truth label column is unavailable for task '{task}'. "
                "The analysis results do not include labels needed for diagnostics."
            ),
        )

    confidences = df[conf_col] if conf_col and conf_col in df.columns else None
    diagnostics = classification_diagnostics(
        y_true=df[label_col],
        y_pred=df[pred_col],
        confidences=confidences,
    )
    return {
        "analysis_id": analysis_id,
        "task": task,
        "label_col": label_col,
        "pred_col": pred_col,
        "conf_col": conf_col if conf_col in df.columns else None,
        "diagnostics": diagnostics,
    }


@router.post("/analyses/{analysis_id}/embeddings", response_model=dict)
async def post_analysis_embeddings(
    analysis_id: str,
    body: EmbeddingsRequest,
    db=Depends(get_db),
    model_registry: ModelRegistry = Depends(get_model_registry),
):
    """Compute or load cached 2D embeddings for analysis rows."""
    df, analysis = _load_analysis_df(
        analysis_id=analysis_id,
        db=db,
    )
    text_col = _resolve_analysis_text_col(analysis, df)
    model_meta = _resolve_embeddings_model(analysis, model_registry, body.model_id)
    payload = _compute_embeddings_payload(
        df=df,
        text_col=text_col,
        analysis_id=analysis_id,
        model_meta=model_meta,
        reuse_cached=body.reuse_cached,
        max_features=body.max_features,
    )
    return {
        "analysis_id": analysis_id,
        "text_col": text_col,
        **payload,
    }


@router.post("/analyses/{analysis_id}/cluster", response_model=dict)
async def post_analysis_cluster(
    analysis_id: str,
    body: ClusterRequest,
    db=Depends(get_db),
    model_registry: ModelRegistry = Depends(get_model_registry),
):
    """Cluster analysis rows in the embedding space."""
    analysis = _get_analysis_or_job(db, analysis_id)
    analytics_dir = _get_artifacts_dir() / analysis_id / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    if not body.reuse_embeddings or not (analytics_dir / "embeddings" / "points.csv").exists():
        df, _ = _load_analysis_df(analysis_id=analysis_id, db=db)
        text_col = _resolve_analysis_text_col(analysis, df)
        model_meta = _resolve_embeddings_model(analysis, model_registry, body.model_id)
        _compute_embeddings_payload(
            df=df,
            text_col=text_col,
            analysis_id=analysis_id,
            model_meta=model_meta,
            reuse_cached=body.reuse_embeddings,
        )

    points_df = _load_points_df(analysis_id)
    return {
        "analysis_id": analysis_id,
        **cluster_points(
            points_df=points_df,
            analytics_dir=analytics_dir,
            method=body.method,
            k=body.k,
            eps=body.eps,
            min_samples=body.min_samples,
        ),
    }


@router.post("/analyses/{analysis_id}/outliers", response_model=dict)
async def post_analysis_outliers(
    analysis_id: str,
    body: OutliersRequest,
    db=Depends(get_db),
    model_registry: ModelRegistry = Depends(get_model_registry),
):
    """Detect outliers in the embedding space."""
    analysis = _get_analysis_or_job(db, analysis_id)
    analytics_dir = _get_artifacts_dir() / analysis_id / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    if not body.reuse_embeddings or not (analytics_dir / "embeddings" / "points.csv").exists():
        df, _ = _load_analysis_df(analysis_id=analysis_id, db=db)
        text_col = _resolve_analysis_text_col(analysis, df)
        model_meta = _resolve_embeddings_model(analysis, model_registry, body.model_id)
        _compute_embeddings_payload(
            df=df,
            text_col=text_col,
            analysis_id=analysis_id,
            model_meta=model_meta,
            reuse_cached=body.reuse_embeddings,
        )

    points_df = _load_points_df(analysis_id)
    return {
        "analysis_id": analysis_id,
        **detect_outliers(
            points_df=points_df,
            analytics_dir=analytics_dir,
            method=body.method,
            contamination=body.contamination,
            n_neighbors=body.n_neighbors,
        ),
    }
