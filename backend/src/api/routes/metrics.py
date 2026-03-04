"""Metrics API routes."""
from __future__ import annotations

from pathlib import Path

import orjson  # type: ignore
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_run_manager
from src.api.schemas import (
    PsychometricsMetricsResponse,
    ClassificationMetricsResponse,
    FusionMetricsResponse,
    ContradictionMetricsResponse,
)
from src.utils.run_manager import RunManager

router = APIRouter(prefix="/api/runs/{run_id}/metrics", tags=["metrics"])


def _load_json(path: Path):
    if not path.exists():
        return None
    return orjson.loads(path.read_bytes())


@router.get("/psychometrics", response_model=PsychometricsMetricsResponse)
async def get_psychometrics_metrics(run_id: str, mgr: RunManager = Depends(get_run_manager)):
    if not mgr.run_exists(run_id):
        raise HTTPException(404, f"Run not found: {run_id}")
    rdir = mgr.get_run_dir(run_id)

    summary = _load_json(rdir / "psychometrics" / "cfa_summary.json")
    if not summary:
        raise HTTPException(404, "Psychometrics results not found — run psychometrics stage first")

    reliability = _load_json(rdir / "psychometrics" / "reliability.json") or {}
    fit_stats = _load_json(rdir / "psychometrics" / "fit_statistics.json") or {}

    # Load loadings as nested dict
    loadings_path = rdir / "psychometrics" / "loadings.csv"
    loadings_dict = None
    if loadings_path.exists():
        df = pd.read_csv(loadings_path, index_col=0)
        loadings_dict = {
            row: {col: float(df.loc[row, col]) for col in df.columns}
            for row in df.index
        }

    return PsychometricsMetricsResponse(
        method=summary.get("method", "unknown"),
        n_obs=summary.get("n_obs", 0),
        factor_names=summary.get("factor_names", []),
        fit_statistics=fit_stats,
        reliability=reliability,
        loadings=loadings_dict,
    )


@router.get("/classification/{task_name}", response_model=list[ClassificationMetricsResponse])
async def get_classification_metrics(
    run_id: str,
    task_name: str,
    mgr: RunManager = Depends(get_run_manager),
):
    if not mgr.run_exists(run_id):
        raise HTTPException(404, f"Run not found: {run_id}")

    valid_tasks = ["language", "sentiment", "detail_level"]
    if task_name not in valid_tasks:
        raise HTTPException(400, f"Unknown task: {task_name}")

    rdir = mgr.get_run_dir(run_id)
    results = []

    task_dir = rdir / "text_tasks" / task_name
    if not task_dir.exists():
        raise HTTPException(404, f"No classification metrics found for task {task_name}")

    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_type = model_dir.name
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = orjson.loads(metrics_path.read_bytes())
        val = data.get("val", {})
        results.append(ClassificationMetricsResponse(
            task=task_name,
            model_type=model_type,
            macro_f1=val.get("macro_f1", 0.0),
            accuracy=val.get("accuracy", 0.0),
            per_class_f1=val.get("per_class_f1", {}),
            confusion_matrix=val.get("confusion_matrix", []),
            classes=data.get("classes", []),
            n_samples=val.get("n_samples", 0),
        ))

    if not results:
        raise HTTPException(404, f"No classification metrics found for task {task_name}")
    return results


@router.get("/fusion", response_model=FusionMetricsResponse)
async def get_fusion_metrics(run_id: str, mgr: RunManager = Depends(get_run_manager)):
    if not mgr.run_exists(run_id):
        raise HTTPException(404, f"Run not found: {run_id}")
    rdir = mgr.get_run_dir(run_id)
    data = _load_json(rdir / "fusion" / "results.json")
    if not data:
        raise HTTPException(404, "Fusion results not found — run fusion stage first")
    return FusionMetricsResponse(
        factor_names=data.get("factor_names", []),
        survey_only=data.get("survey_only", {}),
        text_only=data.get("text_only", {}),
        late_fusion=data.get("late_fusion", {}),
        delta_mae=data.get("delta_mae", {}),
        delta_r2=data.get("delta_r2", {}),
        ablations=data.get("ablations"),
    )


@router.get("/contradiction", response_model=ContradictionMetricsResponse)
async def get_contradiction_metrics(run_id: str, mgr: RunManager = Depends(get_run_manager)):
    if not mgr.run_exists(run_id):
        raise HTTPException(404, f"Run not found: {run_id}")
    rdir = mgr.get_run_dir(run_id)
    data = _load_json(rdir / "contradiction" / "results.json")
    if not data:
        raise HTTPException(404, "Contradiction results not found")
    return ContradictionMetricsResponse(
        overall_rate=data.get("overall_rate", 0.0),
        n_total=data.get("n_total", 0),
        n_contradictions=data.get("n_contradictions", 0),
        by_type=data.get("by_type", {}),
        stratified_by_language=data.get("stratified_by_language", {}),
        stratified_by_detail_level=data.get("stratified_by_detail_level", {}),
        disclaimer=data.get("disclaimer", ""),
    )
