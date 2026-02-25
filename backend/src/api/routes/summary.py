"""Dashboard summary API routes."""
from __future__ import annotations

import re
from pathlib import Path

import orjson
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_db, get_run_manager

router = APIRouter(prefix="/api", tags=["summary"])


class SummaryResponse(BaseModel):
    total_datasets: int
    total_models: int
    total_analyses: int
    total_responses: int
    n_latent_factors: int
    n_survey_items: int


def _read_factor_count(run_dir: Path) -> int:
    summary_path = run_dir / "psychometrics" / "cfa_summary.json"
    if not summary_path.exists():
        return 0
    try:
        data = orjson.loads(summary_path.read_bytes())
    except Exception:
        return 0
    factor_names = data.get("factor_names", [])
    if not isinstance(factor_names, list):
        return 0
    valid_names = [f for f in factor_names if isinstance(f, str) and f.strip()]
    return len(valid_names)


@router.get("/summary", response_model=SummaryResponse)
async def get_summary(
    db=Depends(get_db),
    run_manager=Depends(get_run_manager),
):
    """Return aggregate platform counts for the dashboard."""
    datasets_row = db.fetchone(
        "SELECT COUNT(*) AS cnt, COALESCE(SUM(row_count), 0) AS total_rows "
        "FROM datasets WHERE status = 'active'"
    )
    models_row = db.fetchone(
        "SELECT COUNT(*) AS cnt FROM models WHERE status = 'active'"
    )
    analyses_row = db.fetchone("SELECT COUNT(*) AS cnt FROM analysis_runs")

    total_datasets = int(datasets_row["cnt"] if datasets_row else 0)
    total_models = int(models_row["cnt"] if models_row else 0)
    total_analyses = int(analyses_row["cnt"] if analyses_row else 0)
    total_responses = int(datasets_row["total_rows"] if datasets_row else 0)

    # Estimate survey item count from the most recently created active dataset.
    n_survey_items = 0
    survey_row = db.fetchone(
        """SELECT d.schema_info, dv.column_roles
           FROM datasets d
           LEFT JOIN dataset_versions dv
             ON dv.dataset_id = d.id
            AND dv.version = d.current_version
            AND dv.is_deleted = 0
           WHERE d.status = 'active'
           ORDER BY d.created_at DESC
           LIMIT 1"""
    )
    if survey_row:
        try:
            column_roles = orjson.loads(survey_row["column_roles"] or "{}")
            if isinstance(column_roles, dict):
                items = {
                    str(role)
                    for role in column_roles.values()
                    if isinstance(role, str) and re.match(r"^item_\d+$", role)
                }
                n_survey_items = len(items)
        except Exception:
            n_survey_items = 0

        if n_survey_items == 0:
            try:
                schema_info = orjson.loads(survey_row["schema_info"] or "[]")
                if isinstance(schema_info, list):
                    item_cols = {
                        str(col.get("name"))
                        for col in schema_info
                        if isinstance(col, dict)
                        and isinstance(col.get("name"), str)
                        and re.match(r"^item_\d+$", str(col.get("name")))
                    }
                    n_survey_items = len(item_cols)
            except Exception:
                n_survey_items = 0

    # Latent factors from the latest completed pipeline run's psychometrics output.
    n_latent_factors = 0
    run_row = db.fetchone(
        """SELECT id
           FROM pipeline_runs
           WHERE status = 'completed'
           ORDER BY COALESCE(completed_at, created_at) DESC
           LIMIT 1"""
    )
    if run_row:
        try:
            n_latent_factors = _read_factor_count(run_manager.get_run_dir(run_row["id"]))
        except Exception:
            n_latent_factors = 0

    return SummaryResponse(
        total_datasets=total_datasets,
        total_models=total_models,
        total_analyses=total_analyses,
        total_responses=total_responses,
        n_latent_factors=n_latent_factors,
        n_survey_items=n_survey_items,
    )

