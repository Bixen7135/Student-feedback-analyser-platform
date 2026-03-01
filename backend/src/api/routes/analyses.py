"""Analysis API routes — launch and monitor batch analysis jobs."""
from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

import orjson
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_dataset_manager, get_model_registry, get_db
from src.analysis import runner as analysis_runner
from src.analysis.comparator import compare_analyses
from src.inference.engine import check_compatibility
from src.schema import DatasetSchemaSnapshot, normalize_column_name, resolve_roles

router = APIRouter(prefix="/api/analyses", tags=["analyses"])

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent


def _get_artifacts_dir() -> Path:
    base = Path(
        os.environ.get(
            "SFAP_ANALYSIS_DIR", str(_BACKEND_DIR / "analysis_runs")
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class StartAnalysisRequest(BaseModel):
    dataset_id: str
    model_ids: list[str] = Field(..., min_length=1)
    name: str = Field("", max_length=200)
    description: str = Field("", max_length=2000)
    tags: list[str] = Field(default_factory=list)
    dataset_version: int | None = None
    branch_id: str | None = None
    text_col: str | None = None


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str
    dataset_id: str
    dataset_version: int | None
    branch_id: str | None = None
    model_ids: list[str]
    name: str
    description: str
    tags: list[str]
    started_at: str | None
    completed_at: str | None
    error: str | None
    result_summary: dict | None
    psychometrics_warning: str | None = None


class AnalysisListResponse(BaseModel):
    analyses: list[dict]
    total: int
    page: int
    per_page: int


class UpdateAnalysisRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    comments: str | None = None


class CompareRequest(BaseModel):
    analysis_ids: list[str] = Field(..., min_length=2, max_length=2)


class CrossCompareRequest(BaseModel):
    analysis_ids: list[str] = Field(..., min_length=2)
    columns: list[str] = Field(..., min_length=1)


class ResultsResponse(BaseModel):
    analysis_id: str
    total_rows: int
    offset: int
    limit: int
    columns: list[str]
    rows: list[list[str]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=AnalysisJobResponse, status_code=202)
async def start_analysis(
    body: StartAnalysisRequest,
    background_tasks: BackgroundTasks,
    dataset_manager=Depends(get_dataset_manager),
    model_registry=Depends(get_model_registry),
    db=Depends(get_db),
):
    """Launch a batch analysis job in the background.

    Returns 202 Accepted immediately; poll GET /api/analyses/{job_id}/status
    to track progress.
    """
    ds = dataset_manager.get_dataset(body.dataset_id)
    if ds is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {body.dataset_id}"
        )

    has_psychometrics = db.fetchone(
        "SELECT id FROM pipeline_runs WHERE dataset_id = ? AND status = 'completed' LIMIT 1",
        (body.dataset_id,),
    )
    psychometrics_warning = None if has_psychometrics else (
        "No completed pipeline run found for this dataset. "
        "Psychometrics has not been validated. See Scientific Spec."
    )

    try:
        target_df = dataset_manager.get_dataframe(
            dataset_id=body.dataset_id,
            version=body.dataset_version,
            branch_id=body.branch_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if body.text_col:
        normalized_text_col = normalize_column_name(body.text_col)
        if normalized_text_col not in target_df.columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Text column '{normalized_text_col}' not found in the target dataset.",
                    "reasons": [
                        {
                            "code": "missing_column",
                            "role": "text",
                            "column": normalized_text_col,
                            "suggested_fix": "Choose a valid text column from the dataset schema.",
                        }
                    ],
                },
            )
    compatibility_roles = (
        {normalize_column_name(body.text_col): "text"}
        if body.text_col
        else {}
    )
    resolved_columns = resolve_roles(
        df=target_df,
        column_roles=compatibility_roles,
        overrides={"text_col": body.text_col} if body.text_col else {},
    )
    dataset_schema = DatasetSchemaSnapshot(
        columns=tuple(str(col) for col in target_df.columns),
        normalized_columns=tuple(normalize_column_name(col) for col in target_df.columns),
        column_roles=compatibility_roles,
    )

    incompatible_models: list[dict] = []
    for model_id in body.model_ids:
        m = model_registry.get_model(model_id)
        if m is None:
            raise HTTPException(
                status_code=404, detail=f"Model not found: {model_id}"
            )
        report = check_compatibility(
            model_meta=m,
            dataset_schema=dataset_schema,
            resolved_columns=resolved_columns,
        )
        if not report["ok"]:
            incompatible_models.append(
                {
                    "model_id": model_id,
                    "model_name": m.name,
                    "compatibility": report,
                }
            )

    if incompatible_models:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more models are incompatible with the target dataset.",
                "suggested_fix": (
                    "Assign the required column roles, choose a valid text column, "
                    "or select a dataset version whose schema matches the model input contract."
                ),
                "models": incompatible_models,
            },
        )

    job_id = f"analysis_{uuid.uuid4().hex[:16]}"

    job = analysis_runner.create_job(
        job_id=job_id,
        dataset_id=body.dataset_id,
        model_ids=body.model_ids,
        name=body.name,
        description=body.description,
        tags=body.tags,
        dataset_version=body.dataset_version,
        branch_id=body.branch_id,
        db=db,
    )

    background_tasks.add_task(
        analysis_runner.run_job_background,
        job_id=job_id,
        dataset_id=body.dataset_id,
        model_ids=body.model_ids,
        name=body.name,
        description=body.description,
        tags=body.tags,
        dataset_version=body.dataset_version,
        text_col=body.text_col,
        dataset_manager=dataset_manager,
        model_registry=model_registry,
        db=db,
        artifacts_dir=_get_artifacts_dir(),
        branch_id=body.branch_id,
    )

    return AnalysisJobResponse(**job, psychometrics_warning=psychometrics_warning)


@router.get("", response_model=AnalysisListResponse)
async def list_analyses(
    dataset_id: str | None = Query(None),
    model_id: str | None = Query(None),
    status: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db=Depends(get_db),
):
    """List all analysis runs from persistent storage (most recent first)."""
    analyses, total = analysis_runner.list_analyses_from_db(
        db=db,
        dataset_id=dataset_id,
        model_id=model_id,
        status=status,
        sort=sort,
        order=order,
        page=page,
        per_page=per_page,
    )

    return AnalysisListResponse(
        analyses=analyses,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{analysis_id}", response_model=dict)
async def get_analysis(
    analysis_id: str,
    db=Depends(get_db),
):
    """Get analysis detail from DB, falling back to in-memory job store."""
    # Check in-memory first (for running jobs not yet persisted)
    job = analysis_runner.get_job(analysis_id)
    if job and job["status"] in ("pending", "running"):
        return {
            "id": job["job_id"],
            "status": job["status"],
            "dataset_id": job["dataset_id"],
            "dataset_version": job["dataset_version"],
            "branch_id": job.get("branch_id"),
            "model_ids": job["model_ids"],
            "name": job["name"],
            "description": job["description"],
            "tags": job["tags"],
            "comments": "",
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "error": job["error"],
            "result_summary": job["result_summary"],
            "created_at": job.get("started_at"),
            "pipeline_run_id": "",
        }

    # Load from DB
    analysis = analysis_runner.get_analysis_from_db(db, analysis_id)
    if analysis is None:
        # Fall back to in-memory for any status
        if job:
            return {
                "id": job["job_id"],
                "status": job["status"],
                "dataset_id": job["dataset_id"],
                "dataset_version": job["dataset_version"],
                "branch_id": job.get("branch_id"),
                "model_ids": job["model_ids"],
                "name": job["name"],
                "description": job["description"],
                "tags": job["tags"],
                "comments": "",
                "started_at": job["started_at"],
                "completed_at": job["completed_at"],
                "error": job["error"],
                "result_summary": job["result_summary"],
                "created_at": job.get("started_at"),
                "pipeline_run_id": "",
            }
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )

    return analysis


@router.patch("/{analysis_id}", response_model=dict)
async def update_analysis(
    analysis_id: str,
    body: UpdateAnalysisRequest,
    db=Depends(get_db),
):
    """Update editable analysis metadata (name, description, tags, comments)."""
    result = analysis_runner.update_analysis_metadata(
        db=db,
        analysis_id=analysis_id,
        name=body.name,
        description=body.description,
        tags=body.tags,
        comments=body.comments,
    )
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )
    return result


@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    db=Depends(get_db),
):
    """Delete analysis metadata and results artifacts."""
    import shutil

    deleted = analysis_runner.delete_analysis_from_db(db, analysis_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )

    # Remove artifacts directory if it exists
    artifacts_dir = _get_artifacts_dir() / analysis_id
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir, ignore_errors=True)

    # Remove from in-memory store
    with analysis_runner._jobs_lock:
        analysis_runner._jobs.pop(analysis_id, None)

    return {"deleted": True, "analysis_id": analysis_id}


@router.post("/compare", response_model=dict)
async def compare_two_analyses(
    body: CompareRequest,
    db=Depends(get_db),
):
    """Compare two analysis runs — distribution deltas and disagreement rates."""
    try:
        result = compare_analyses(
            analysis_id_1=body.analysis_ids[0],
            analysis_id_2=body.analysis_ids[1],
            db=db,
            artifacts_dir=_get_artifacts_dir(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@router.post("/cross-compare", response_model=dict)
async def cross_compare_analyses(
    body: CrossCompareRequest,
    db=Depends(get_db),
):
    """Compare N analyses: per-column distributions and row-level disagreement rates.

    Body: ``{ analysis_ids: [...], columns: [...] }``
    Response: per-analysis value count distributions + per-column disagreement rates.
    """
    artifacts_dir = _get_artifacts_dir()

    for aid in body.analysis_ids:
        a = analysis_runner.get_analysis_from_db(db, aid)
        if a is None:
            raise HTTPException(status_code=404, detail=f"Analysis not found: {aid}")
        if a.get("status") != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Analysis {aid} not completed (status={a.get('status')})",
            )

    per_analysis: dict[str, dict] = {}
    for aid in body.analysis_ids:
        dist = analysis_runner.get_distributions(artifacts_dir, aid, body.columns)
        per_analysis[aid] = dist["distributions"]

    disagreement_rates = analysis_runner.get_cross_compare_disagreements(
        artifacts_dir=artifacts_dir,
        analysis_ids=body.analysis_ids,
        columns=body.columns,
    )

    return {
        "analysis_ids": body.analysis_ids,
        "columns": body.columns,
        "per_analysis": per_analysis,
        "disagreement_rates": disagreement_rates,
    }


@router.get("/{analysis_id}/distributions", response_model=dict)
async def get_distributions(
    analysis_id: str,
    columns: str = Query(..., description="Comma-separated column names"),
    db=Depends(get_db),
):
    """Get value count distributions for the requested columns of a completed analysis."""
    analysis = analysis_runner.get_analysis_from_db(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )
    if analysis.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis not completed. Current status: {analysis.get('status')}",
        )

    col_list = [c.strip() for c in columns.split(",") if c.strip()]
    return analysis_runner.get_distributions(
        artifacts_dir=_get_artifacts_dir(),
        analysis_id=analysis_id,
        columns=col_list,
    )


@router.get("/{analysis_id}/segment-stats", response_model=dict)
async def get_segment_stats(
    analysis_id: str,
    group_by: str = Query(..., description="Categorical column to group by"),
    metric_col: str = Query(..., description="Numeric column to aggregate"),
    db=Depends(get_db),
):
    """Get grouped segment stats (count/mean/median/std) for a numeric column."""
    analysis = analysis_runner.get_analysis_from_db(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )
    if analysis.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis not completed. Current status: {analysis.get('status')}",
        )

    return analysis_runner.get_segment_stats(
        artifacts_dir=_get_artifacts_dir(),
        analysis_id=analysis_id,
        group_by=group_by,
        metric_col=metric_col,
    )


@router.get("/{analysis_id}/status", response_model=AnalysisJobResponse)
async def get_analysis_status(
    analysis_id: str,
    db=Depends(get_db),
):
    """Poll the live status of an analysis job."""
    job = analysis_runner.get_job(analysis_id)
    if job:
        return AnalysisJobResponse(**job)

    # Reconstruct from DB for completed analyses
    analysis = analysis_runner.get_analysis_from_db(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )
    return AnalysisJobResponse(
        job_id=analysis["id"],
        status=analysis["status"],
        dataset_id=analysis.get("dataset_id", ""),
        dataset_version=analysis.get("dataset_version"),
        branch_id=analysis.get("branch_id"),
        model_ids=analysis.get("model_ids", []),
        name=analysis.get("name", ""),
        description=analysis.get("description", ""),
        tags=analysis.get("tags", []),
        started_at=analysis.get("created_at"),
        completed_at=analysis.get("created_at"),
        error=None,
        result_summary=analysis.get("result_summary"),
    )


@router.get("/{analysis_id}/results", response_model=ResultsResponse)
async def get_analysis_results(
    analysis_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    sort_col: str | None = Query(None),
    sort_order: str = Query("asc"),
    # Legacy single-filter (kept for backward compat)
    filter_col: str | None = Query(None),
    filter_val: str | None = Query(None),
    # Phase 5: multi-filter as JSON-encoded list of {col, op, val}
    filters: str | None = Query(None, description="JSON array: [{col,op,val}]"),
    # Phase 5: full-text search across string columns
    search: str | None = Query(None, description="Search string across text columns"),
    db=Depends(get_db),
):
    """Get paginated analysis results table.

    Phase 5 adds multi-column filtering via the ``filters`` query param
    (JSON-encoded list of ``{col, op, val}`` objects) and full-text ``search``
    across all string columns.  Supported ops: ``eq``, ``ne``, ``contains``,
    ``gt``, ``lt``, ``gte``, ``lte``.
    """
    # Check analysis exists
    analysis = analysis_runner.get_analysis_from_db(db, analysis_id)
    job = analysis_runner.get_job(analysis_id)
    if analysis is None and job is None:
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )

    current_status = analysis.get("status") if analysis else (job or {}).get("status")
    if current_status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis not completed. Current status: {current_status}",
        )

    parsed_filters: list[dict] | None = None
    if filters:
        try:
            parsed_filters = orjson.loads(filters)
            if not isinstance(parsed_filters, list):
                raise ValueError("filters must be a JSON array")
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid filters JSON: {exc}",
            )

    page_data = analysis_runner.load_results_page(
        artifacts_dir=_get_artifacts_dir(),
        analysis_id=analysis_id,
        offset=offset,
        limit=limit,
        sort_col=sort_col,
        sort_order=sort_order,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=parsed_filters,
        search=search,
    )
    return ResultsResponse(
        analysis_id=analysis_id,
        **page_data,
    )


@router.get("/{analysis_id}/results/export")
async def export_analysis_results(
    analysis_id: str,
    format: str = Query("csv", pattern="^(csv|json)$"),
    # Phase 5: accept the same filter params as the results endpoint
    filter_col: str | None = Query(None),
    filter_val: str | None = Query(None),
    filters: str | None = Query(None, description="JSON array: [{col,op,val}]"),
    search: str | None = Query(None),
    sort_col: str | None = Query(None),
    sort_order: str = Query("asc"),
    db=Depends(get_db),
):
    """Export analysis results as CSV or JSON.

    Phase 5: when ``filters``, ``search``, or ``sort_col`` are supplied the
    export reflects the filtered / sorted view rather than the raw full file.
    """
    analysis = analysis_runner.get_analysis_from_db(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )
    if analysis.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis not completed. Current status: {analysis.get('status')}",
        )

    results_path = analysis_runner.get_results_path(_get_artifacts_dir(), analysis_id)
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found.")

    # Determine whether we need filtering / sorting
    parsed_filters: list[dict] | None = None
    if filters:
        try:
            parsed_filters = orjson.loads(filters)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid filters JSON: {exc}")

    needs_filter = bool(parsed_filters or filter_col or search or sort_col)

    if not needs_filter:
        # Fast path: stream the raw file unchanged
        if format == "csv":
            def _csv_stream():
                with open(results_path, "rb") as f:
                    while chunk := f.read(65536):
                        yield chunk

            filename = f"analysis_{analysis_id[:12]}_results.csv"
            return StreamingResponse(
                _csv_stream(),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            df_full = pd.read_csv(results_path, dtype=str, keep_default_na=False)
            records = df_full.to_dict(orient="records")

            def _json_stream():
                yield orjson.dumps(
                    {"analysis_id": analysis_id, "rows": records},
                    option=orjson.OPT_INDENT_2,
                )

            filename = f"analysis_{analysis_id[:12]}_results.json"
            return StreamingResponse(
                _json_stream(),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

    # Filtered / sorted path
    df = analysis_runner.load_filtered_df(
        artifacts_dir=_get_artifacts_dir(),
        analysis_id=analysis_id,
        sort_col=sort_col or None,
        sort_order=sort_order,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=parsed_filters,
        search=search,
    )
    if df is None:
        df = pd.DataFrame()

    filename_suffix = "filtered_results" if needs_filter else "results"

    if format == "csv":
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

        def _filtered_csv_stream():
            yield csv_bytes

        filename = f"analysis_{analysis_id[:12]}_{filename_suffix}.csv"
        return StreamingResponse(
            _filtered_csv_stream(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    else:
        records = df.to_dict(orient="records")

        def _filtered_json_stream():
            yield orjson.dumps(
                {"analysis_id": analysis_id, "rows": records},
                option=orjson.OPT_INDENT_2,
            )

        filename = f"analysis_{analysis_id[:12]}_{filename_suffix}.json"
        return StreamingResponse(
            _filtered_json_stream(),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


@router.get("/{analysis_id}/anomalies")
async def get_analysis_anomalies(
    analysis_id: str,
    conf_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Confidence below this threshold is flagged"),
    db=Depends(get_db),
):
    """Detect anomalous rows: low-confidence predictions or empty text.

    Returns up to 500 anomaly rows with their row index, reasons, and data.
    """
    analysis = analysis_runner.get_analysis_from_db(db, analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404, detail=f"Analysis not found: {analysis_id}"
        )
    if analysis.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis not completed. Current status: {analysis.get('status')}",
        )

    return analysis_runner.get_anomalies(
        artifacts_dir=_get_artifacts_dir(),
        analysis_id=analysis_id,
        conf_threshold=conf_threshold,
    )
