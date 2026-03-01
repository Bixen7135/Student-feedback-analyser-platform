"""Analysis runner — apply registered models to uploaded datasets."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

from src.inference.engine import check_compatibility, run_inference
from src.schema import DatasetSchemaSnapshot, normalize_column_name, resolve_roles
from src.storage.database import Database
from src.storage.dataset_manager import DatasetManager
from src.storage.model_registry import ModelRegistry
from src.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory job store (reset on restart — batch use only)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_text_col(df: pd.DataFrame) -> str | None:
    """Return the resolved text column, or None when no string column exists."""
    return resolve_roles(df=df, column_roles={}, overrides={}).text_col


def run_analysis(
    dataset_id: str,
    model_ids: list[str],
    dataset_manager: DatasetManager,
    model_registry: ModelRegistry,
    db: Database,
    artifacts_dir: Path,
    analysis_id: str,
    dataset_version: int | None = None,
    text_col: str | None = None,
    name: str = "",
    description: str = "",
    tags: list[str] | None = None,
    branch_id: str | None = None,
) -> dict[str, Any]:
    """
    Apply registered models to a dataset and save results.

    Returns a summary dict with keys:
        analysis_id, dataset_id, n_rows, n_rows_processed, models_applied, created_at
    """
    tags = tags or []

    # 1. Load dataset
    df = dataset_manager.get_dataframe(dataset_id, dataset_version, branch_id=branch_id)
    if df is None or len(df) == 0:
        raise ValueError(f"Dataset '{dataset_id}' is empty or not found.")

    n_rows_total = len(df)
    log.info(
        "analysis_start",
        analysis_id=analysis_id,
        dataset_id=dataset_id,
        n_rows=n_rows_total,
        n_models=len(model_ids),
    )

    # 2. Resolve text column (explicit override or auto-detect)
    if text_col:
        text_col = normalize_column_name(text_col)
        if text_col not in df.columns:
            raise ValueError(
                f"Text column '{text_col}' not found in dataset. "
                f"Available columns: {list(df.columns)}."
            )
    compatibility_roles = (
        {normalize_column_name(text_col): "text"}
        if text_col
        else {}
    )
    resolved_columns = resolve_roles(
        df=df,
        column_roles=compatibility_roles,
        overrides={"text_col": text_col} if text_col else {},
    )
    text_col = resolved_columns.text_col
    dataset_schema = DatasetSchemaSnapshot(
        columns=tuple(str(col) for col in df.columns),
        normalized_columns=tuple(normalize_column_name(col) for col in df.columns),
        column_roles=compatibility_roles,
    )

    # 3. Prepare results DataFrame (copy of original)
    results_df = df.copy()

    models_applied: list[dict[str, Any]] = []

    # 4. For each model: load + predict
    for model_id in model_ids:
        model_meta = model_registry.get_model(model_id)
        if model_meta is None:
            log.warning("model_not_found", model_id=model_id)
            continue

        compatibility = check_compatibility(
            model_meta=model_meta,
            dataset_schema=dataset_schema,
            resolved_columns=resolved_columns,
        )

        # Build prediction column names using short model_id prefix
        mid_short = model_id[:8]
        pred_col = f"{model_meta.task}_{mid_short}_pred"
        conf_col = f"{model_meta.task}_{mid_short}_conf"

        if not compatibility["ok"]:
            error_message = "; ".join(
                str(reason.get("message") or reason.get("code") or "incompatible")
                for reason in compatibility["reasons"]
            )
            models_applied.append(
                {
                    "model_id": model_id,
                    "model_name": model_meta.name,
                    "task": model_meta.task,
                    "model_type": model_meta.model_type,
                    "error": error_message,
                    "pred_col": None,
                    "conf_col": None,
                    "n_predicted": 0,
                    "class_distribution": {},
                    "classes": [],
                    "compatibility": compatibility,
                    "preprocess_spec_applied": compatibility.get("preprocess_spec_id"),
                    "text_col_used": compatibility.get("text_col_used"),
                    "model_input_col": None,
                }
            )
            log.warning(
                "model_incompatible",
                analysis_id=analysis_id,
                model_id=model_id,
                reasons=compatibility["reasons"],
            )
            continue

        try:
            inference = run_inference(
                df=df,
                model_meta=model_meta,
                resolved_columns=resolved_columns,
            )
            preds = inference["predictions"]
            conf_vals = inference["confidences"]

            results_df[pred_col] = preds
            results_df[conf_col] = conf_vals

            models_applied.append(
                {
                    "model_id": model_id,
                    "model_name": model_meta.name,
                    "task": model_meta.task,
                    "model_type": model_meta.model_type,
                    "error": None,
                    "pred_col": pred_col,
                    "conf_col": conf_col,
                    "n_predicted": inference["n_predicted"],
                    "class_distribution": inference["class_distribution"],
                    "classes": inference["classes"],
                    "compatibility": inference["compatibility"],
                    "preprocess_spec_applied": inference["preprocess_spec_applied"],
                    "text_col_used": inference["text_col_used"],
                    "model_input_col": inference["model_input_col"],
                }
            )
            log.info(
                "model_predictions_done",
                analysis_id=analysis_id,
                model_id=model_id,
                task=model_meta.task,
                n_predicted=inference["n_predicted"],
            )
        except Exception as exc:
            log.error(
                "model_predict_failed",
                model_id=model_id,
                error=str(exc),
            )
            models_applied.append(
                {
                    "model_id": model_id,
                    "model_name": model_meta.name,
                    "task": model_meta.task,
                    "model_type": model_meta.model_type,
                    "error": str(exc),
                    "pred_col": None,
                    "conf_col": None,
                    "n_predicted": 0,
                    "class_distribution": {},
                    "classes": [],
                    "compatibility": compatibility,
                    "preprocess_spec_applied": compatibility.get("preprocess_spec_id"),
                    "text_col_used": compatibility.get("text_col_used"),
                    "model_input_col": None,
                }
            )

    # 5. Save results
    run_dir = artifacts_dir / analysis_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "analysis_id": analysis_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "branch_id": branch_id,
        "n_rows": n_rows_total,
        "n_rows_processed": n_rows_total,
        "text_col": text_col,
        "models_applied": models_applied,
        "created_at": _utcnow(),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    # 6. Persist to SQLite
    _upsert_analysis_run(
        db=db,
        analysis_id=analysis_id,
        name=name,
        description=description,
        tags=tags,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        model_ids=model_ids,
        status="completed",
        result_summary=summary,
        branch_id=branch_id,
    )

    log.info(
        "analysis_complete",
        analysis_id=analysis_id,
        n_models_applied=len(models_applied),
    )
    return summary


# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------


def _upsert_analysis_run(
    db: Database,
    analysis_id: str,
    name: str,
    description: str,
    tags: list[str],
    dataset_id: str | None,
    dataset_version: int | None,
    model_ids: list[str],
    status: str,
    result_summary: dict | None = None,
    branch_id: str | None = None,
    pipeline_run_id: str | None = None,
) -> None:
    db.execute(
        """INSERT OR REPLACE INTO analysis_runs
        (id, name, description, tags, dataset_id, dataset_version, model_ids,
         created_at, status, pipeline_run_id, result_summary, comments, branch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)""",
        (
            analysis_id,
            name,
            description,
            orjson.dumps(tags).decode(),
            dataset_id,
            dataset_version,
            orjson.dumps(model_ids).decode(),
            _utcnow(),
            status,
            pipeline_run_id or "",  # pipeline_run_id links to a pipeline run if spawned from one
            orjson.dumps(result_summary or {}).decode(),
            branch_id,
        ),
    )
    # Populate normalized model-ref table (INSERT OR IGNORE is idempotent)
    for mid in model_ids:
        if mid:
            db.execute(
                "INSERT OR IGNORE INTO analysis_model_refs (analysis_id, model_id) VALUES (?, ?)",
                (analysis_id, mid),
            )
    db.commit()


def _update_analysis_status(
    db: Database,
    analysis_id: str,
    status: str,
    result_summary: dict | None = None,
) -> None:
    if result_summary is not None:
        db.execute(
            "UPDATE analysis_runs SET status = ?, result_summary = ? WHERE id = ?",
            (status, orjson.dumps(result_summary).decode(), analysis_id),
        )
    else:
        db.execute(
            "UPDATE analysis_runs SET status = ? WHERE id = ?",
            (status, analysis_id),
        )
    db.commit()


# ---------------------------------------------------------------------------
# Job store helpers
# ---------------------------------------------------------------------------


def create_job(
    job_id: str,
    dataset_id: str,
    model_ids: list[str],
    name: str,
    description: str,
    tags: list[str],
    dataset_version: int | None,
    branch_id: str | None = None,
    db: Database | None = None,
) -> dict[str, Any]:
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "branch_id": branch_id,
        "model_ids": model_ids,
        "name": name,
        "description": description,
        "tags": tags,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_summary": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
    # Persist immediately so the analysis is visible even before the background thread starts
    if db is not None:
        _upsert_analysis_run(
            db=db,
            analysis_id=job_id,
            name=name,
            description=description,
            tags=tags,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_ids=model_ids,
            status="pending",
            branch_id=branch_id,
        )
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def list_jobs() -> list[dict[str, Any]]:
    with _jobs_lock:
        return [dict(j) for j in reversed(list(_jobs.values()))]


def run_job_background(
    job_id: str,
    dataset_id: str,
    model_ids: list[str],
    name: str,
    description: str,
    tags: list[str],
    dataset_version: int | None,
    text_col: str | None,
    dataset_manager: DatasetManager,
    model_registry: ModelRegistry,
    db: Database,
    artifacts_dir: Path,
    branch_id: str | None = None,
) -> None:
    """Execute analysis synchronously in a background thread."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = _utcnow()

    # Record pending state in DB
    _upsert_analysis_run(
        db=db,
        analysis_id=job_id,
        name=name,
        description=description,
        tags=tags,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        model_ids=model_ids,
        status="running",
        branch_id=branch_id,
    )

    try:
        result = run_analysis(
            dataset_id=dataset_id,
            model_ids=model_ids,
            dataset_manager=dataset_manager,
            model_registry=model_registry,
            db=db,
            artifacts_dir=artifacts_dir,
            analysis_id=job_id,
            dataset_version=dataset_version,
            text_col=text_col,
            name=name,
            description=description,
            tags=tags,
            branch_id=branch_id,
        )
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = _utcnow()
            _jobs[job_id]["result_summary"] = result
    except Exception as exc:
        log.error("analysis_job_failed", job_id=job_id, error=str(exc))
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["completed_at"] = _utcnow()
            _jobs[job_id]["error"] = str(exc)
        _update_analysis_status(db, job_id, "failed")


# ---------------------------------------------------------------------------
# DB-backed listing (for history persistence across restarts)
# ---------------------------------------------------------------------------


def list_analyses_from_db(
    db: Database,
    dataset_id: str | None = None,
    model_id: str | None = None,
    status: str | None = None,
    sort: str = "created_at",
    order: str = "desc",
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    """List analysis runs from SQLite with optional filters and pagination."""
    conditions: list[str] = []
    params: list[Any] = []

    if dataset_id:
        conditions.append("dataset_id = ?")
        params.append(dataset_id)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if model_id:
        # Use normalized table for exact, reliable model_id lookups
        conditions.append("id IN (SELECT analysis_id FROM analysis_model_refs WHERE model_id = ?)")
        params.append(model_id)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    valid_sorts = {"created_at", "name", "status"}
    if sort not in valid_sorts:
        sort = "created_at"
    order_dir = "DESC" if order.lower() == "desc" else "ASC"

    count_row = db.fetchone(
        f"SELECT COUNT(*) as cnt FROM analysis_runs {where}", tuple(params)
    )
    total = count_row["cnt"] if count_row else 0

    offset = (page - 1) * per_page
    rows = db.fetchall(
        f"SELECT * FROM analysis_runs {where} ORDER BY {sort} {order_dir} LIMIT ? OFFSET ?",
        (*params, per_page, offset),
    )
    return [_row_to_dict(r) for r in rows], total


def get_analysis_from_db(db: Database, analysis_id: str) -> dict[str, Any] | None:
    """Fetch a single analysis run from SQLite."""
    row = db.fetchone("SELECT * FROM analysis_runs WHERE id = ?", (analysis_id,))
    if row is None:
        return None
    return _row_to_dict(row)


def delete_analysis_from_db(db: Database, analysis_id: str) -> bool:
    """Delete analysis metadata from SQLite. Returns True if deleted."""
    result = db.execute(
        "DELETE FROM analysis_runs WHERE id = ?", (analysis_id,)
    )
    db.execute(
        "DELETE FROM analysis_model_refs WHERE analysis_id = ?", (analysis_id,)
    )
    db.commit()
    return result.rowcount > 0


def update_analysis_metadata(
    db: Database,
    analysis_id: str,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    comments: str | None = None,
) -> dict[str, Any] | None:
    """Patch mutable analysis metadata fields."""
    updates: list[str] = []
    params: list[Any] = []
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if description is not None:
        updates.append("description = ?")
        params.append(description)
    if tags is not None:
        updates.append("tags = ?")
        params.append(orjson.dumps(tags).decode())
    if comments is not None:
        updates.append("comments = ?")
        params.append(comments)
    if not updates:
        return get_analysis_from_db(db, analysis_id)
    params.append(analysis_id)
    db.execute(
        f"UPDATE analysis_runs SET {', '.join(updates)} WHERE id = ?",
        tuple(params),
    )
    db.commit()
    return get_analysis_from_db(db, analysis_id)


def _row_to_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    for field in ("tags", "model_ids", "result_summary"):
        if isinstance(d.get(field), str):
            try:
                d[field] = orjson.loads(d[field])
            except Exception:
                d[field] = [] if field != "result_summary" else {}
    return d


# ---------------------------------------------------------------------------
# Results file access
# ---------------------------------------------------------------------------


def get_results_path(artifacts_dir: Path, analysis_id: str) -> Path:
    """Return the path to the results CSV for an analysis."""
    return artifacts_dir / analysis_id / "results.csv"


def get_summary_path(artifacts_dir: Path, analysis_id: str) -> Path:
    """Return the path to the summary JSON for an analysis."""
    return artifacts_dir / analysis_id / "summary.json"


def _apply_result_filters(
    df: pd.DataFrame,
    filter_col: str | None = None,
    filter_val: str | None = None,
    filters: list[dict] | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    """
    Apply filter rules and optional full-text search to a results DataFrame.

    Phase 5 multi-filter takes priority over legacy filter_col/filter_val.
    Filter rule ops: eq, ne, contains, gt, lt, gte, lte.
    """
    if filters:
        for f in filters:
            col = f.get("col", "")
            op = f.get("op", "eq")
            val = f.get("val", "")
            if not col or col not in df.columns:
                continue
            col_str = df[col].fillna("").astype(str)
            if op == "eq":
                df = df[col_str.str.lower() == str(val).lower()]
            elif op == "ne":
                df = df[col_str.str.lower() != str(val).lower()]
            elif op == "contains":
                df = df[
                    col_str.str.lower().str.contains(
                        str(val).lower(), na=False, regex=False
                    )
                ]
            elif op in ("gt", "lt", "gte", "lte"):
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
                except (ValueError, TypeError):
                    pass
    elif filter_col and filter_val and filter_col in df.columns:
        # Legacy single-filter
        df = df[df[filter_col].fillna("").astype(str).str.lower() == filter_val.lower()]

    if search and search.strip():
        # Search across all string-dtype columns
        text_cols = [
            c
            for c in df.columns
            if df[c].dtype == object or df[c].dtype.kind in ("O", "U", "S")
        ]
        if text_cols:
            mask = pd.Series(False, index=df.index)
            lsearch = search.lower()
            for tc in text_cols:
                mask = mask | df[tc].fillna("").str.lower().str.contains(
                    lsearch, na=False, regex=False
                )
            df = df[mask]

    return df


def load_filtered_df(
    artifacts_dir: Path,
    analysis_id: str,
    sort_col: str | None = None,
    sort_order: str = "asc",
    filter_col: str | None = None,
    filter_val: str | None = None,
    filters: list[dict] | None = None,
    search: str | None = None,
) -> pd.DataFrame | None:
    """
    Load the results CSV, apply filters + sort, return a DataFrame.

    Returns None if the results file does not exist.
    Used for both paginated reads and full filtered exports.
    """
    path = get_results_path(artifacts_dir, analysis_id)
    if not path.exists():
        return None

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = _apply_result_filters(df, filter_col, filter_val, filters, search)

    if sort_col and sort_col in df.columns:
        ascending = sort_order.lower() != "desc"
        try:
            df = df.sort_values(sort_col, ascending=ascending, na_position="last")
        except Exception:
            pass

    return df


def load_results_page(
    artifacts_dir: Path,
    analysis_id: str,
    offset: int = 0,
    limit: int = 50,
    sort_col: str | None = None,
    sort_order: str = "asc",
    filter_col: str | None = None,
    filter_val: str | None = None,
    filters: list[dict] | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """
    Load a paginated slice of the results CSV.

    Phase 5: supports multi-column filters (filters=[{col,op,val}]) and
    full-text search across string columns.

    Returns: {total_rows, offset, limit, columns, rows}
    """
    df = load_filtered_df(
        artifacts_dir=artifacts_dir,
        analysis_id=analysis_id,
        sort_col=sort_col,
        sort_order=sort_order,
        filter_col=filter_col,
        filter_val=filter_val,
        filters=filters,
        search=search,
    )
    if df is None:
        return {"total_rows": 0, "offset": offset, "limit": limit, "columns": [], "rows": []}

    total = len(df)
    page_df = df.iloc[offset: offset + limit]

    return {
        "total_rows": total,
        "offset": offset,
        "limit": limit,
        "columns": list(df.columns),
        "rows": page_df.values.tolist(),
    }


def get_distributions(
    artifacts_dir: Path,
    analysis_id: str,
    columns: list[str],
) -> dict[str, Any]:
    """
    Compute value counts per requested column from the results CSV.

    Returns: { distributions: { col: { value: count } } }
    """
    path = get_results_path(artifacts_dir, analysis_id)
    if not path.exists():
        return {"distributions": {}}

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    distributions: dict[str, dict[str, int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        distributions[col] = {str(k): int(v) for k, v in vc.items()}

    return {"distributions": distributions}


def get_segment_stats(
    artifacts_dir: Path,
    analysis_id: str,
    group_by: str,
    metric_col: str,
) -> dict[str, Any]:
    """
    Compute grouped stats (count, mean, median, std) for a numeric metric_col
    broken down by a categorical group_by column.

    Returns: { group_by, metric_col, groups: [{ group, count, mean, median, std }] }
    """
    path = get_results_path(artifacts_dir, analysis_id)
    if not path.exists():
        return {"group_by": group_by, "metric_col": metric_col, "groups": []}

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if group_by not in df.columns or metric_col not in df.columns:
        return {"group_by": group_by, "metric_col": metric_col, "groups": []}

    df = df.copy()
    df["_metric"] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["_metric"])

    groups: list[dict[str, Any]] = []
    for name, grp in df.groupby(group_by, sort=True):
        metrics = grp["_metric"]
        groups.append(
            {
                "group": str(name),
                "count": int(len(grp)),
                "mean": round(float(metrics.mean()), 4),
                "median": round(float(metrics.median()), 4),
                "std": round(float(metrics.std()), 4) if len(grp) > 1 else 0.0,
            }
        )

    return {"group_by": group_by, "metric_col": metric_col, "groups": groups}


def get_cross_compare_disagreements(
    artifacts_dir: Path,
    analysis_ids: list[str],
    columns: list[str],
) -> dict[str, float]:
    """
    Compute per-column disagreement rates across multiple analyses.

    Disagreement rate = fraction of rows where not all analyses agree on that column's value.
    Rows are aligned by position; only the minimum shared row count is compared.
    """
    dfs: list[pd.DataFrame] = []
    for aid in analysis_ids:
        path = get_results_path(artifacts_dir, aid)
        if path.exists():
            dfs.append(pd.read_csv(path, dtype=str, keep_default_na=False))

    if len(dfs) < 2:
        return {}

    min_rows = min(len(df) for df in dfs)
    if min_rows == 0:
        return {}

    disagreements: dict[str, float] = {}
    for col in columns:
        if not all(col in df.columns for df in dfs):
            continue
        col_data = [df[col].iloc[:min_rows].tolist() for df in dfs]
        n_disagree = sum(len(set(row_vals)) > 1 for row_vals in zip(*col_data))
        disagreements[col] = round(n_disagree / min_rows, 4)

    return disagreements


def get_anomalies(
    artifacts_dir: Path,
    analysis_id: str,
    conf_threshold: float = 0.6,
) -> dict[str, Any]:
    """
    Detect anomalous rows in the results CSV.

    Anomalies are rows where:
    - Any confidence column value is below conf_threshold, OR
    - The detected text column is empty / whitespace-only.

    Returns: {analysis_id, anomalies, total, conf_threshold}
    Each anomaly entry: {row_index, reasons, data}
    """
    path = get_results_path(artifacts_dir, analysis_id)
    if not path.exists():
        return {
            "analysis_id": analysis_id,
            "anomalies": [],
            "total": 0,
            "conf_threshold": conf_threshold,
        }

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    conf_cols = [c for c in df.columns if c.endswith("_conf")]

    text_col = _detect_text_col(df)

    anomaly_data: list[dict[str, Any]] = []
    for i in range(len(df)):
        row = df.iloc[i]
        reasons: list[str] = []

        for cc in conf_cols:
            try:
                cv = float(row[cc])
                if cv < conf_threshold:
                    reasons.append(f"low_confidence({cc}={cv:.3f})")
            except (ValueError, TypeError):
                pass

        if text_col and row.get(text_col, "").strip() == "":
            reasons.append(f"empty_text({text_col})")

        if reasons:
            anomaly_data.append(
                {
                    "row_index": i,
                    "reasons": reasons,
                    "data": dict(row),
                }
            )

    # Cap response at 500 anomaly rows to keep response size manageable
    return {
        "analysis_id": analysis_id,
        "anomalies": anomaly_data[:500],
        "total": len(anomaly_data),
        "conf_threshold": conf_threshold,
    }
