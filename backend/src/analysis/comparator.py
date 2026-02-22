"""Analysis comparator — compare two batch analysis runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.runner import (
    get_analysis_from_db,
    get_results_path,
    get_summary_path,
)
from src.storage.database import Database
from src.utils.logging import get_logger

import orjson

log = get_logger(__name__)


def compare_analyses(
    analysis_id_1: str,
    analysis_id_2: str,
    db: Database,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """
    Compare two analysis runs.

    Returns a comparison dict with:
      - run_1, run_2: basic metadata
      - shared_tasks: tasks predicted by both runs
      - task_comparisons: per-task distribution deltas
      - disagreement: agreement rate if same dataset, else null
    """
    run_1 = get_analysis_from_db(db, analysis_id_1)
    run_2 = get_analysis_from_db(db, analysis_id_2)

    if run_1 is None:
        raise ValueError(f"Analysis not found: {analysis_id_1}")
    if run_2 is None:
        raise ValueError(f"Analysis not found: {analysis_id_2}")

    # Load summaries from files (have model/distribution detail)
    summary_path_1 = get_summary_path(artifacts_dir, analysis_id_1)
    summary_path_2 = get_summary_path(artifacts_dir, analysis_id_2)

    summary_1: dict[str, Any] = {}
    summary_2: dict[str, Any] = {}

    if summary_path_1.exists():
        summary_1 = orjson.loads(summary_path_1.read_bytes())
    if summary_path_2.exists():
        summary_2 = orjson.loads(summary_path_2.read_bytes())

    # Extract models_applied from summaries
    models_1 = {m["task"]: m for m in summary_1.get("models_applied", []) if not m.get("error")}
    models_2 = {m["task"]: m for m in summary_2.get("models_applied", []) if not m.get("error")}

    shared_tasks = sorted(set(models_1.keys()) & set(models_2.keys()))

    task_comparisons: list[dict[str, Any]] = []
    for task in shared_tasks:
        m1 = models_1[task]
        m2 = models_2[task]
        dist_1 = m1.get("class_distribution", {})
        dist_2 = m2.get("class_distribution", {})

        # Compute deltas for all classes seen in either distribution
        all_classes = sorted(set(dist_1.keys()) | set(dist_2.keys()))
        distribution_deltas = {
            cls: round(dist_2.get(cls, 0.0) - dist_1.get(cls, 0.0), 4)
            for cls in all_classes
        }

        task_comparisons.append(
            {
                "task": task,
                "run_1_model": {
                    "model_id": m1["model_id"],
                    "model_name": m1["model_name"],
                    "model_type": m1["model_type"],
                    "n_predicted": m1["n_predicted"],
                    "class_distribution": dist_1,
                },
                "run_2_model": {
                    "model_id": m2["model_id"],
                    "model_name": m2["model_name"],
                    "model_type": m2["model_type"],
                    "n_predicted": m2["n_predicted"],
                    "class_distribution": dist_2,
                },
                "distribution_deltas": distribution_deltas,
            }
        )

    # Compute disagreement rate if same dataset
    disagreement: dict[str, Any] | None = None
    same_dataset = (
        run_1.get("dataset_id") == run_2.get("dataset_id")
        and run_1.get("dataset_version") == run_2.get("dataset_version")
    )

    if same_dataset and shared_tasks:
        results_path_1 = get_results_path(artifacts_dir, analysis_id_1)
        results_path_2 = get_results_path(artifacts_dir, analysis_id_2)

        if results_path_1.exists() and results_path_2.exists():
            try:
                df1 = pd.read_csv(results_path_1, dtype=str, keep_default_na=False)
                df2 = pd.read_csv(results_path_2, dtype=str, keep_default_na=False)

                if len(df1) == len(df2):
                    disagreement_by_task: dict[str, float] = {}
                    for task in shared_tasks:
                        pred_col_1 = models_1[task].get("pred_col")
                        pred_col_2 = models_2[task].get("pred_col")
                        if (
                            pred_col_1
                            and pred_col_2
                            and pred_col_1 in df1.columns
                            and pred_col_2 in df2.columns
                        ):
                            preds_1 = df1[pred_col_1].fillna("")
                            preds_2 = df2[pred_col_2].fillna("")
                            n_disagree = int((preds_1 != preds_2).sum())
                            total = len(df1)
                            disagreement_by_task[task] = round(n_disagree / total, 4) if total > 0 else 0.0

                    disagreement = {
                        "same_dataset": True,
                        "by_task": disagreement_by_task,
                        "overall": round(
                            sum(disagreement_by_task.values()) / len(disagreement_by_task), 4
                        ) if disagreement_by_task else None,
                    }
            except Exception as exc:
                log.warning("disagreement_compute_failed", error=str(exc))
                disagreement = {"same_dataset": True, "error": str(exc)}

    return {
        "run_1": {
            "id": run_1["id"],
            "name": run_1.get("name", ""),
            "dataset_id": run_1.get("dataset_id"),
            "dataset_version": run_1.get("dataset_version"),
            "status": run_1.get("status"),
            "created_at": run_1.get("created_at"),
            "n_rows": summary_1.get("n_rows"),
        },
        "run_2": {
            "id": run_2["id"],
            "name": run_2.get("name", ""),
            "dataset_id": run_2.get("dataset_id"),
            "dataset_version": run_2.get("dataset_version"),
            "status": run_2.get("status"),
            "created_at": run_2.get("created_at"),
            "n_rows": summary_2.get("n_rows"),
        },
        "shared_tasks": shared_tasks,
        "task_comparisons": task_comparisons,
        "disagreement": disagreement,
    }
