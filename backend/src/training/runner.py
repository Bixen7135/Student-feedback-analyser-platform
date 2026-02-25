"""Training runner — trains text classifiers on user datasets from DatasetManager."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd

import orjson

from src.ingest.loader import COLUMN_MAP
from src.storage.database import Database
from src.storage.dataset_manager import DatasetManager
from src.storage.model_registry import ModelRegistry
from src.text_tasks.char_ngram_classifier import CharNgramClassifier
from src.text_tasks.tfidf_classifier import TfidfClassifier
from src.text_tasks.base import TextClassifier
from src.training.config import TrainingConfig
from src.reporting.model_card import generate_model_card_for_registry_model
from src.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_LABEL_COLS: dict[str, str] = {
    "language": "language",
    "sentiment": "sentiment_class",
    "detail_level": "detail_level",
}

VALID_TASKS = frozenset(TASK_LABEL_COLS.keys())
VALID_MODEL_TYPES = frozenset({"tfidf", "char_ngram"})

# Text column names tried in priority order when auto-detecting
_TEXT_COL_CANDIDATES = [
    "text_processed",
    "text_feedback",
    "text",
    "feedback",
    "comment",
    "review",
    "response",
]

MIN_SAMPLES_PER_CLASS = 5
MIN_CLASSES = 2

# ---------------------------------------------------------------------------
# In-memory job store (reset on restart — batch use only)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LabelValidationError(ValueError):
    """Raised when label column is missing or has insufficient data."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_text_col(df: pd.DataFrame) -> str:
    """Return the first matching text column, or the first object-dtype column."""
    for candidate in _TEXT_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.kind in ("O", "U", "S"):
            return col
    raise LabelValidationError(
        "No text column found in dataset. "
        f"Expected one of: {_TEXT_COL_CANDIDATES}. "
        f"Available columns: {list(df.columns)}."
    )


def _detect_text_col_with_roles(
    df: pd.DataFrame, column_roles: dict[str, str]
) -> str:
    """Detect text column using column_roles first, then fall back to auto-detection."""
    for col, role in column_roles.items():
        if role == "text" and col in df.columns:
            return col
    return _detect_text_col(df)


def _detect_label_col_with_roles(
    df: pd.DataFrame, column_roles: dict[str, str], task: str
) -> str:
    """Detect label column using column_roles first, then fall back to TASK_LABEL_COLS."""
    for col, role in column_roles.items():
        if role == task and col in df.columns:
            return col
    # Fall back to standard column name for task
    default = TASK_LABEL_COLS.get(task, "")
    if default and default in df.columns:
        return default
    raise LabelValidationError(
        f"Label column for task '{task}' not found. "
        f"Tried column_roles ({column_roles}) and default '{default}'. "
        f"Available columns: {list(df.columns)}. "
        f"Assign a column role 'text' and '{task}' in the dataset version settings."
    )


def _validate_labels(
    df: pd.DataFrame,
    label_col: str,
    task: str,
) -> pd.DataFrame:
    """
    Validate and clean label column for training.

    Steps:
      1. Check column exists.
      2. Drop rows with missing / empty labels.
      3. Enforce MIN_CLASSES and MIN_SAMPLES_PER_CLASS.

    Returns cleaned DataFrame (may be smaller than input).
    Raises LabelValidationError on unrecoverable issues.
    """
    if label_col not in df.columns:
        raise LabelValidationError(
            f"Label column '{label_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}. "
            f"Task '{task}' requires column '{label_col}'."
        )

    before = len(df)
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""].copy()
    after = len(df)

    if after == 0:
        raise LabelValidationError(
            f"No valid labels in column '{label_col}' — all values are missing or empty."
        )

    if before > after:
        log.info(
            "dropped_missing_labels",
            task=task,
            dropped=before - after,
            remaining=after,
        )

    # Check class counts
    class_counts = df[label_col].value_counts()
    n_classes = len(class_counts)

    if n_classes < MIN_CLASSES:
        raise LabelValidationError(
            f"Task '{task}' requires at least {MIN_CLASSES} distinct classes, "
            f"but only {n_classes} found: {class_counts.to_dict()}."
        )

    # Warn about and remove classes with too few samples
    small = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
    if len(small) > 0:
        log.warning(
            "small_class_removed",
            task=task,
            classes=small.to_dict(),
            min_required=MIN_SAMPLES_PER_CLASS,
        )
        valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
        df = df[df[label_col].isin(valid_classes)].copy()
        remaining = df[label_col].nunique()
        if remaining < MIN_CLASSES:
            raise LabelValidationError(
                f"After removing classes with < {MIN_SAMPLES_PER_CLASS} samples, "
                f"only {remaining} class(es) remain for task '{task}'. Cannot train."
            )

    return df.reset_index(drop=True)


def _build_classifier(model_type: str, config: TrainingConfig) -> TextClassifier:
    kwargs: dict[str, Any] = {}
    if config.C is not None:
        kwargs["C"] = config.C
    if config.max_iter is not None:
        kwargs["max_iter"] = config.max_iter
    if config.max_features is not None:
        kwargs["max_features"] = config.max_features

    if model_type == "tfidf":
        return TfidfClassifier(**kwargs)
    elif model_type == "char_ngram":
        return CharNgramClassifier(**kwargs)
    raise ValueError(
        f"Unknown model_type: {model_type!r}. Valid: {sorted(VALID_MODEL_TYPES)}"
    )


def _apply_class_balancing(
    train_df: pd.DataFrame,
    label_col: str,
    strategy: str,
    seed: int,
) -> pd.DataFrame:
    """Apply oversampling if requested; class_weight handled by classifier."""
    if strategy in ("none", "class_weight"):
        return train_df

    if strategy == "oversample":
        max_count = int(train_df[label_col].value_counts().max())
        rng = np.random.default_rng(seed)
        parts: list[pd.DataFrame] = [train_df]
        for cls, count in train_df[label_col].value_counts().items():
            if count < max_count:
                n_extra = max_count - int(count)
                cls_rows = train_df[train_df[label_col] == cls]
                idx = rng.integers(0, len(cls_rows), size=n_extra)
                parts.append(cls_rows.iloc[idx])
        combined = pd.concat(parts, ignore_index=True)
        return combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    raise ValueError(f"Unknown class_balancing strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------


def run_training(
    dataset_id: str,
    task: str,
    model_type: str,
    dataset_manager: DatasetManager,
    model_registry: ModelRegistry,
    artifacts_dir: Path,
    config: TrainingConfig | None = None,
    dataset_version: int | None = None,
    branch_id: str | None = None,
    seed: int = 42,
    name: str | None = None,
    base_model_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """
    Train a text classifier on a user dataset and auto-register in ModelRegistry.

    Returns a result dict with keys:
        run_id, model_id, model_name, model_version, metrics, config
    """
    if task not in VALID_TASKS:
        raise ValueError(
            f"Unknown task: {task!r}. Valid: {sorted(VALID_TASKS)}"
        )
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Valid: {sorted(VALID_MODEL_TYPES)}"
        )

    config = config or TrainingConfig()
    # base_model_id can also come from config
    if base_model_id is None:
        base_model_id = config.base_model_id

    # 1. Load dataset from storage
    df = dataset_manager.get_dataframe(dataset_id, dataset_version, branch_id=branch_id)
    if df is None or len(df) == 0:
        raise LabelValidationError(
            f"Dataset '{dataset_id}' is empty or not found."
        )

    # 2. Load column_roles from version metadata (supports user-renamed columns)
    # If branch_id is provided, get the head version's column roles
    try:
        if branch_id:
            head = dataset_manager.get_branch_head_version(branch_id)
            column_roles = dataset_manager.get_column_roles(dataset_id, version_id=head.id if head else None)
        else:
            column_roles = dataset_manager.get_column_roles(dataset_id, version=dataset_version)
    except Exception:
        column_roles = {}

    # 3. Resolve column names (config override → column_roles → auto-detect)
    if config.label_col:
        label_col = COLUMN_MAP.get(config.label_col.strip(), config.label_col.strip())
    else:
        label_col = _detect_label_col_with_roles(df, column_roles, task)
        label_col = COLUMN_MAP.get(label_col.strip(), label_col.strip())

    if config.text_col:
        text_col = COLUMN_MAP.get(config.text_col.strip(), config.text_col.strip())
    else:
        text_col = _detect_text_col_with_roles(df, column_roles)
        text_col = COLUMN_MAP.get(text_col.strip(), text_col.strip())

    log.info(
        "training_start",
        dataset_id=dataset_id,
        task=task,
        model_type=model_type,
        text_col=text_col,
        label_col=label_col,
        n_rows=len(df),
    )

    # 3. Validate and clean labels
    df = _validate_labels(df, label_col, task)

    # 4. Stratified split
    from src.splits.splitter import stratified_split

    try:
        train_df, val_df, test_df = stratified_split(
            df,
            stratify_col=label_col,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=seed,
        )
    except ValueError as exc:
        raise LabelValidationError(
            f"Stratified split failed for task '{task}': {exc}. "
            "The dataset may be too small or a class may have too few samples."
        ) from exc

    # 5. Class balancing on training set
    train_df = _apply_class_balancing(
        train_df, label_col, config.class_balancing, seed
    )

    # 6. Build and train classifier
    clf = _build_classifier(model_type, config)
    train_texts = train_df[text_col].fillna("").tolist()
    train_labels = train_df[label_col].tolist()
    clf.fit(train_texts, train_labels, seed=seed)

    # 6b. Fine-tuning: warm-start from base model when base_model_id is provided
    warm_started = False
    base_clf: TextClassifier | None = None
    if base_model_id is not None:
        base_model = model_registry.get_model(base_model_id)
        if base_model is None:
            raise ValueError(
                f"Base model not found: {base_model_id}. "
                "Check the model registry for a valid model_id."
            )
        if base_model.task != task:
            raise ValueError(
                f"Base model task '{base_model.task}' does not match requested task '{task}'. "
                "Fine-tuning requires the same task."
            )
        if base_model.model_type != model_type:
            raise ValueError(
                f"Base model type '{base_model.model_type}' does not match requested model_type '{model_type}'. "
                "Fine-tuning requires the same model type."
            )
        base_model_path = model_registry.load_model_artifact(base_model_id)
        base_clf_instance = clf.__class__.load(base_model_path)
        warm_started = clf.warm_start_from(
            base_clf_instance, train_texts, train_labels, seed=seed
        )
        base_clf = base_clf_instance
        if not warm_started:
            log.warning(
                "warm_start_skipped_class_mismatch",
                base_model_id=base_model_id,
                base_classes=base_clf_instance.classes_,
                new_classes=clf.classes_,
            )
        else:
            log.info(
                "warm_start_applied",
                base_model_id=base_model_id,
                task=task,
                model_type=model_type,
            )

    # 7. Evaluate on val set
    from src.evaluation.classification_metrics import compute_classification_metrics

    classes = clf.classes_
    val_texts = val_df[text_col].fillna("").tolist()
    val_labels = val_df[label_col].tolist()
    val_preds = clf.predict(val_texts)
    val_m = compute_classification_metrics(np.array(val_labels), val_preds, classes)

    train_preds = clf.predict(train_texts)
    train_m = compute_classification_metrics(
        np.array(train_labels), train_preds, classes
    )

    # 7b. Evaluate both base and new model on test set when fine-tuning
    test_texts = test_df[text_col].fillna("").tolist()
    test_labels = test_df[label_col].tolist()
    fine_tuning_info: dict[str, Any] | None = None
    if base_model_id is not None and base_clf is not None:
        new_test_preds = clf.predict(test_texts)
        new_test_m = compute_classification_metrics(
            np.array(test_labels), new_test_preds, classes
        )
        base_test_preds = base_clf.predict(test_texts)
        base_test_classes = base_clf.classes_
        base_test_m = compute_classification_metrics(
            np.array(test_labels), base_test_preds, base_test_classes
        )
        fine_tuning_info = {
            "base_model_id": base_model_id,
            "warm_started": warm_started,
            "base_model_test": {
                "macro_f1": base_test_m.macro_f1,
                "accuracy": base_test_m.accuracy,
            },
            "new_model_test": {
                "macro_f1": new_test_m.macro_f1,
                "accuracy": new_test_m.accuracy,
            },
            "delta_macro_f1": round(new_test_m.macro_f1 - base_test_m.macro_f1, 4),
            "delta_accuracy": round(new_test_m.accuracy - base_test_m.accuracy, 4),
        }

    metrics: dict[str, Any] = {
        "task": task,
        "model_type": model_type,
        "seed": seed,
        "hyperparameters": clf.get_hyperparameters(),
        "classes": classes,
        "train": {
            "macro_f1": train_m.macro_f1,
            "accuracy": train_m.accuracy,
        },
        "val": {
            "macro_f1": val_m.macro_f1,
            "accuracy": val_m.accuracy,
            "per_class_f1": val_m.per_class_f1,
            "confusion_matrix": val_m.confusion_matrix.tolist(),
        },
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "text_col": text_col,
        "label_col": label_col,
    }
    if fine_tuning_info is not None:
        metrics["fine_tuning"] = fine_tuning_info

    log.info(
        "training_complete",
        task=task,
        model_type=model_type,
        val_macro_f1=round(val_m.macro_f1, 4),
    )

    # 8. Persist artifacts to a training-run directory
    run_id = f"training_{uuid.uuid4().hex[:12]}"
    run_dir = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"
    clf.save(model_path)
    metrics_path.write_bytes(orjson.dumps(metrics, option=orjson.OPT_INDENT_2))

    # 9. Auto-register in ModelRegistry
    model_name = name or f"{task}_{model_type}_ds{dataset_id[:8]}"
    model_meta = model_registry.register_model(
        name=model_name,
        task=task,
        model_type=model_type,
        source_model_path=model_path,
        source_metrics_path=metrics_path,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        config=config.to_dict(),
        metrics=metrics,
        run_id=run_id,
        base_model_id=base_model_id,
        job_id=job_id,
    )

    return {
        "run_id": run_id,
        "model_id": model_meta.id,
        "model_name": model_meta.name,
        "model_version": model_meta.version,
        "metrics": metrics,
        "config": config.to_dict(),
    }


# ---------------------------------------------------------------------------
# Job store helpers
# ---------------------------------------------------------------------------


def create_job(
    job_id: str,
    dataset_id: str,
    task: str,
    model_type: str,
    dataset_version: int | None,
    branch_id: str | None,
    seed: int,
    name: str | None,
    base_model_id: str | None = None,
    db: Database | None = None,
) -> dict[str, Any]:
    now = _utcnow()
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "branch_id": branch_id,
        "task": task,
        "model_type": model_type,
        "seed": seed,
        "name": name,
        "base_model_id": base_model_id,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "model_id": None,
        "model_name": None,
        "model_version": None,
        "metrics": None,
        "config": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job

    if db is not None:
        try:
            db.execute(
                """INSERT OR IGNORE INTO training_jobs
                (id, dataset_id, dataset_version, branch_id, task, model_type, seed,
                 name, base_model_id, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, dataset_id, dataset_version, branch_id, task, model_type,
                 seed, name, base_model_id, "pending", now),
            )
            db.commit()
        except Exception:
            log.warning("training_job_db_insert_failed", job_id=job_id)

    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def list_jobs() -> list[dict[str, Any]]:
    with _jobs_lock:
        return [dict(j) for j in reversed(list(_jobs.values()))]


def list_jobs_from_db(
    db: Database,
    task: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List training jobs from SQLite, merging with in-memory for live running state."""
    conditions: list[str] = []
    params: list[Any] = []
    if task:
        conditions.append("task = ?")
        params.append(task)
    if status:
        conditions.append("status = ?")
        params.append(status)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = db.fetchall(
        f"SELECT * FROM training_jobs {where} ORDER BY created_at DESC",
        tuple(params),
    )

    # Convert rows to dicts matching the in-memory job schema
    result: list[dict[str, Any]] = []
    with _jobs_lock:
        mem_jobs = dict(_jobs)

    for row in rows:
        d = dict(row)
        job_id = d["id"]
        # Merge hot in-memory state (status may be more current for running jobs)
        if job_id in mem_jobs:
            mem = mem_jobs[job_id]
            d["status"] = mem["status"]
            d["started_at"] = mem.get("started_at") or d.get("started_at")
            d["completed_at"] = mem.get("completed_at") or d.get("completed_at")
            d["error"] = mem.get("error") or d.get("error")
            d["model_id"] = mem.get("model_id") or d.get("model_id")
        # Normalise to the expected output schema
        metrics_raw = d.get("metrics")
        if isinstance(metrics_raw, str):
            try:
                d["metrics"] = orjson.loads(metrics_raw)
            except Exception:
                d["metrics"] = None
        config_raw = d.get("config")
        if isinstance(config_raw, str):
            try:
                d["config"] = orjson.loads(config_raw)
            except Exception:
                d["config"] = None
        result.append({
            "job_id": job_id,
            "status": d.get("status", "pending"),
            "dataset_id": d.get("dataset_id", ""),
            "dataset_version": d.get("dataset_version"),
            "branch_id": d.get("branch_id"),
            "task": d.get("task", ""),
            "model_type": d.get("model_type", ""),
            "seed": d.get("seed", 42),
            "name": d.get("name"),
            "base_model_id": d.get("base_model_id"),
            "started_at": d.get("started_at"),
            "completed_at": d.get("completed_at"),
            "error": d.get("error"),
            "model_id": d.get("model_id"),
            "model_name": None,
            "model_version": None,
            "metrics": d.get("metrics"),
            "config": d.get("config"),
        })

    # Append in-memory jobs not yet written to DB (e.g. created in same request)
    db_ids = {r["job_id"] for r in result}
    for mem in mem_jobs.values():
        if mem["job_id"] not in db_ids:
            if task and mem.get("task") != task:
                continue
            if status and mem.get("status") != status:
                continue
            result.append(dict(mem))

    return result


def run_job_background(
    job_id: str,
    dataset_id: str,
    task: str,
    model_type: str,
    dataset_manager: DatasetManager,
    model_registry: ModelRegistry,
    artifacts_dir: Path,
    config: TrainingConfig | None,
    dataset_version: int | None,
    branch_id: str | None,
    seed: int,
    name: str | None,
    base_model_id: str | None = None,
    db: Database | None = None,
) -> None:
    """Execute training synchronously in a background thread."""
    started_at = _utcnow()
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = started_at

    if db is not None:
        try:
            db.execute(
                "UPDATE training_jobs SET status = 'running', started_at = ? WHERE id = ?",
                (started_at, job_id),
            )
            db.commit()
        except Exception:
            log.warning("training_job_db_update_failed", job_id=job_id, phase="running")

    try:
        result = run_training(
            dataset_id=dataset_id,
            task=task,
            model_type=model_type,
            dataset_manager=dataset_manager,
            model_registry=model_registry,
            artifacts_dir=artifacts_dir,
            config=config,
            dataset_version=dataset_version,
            branch_id=branch_id,
            seed=seed,
            name=name,
            base_model_id=base_model_id,
            job_id=job_id,
        )
        completed_at = _utcnow()
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = completed_at
            _jobs[job_id]["model_id"] = result["model_id"]
            _jobs[job_id]["model_name"] = result["model_name"]
            _jobs[job_id]["model_version"] = result["model_version"]
            _jobs[job_id]["metrics"] = result["metrics"]
            _jobs[job_id]["config"] = result["config"]

        if db is not None:
            try:
                db.execute(
                    """UPDATE training_jobs
                    SET status = 'completed', completed_at = ?, model_id = ?,
                        training_run_id = ?, metrics = ?
                    WHERE id = ?""",
                    (
                        completed_at,
                        result["model_id"],
                        result["run_id"],
                        orjson.dumps(result.get("metrics") or {}).decode(),
                        job_id,
                    ),
                )
                db.commit()
            except Exception:
                log.warning("training_job_db_update_failed", job_id=job_id, phase="completed")

        # Generate and store a model card alongside the registered model lineage.
        try:
            model_meta = model_registry.get_model(result["model_id"])
            if model_meta is not None:
                reports_dir = Path(model_meta.storage_path).parent / "reports"
                generate_model_card_for_registry_model(model_meta, reports_dir)
        except Exception:
            log.warning("model_card_generation_failed", job_id=job_id, model_id=result["model_id"])

    except Exception as exc:
        log.error("training_job_failed", job_id=job_id, error=str(exc))
        failed_at = _utcnow()
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["completed_at"] = failed_at
            _jobs[job_id]["error"] = str(exc)

        if db is not None:
            try:
                db.execute(
                    "UPDATE training_jobs SET status = 'failed', completed_at = ?, error = ? WHERE id = ?",
                    (failed_at, str(exc), job_id),
                )
                db.commit()
            except Exception:
                log.warning("training_job_db_update_failed", job_id=job_id, phase="failed")
