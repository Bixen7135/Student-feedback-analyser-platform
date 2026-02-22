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

from src.ingest.loader import COLUMN_MAP
from src.storage.dataset_manager import DatasetManager
from src.storage.model_registry import ModelRegistry
from src.text_tasks.char_ngram_classifier import CharNgramClassifier
from src.text_tasks.tfidf_classifier import TfidfClassifier
from src.text_tasks.base import TextClassifier
from src.training.config import TrainingConfig
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
) -> dict[str, Any]:
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
) -> None:
    """Execute training synchronously in a background thread."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = _utcnow()

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
        )
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = _utcnow()
            _jobs[job_id]["model_id"] = result["model_id"]
            _jobs[job_id]["model_name"] = result["model_name"]
            _jobs[job_id]["model_version"] = result["model_version"]
            _jobs[job_id]["metrics"] = result["metrics"]
            _jobs[job_id]["config"] = result["config"]
    except Exception as exc:
        log.error("training_job_failed", job_id=job_id, error=str(exc))
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["completed_at"] = _utcnow()
            _jobs[job_id]["error"] = str(exc)
