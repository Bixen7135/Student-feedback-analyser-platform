"""Task trainer — independently trains one text classifier per task."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import orjson  # type: ignore
import pandas as pd

from src.text_tasks.base import ClassificationResult, TextClassifier
from src.text_tasks.tfidf_classifier import TfidfClassifier
from src.text_tasks.char_ngram_classifier import CharNgramClassifier
from src.text_tasks.xlm_roberta_classifier import XlmRobertaClassifier
from src.utils.logging import get_logger

log = get_logger(__name__)

TASK_LABEL_COLS = {
    "language": "language",
    "sentiment": "sentiment_class",
    "detail_level": "detail_level",
}

MODEL_FACTORY: dict[str, type[TextClassifier]] = {
    "tfidf": TfidfClassifier,
    "char_ngram": CharNgramClassifier,
    "xlm_roberta": XlmRobertaClassifier,
}


def _make_classifier(model_type: str, config: dict | None = None) -> TextClassifier:
    cls = MODEL_FACTORY[model_type]
    if config:
        return cls(**config)
    return cls()


def train_single_task(
    task_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_type: str,
    run_dir: Path,
    seed: int,
    classifier_config: dict | None = None,
) -> ClassificationResult:
    """
    Train one text classifier for one task.
    Strict isolation: no shared state, no shared gradients with other tasks.
    """
    from src.evaluation.classification_metrics import compute_classification_metrics

    log.info("training_task", task=task_name, model=model_type, seed=seed)

    train_texts = train_df[text_col].fillna("").tolist()
    train_labels = train_df[label_col].fillna("unknown").tolist()
    val_texts = val_df[text_col].fillna("").tolist()
    val_labels = val_df[label_col].fillna("unknown").tolist()

    clf = _make_classifier(model_type, classifier_config)
    clf.fit(train_texts, train_labels, seed=seed)

    # Training set predictions
    train_preds = clf.predict(train_texts)
    train_proba = clf.predict_proba(train_texts)
    classes = clf.classes_
    train_metrics_obj = compute_classification_metrics(
        np.array(train_labels), train_preds, classes
    )

    # Validation set predictions
    val_preds = clf.predict(val_texts)
    val_proba = clf.predict_proba(val_texts)
    val_metrics_obj = compute_classification_metrics(
        np.array(val_labels), val_preds, classes
    )

    # Sentinel check: monitor neutral inflation for sentiment task
    if task_name == "sentiment":
        _check_neutral_inflation(val_preds, classes)

    # Save model
    model_dir = run_dir / "text_tasks" / task_name / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / ("model" if model_type == "xlm_roberta" else "model.joblib")
    clf.save(model_path)

    # Save metrics
    metrics_data: dict[str, Any] = {
        "task": task_name,
        "model_type": model_type,
        "seed": seed,
        "hyperparameters": clf.get_hyperparameters(),
        "classes": classes,
        "train": {
            "macro_f1": train_metrics_obj.macro_f1,
            "accuracy": train_metrics_obj.accuracy,
            "per_class_f1": train_metrics_obj.per_class_f1,
        },
        "val": {
            "macro_f1": val_metrics_obj.macro_f1,
            "accuracy": val_metrics_obj.accuracy,
            "per_class_f1": val_metrics_obj.per_class_f1,
            "confusion_matrix": val_metrics_obj.confusion_matrix.tolist(),
        },
        "predicted_class_distribution": {
            c: float((val_preds == c).mean()) for c in classes
        },
    }
    (model_dir / "metrics.json").write_bytes(
        orjson.dumps(metrics_data, option=orjson.OPT_INDENT_2)
    )

    log.info(
        "task_trained",
        task=task_name,
        model=model_type,
        val_macro_f1=round(val_metrics_obj.macro_f1, 4),
    )

    return ClassificationResult(
        task=task_name,
        model_type=model_type,
        predictions=val_preds,
        probabilities=val_proba,
        classes=classes,
        model_path=model_path,
        hyperparameters=clf.get_hyperparameters(),
        train_metrics={"macro_f1": train_metrics_obj.macro_f1},
        val_metrics={"macro_f1": val_metrics_obj.macro_f1},
    )


def train_all_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_dir: Path,
    seed: int,
    text_col: str = "text_processed",
    tfidf_config: dict | None = None,
    char_ngram_config: dict | None = None,
) -> dict[str, dict[str, ClassificationResult]]:
    """
    Train all text tasks × all model types independently.
    Returns: {task_name: {model_type: ClassificationResult}}
    """
    results: dict[str, dict[str, ClassificationResult]] = {}

    for task_name, label_col in TASK_LABEL_COLS.items():
        results[task_name] = {}
        for model_type in ["tfidf", "char_ngram"]:
            cfg = tfidf_config if model_type == "tfidf" else char_ngram_config
            result = train_single_task(
                task_name=task_name,
                train_df=train_df,
                val_df=val_df,
                text_col=text_col,
                label_col=label_col,
                model_type=model_type,
                run_dir=run_dir,
                seed=seed,
                classifier_config=cfg,
            )
            results[task_name][model_type] = result

    return results


def _check_neutral_inflation(
    predictions: np.ndarray,
    classes: list[str],
    threshold: float = 0.60,
) -> None:
    """Log a warning if neutral class is over-predicted (sentinel check)."""
    if "neutral" not in classes:
        return
    neutral_rate = float((predictions == "neutral").mean())
    if neutral_rate > threshold:
        log.warning(
            "sentiment_neutral_inflation_detected",
            neutral_rate=round(neutral_rate, 3),
            threshold=threshold,
            note="Sentiment collapse risk: neutral class dominates predictions.",
        )
    else:
        log.info("sentiment_neutral_rate_ok", neutral_rate=round(neutral_rate, 3))
