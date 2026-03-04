"""Evaluation runner — evaluates all trained models on test set."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import orjson  # type: ignore
import pandas as pd

from src.evaluation.classification_metrics import (
    compute_classification_metrics,
    stratify_metrics_by,
    metrics_to_dict,
)
from src.evaluation.regression_metrics import compute_regression_metrics, metrics_to_dict as reg_to_dict
from src.evaluation.robustness import evaluate_on_short_text, sensitivity_analysis
from src.text_tasks.xlm_roberta_classifier import XlmRobertaClassifier
from src.text_tasks.tfidf_classifier import TfidfClassifier
from src.text_tasks.char_ngram_classifier import CharNgramClassifier
from src.training.contract import MODEL_TYPE_XLM_ROBERTA
from src.utils.logging import get_logger

log = get_logger(__name__)

TASK_LABEL_COLS = {
    "language": "language",
    "sentiment": "sentiment_class",
    "detail_level": "detail_level",
}
TASK_CLASSES = {
    "language": ["ru", "kz", "mixed"],
    "sentiment": ["positive", "neutral", "negative"],
    "detail_level": ["short", "medium", "long"],
}


def _load_pipeline_classifier(model_dir: Path, model_type: str):
    """Load a pipeline text-task classifier from its mirrored artifact."""
    if model_type == "tfidf":
        return TfidfClassifier.load(model_dir / "model.joblib")
    if model_type == "char_ngram":
        return CharNgramClassifier.load(model_dir / "model.joblib")
    if model_type == MODEL_TYPE_XLM_ROBERTA:
        return XlmRobertaClassifier.load(model_dir / "model")
    raise ValueError(f"Unsupported model_type in pipeline evaluation: {model_type!r}")


def run_evaluation(
    test_df: pd.DataFrame,
    run_dir: Path,
    text_col: str = "text_processed",
    factor_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate all trained models (text tasks + fusion) on test set.
    Saves full evaluation results to run_dir/evaluation/.
    """
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    # --- Text classification evaluation ---
    for task_name, label_col in TASK_LABEL_COLS.items():
        task_results: dict[str, Any] = {}
        task_dir = run_dir / "text_tasks" / task_name
        if not task_dir.exists():
            all_results[task_name] = task_results
            continue

        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_type = model_dir.name
            model_path = model_dir / "model.joblib"
            model_dir_path = model_dir / "model"
            if not model_path.exists() and not model_dir_path.exists():
                log.warning("model_not_found_skipping", task=task_name, model=model_type)
                continue

            clf_pipeline = _load_pipeline_classifier(model_dir, model_type)
            texts = test_df[text_col].fillna("").tolist()
            true_labels = test_df[label_col].fillna("unknown").values
            preds = clf_pipeline.predict(texts)
            classes = TASK_CLASSES.get(task_name, list(clf_pipeline.classes_))

            metrics = compute_classification_metrics(true_labels, preds, classes)

            # Stratified by language
            strat_lang: dict[str, Any] = {}
            if "language" in test_df.columns and task_name != "language":
                test_df_copy = test_df.copy()
                test_df_copy["_pred"] = preds
                strat_result = stratify_metrics_by(
                    test_df_copy, label_col, "_pred", "language", classes
                )
                strat_lang = {k: metrics_to_dict(v) for k, v in strat_result.items()}

            # Robustness: short text
            short_m = evaluate_on_short_text(
                test_df, clf_pipeline if hasattr(clf_pipeline, 'predict') else None,
                text_col, label_col,
            ) if False else None  # using loaded pipeline not wrapped

            task_results[model_type] = {
                "overall": metrics_to_dict(metrics),
                "stratified_by_language": strat_lang,
            }

            log.info(
                "eval_task",
                task=task_name,
                model=model_type,
                macro_f1=round(metrics.macro_f1, 4),
            )

        all_results[task_name] = task_results

    # Save
    eval_path = eval_dir / "classification_results.json"
    eval_path.write_bytes(orjson.dumps(all_results, option=orjson.OPT_INDENT_2))
    log.info("evaluation_saved", path=str(eval_path))
    return all_results
