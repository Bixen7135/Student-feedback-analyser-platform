"""Fusion pipeline runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson  # type: ignore
import pandas as pd

from src.fusion.embeddings import extract_tfidf_embeddings, save_vectorizer
from src.fusion.regression import train_fusion_models, save_fusion_models, FusionResult
from src.fusion.ablations import run_ablations
from src.evaluation.regression_metrics import metrics_to_dict
from src.utils.logging import get_logger

log = get_logger(__name__)

ITEM_COLS = [f"item_{i}" for i in range(1, 10)]


def run_fusion(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    factor_scores_train: pd.DataFrame,
    factor_scores_test: pd.DataFrame,
    run_dir: Path,
    seed: int,
    text_col: str = "text_processed",
    embedding_max_features: int = 5000,
    huber_epsilon: float = 1.35,
    huber_max_iter: int = 500,
) -> dict[str, Any]:
    """
    Full fusion pipeline:
    1. Extract TF-IDF text embeddings
    2. Train survey-only, text-only, and late-fusion regression models
    3. Run ablation studies
    4. Save all results to run_dir/fusion/

    Returns full results dict.
    """
    fusion_dir = run_dir / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    factor_names = list(factor_scores_train.columns)
    log.info("fusion_start", factor_names=factor_names, text_col=text_col)

    # Extract survey features (9 items)
    train_survey = train_df[ITEM_COLS].fillna(0).astype(float).values
    test_survey = test_df[ITEM_COLS].fillna(0).astype(float).values

    # Extract text embeddings
    train_texts = train_df[text_col].fillna("").tolist()
    test_texts = test_df[text_col].fillna("").tolist()
    train_text, test_text, vectorizer = extract_tfidf_embeddings(
        train_texts, test_texts, max_features=embedding_max_features, seed=seed
    )

    # Align factor scores to dataframe indices
    train_targets = factor_scores_train.values
    test_targets = factor_scores_test.values

    # Train and evaluate fusion models
    fusion_result = train_fusion_models(
        train_survey=train_survey,
        train_text=train_text,
        train_targets=train_targets,
        test_survey=test_survey,
        test_text=test_text,
        test_targets=test_targets,
        factor_names=factor_names,
        seed=seed,
        huber_epsilon=huber_epsilon,
        huber_max_iter=huber_max_iter,
    )

    # Run ablations
    log.info("fusion_running_ablations")
    ablation_results = run_ablations(
        train_survey=train_survey,
        train_text=train_text,
        train_targets=train_targets,
        test_survey=test_survey,
        test_text=test_text,
        test_targets=test_targets,
        train_df=train_df,
        test_df=test_df,
        factor_names=factor_names,
        seed=seed,
    )

    # Save vectorizer
    save_vectorizer(vectorizer, fusion_dir / "tfidf_vectorizer.joblib")

    # Compile results
    results: dict[str, Any] = {
        "factor_names": factor_names,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "survey_only": metrics_to_dict(fusion_result.survey_only),
        "text_only": metrics_to_dict(fusion_result.text_only),
        "late_fusion": metrics_to_dict(fusion_result.late_fusion),
        "delta_mae": fusion_result.delta_mae,
        "delta_r2": fusion_result.delta_r2,
        "ablations": ablation_results,
        "note": (
            "Positive delta_mae means fusion is WORSE than survey-only. "
            "Negative delta_mae means fusion IMPROVES over survey-only. "
            "Null results reported without spin."
        ),
    }

    # Save results
    results_path = fusion_dir / "results.json"
    results_path.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    log.info("fusion_done", path=str(results_path))

    return results
