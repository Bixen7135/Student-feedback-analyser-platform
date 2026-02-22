"""Fusion ablation studies: shuffle-text and length-only baselines."""
from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger
from src.utils.reproducibility import set_all_seeds

log = get_logger(__name__)


def shuffle_text_ablation(
    train_text: np.ndarray,
    test_text: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle text embeddings within the training set.
    If the fusion model trained on shuffled text performs similarly to the
    one with real text, it suggests text provides no real signal.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_text))
    return train_text[idx], test_text


def length_only_ablation(
    train_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
    length_col: str = "char_count",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace text embeddings with a single numeric length feature (char_count).
    Tests whether the fusion model's signal comes purely from text length.
    Uses ONLY numeric length — no lexical features, no embeddings.
    """
    import pandas as pd  # noqa

    train_len = train_df[[length_col]].fillna(0).values.astype(float)
    test_len = test_df[[length_col]].fillna(0).values.astype(float)

    # Normalize
    mean = train_len.mean()
    std = train_len.std() + 1e-8
    return (train_len - mean) / std, (test_len - mean) / std


def run_ablations(
    train_survey: np.ndarray,
    train_text: np.ndarray,
    train_targets: np.ndarray,
    test_survey: np.ndarray,
    test_text: np.ndarray,
    test_targets: np.ndarray,
    train_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
    factor_names: list[str],
    seed: int,
) -> dict:
    """Run all ablation experiments and return results dict."""
    from src.fusion.regression import train_fusion_models
    from src.evaluation.regression_metrics import metrics_to_dict

    results = {}

    # Shuffle text ablation
    log.info("ablation_shuffle_text")
    train_text_shuffled, _ = shuffle_text_ablation(train_text, test_text, seed)
    shuffle_result = train_fusion_models(
        train_survey, train_text_shuffled, train_targets,
        test_survey, test_text, test_targets,
        factor_names, seed,
    )
    results["shuffle_text"] = {
        "late_fusion_mae": shuffle_result.late_fusion.mae,
        "late_fusion_r2": shuffle_result.late_fusion.r_squared,
        "per_factor": metrics_to_dict(shuffle_result.late_fusion),
    }

    # Length-only ablation
    if "char_count" in train_df.columns:
        log.info("ablation_length_only")
        train_len, test_len = length_only_ablation(train_df, test_df)
        len_result = train_fusion_models(
            train_survey, train_len, train_targets,
            test_survey, test_len, test_targets,
            factor_names, seed,
        )
        results["length_only"] = {
            "late_fusion_mae": len_result.late_fusion.mae,
            "late_fusion_r2": len_result.late_fusion.r_squared,
            "per_factor": metrics_to_dict(len_result.late_fusion),
        }

    return results
