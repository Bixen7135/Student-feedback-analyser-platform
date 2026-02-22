"""Stratified data splitting for reproducible train/val/test sets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.logging import get_logger

log = get_logger(__name__)


def stratified_split(
    df: pd.DataFrame,
    stratify_col: str = "sentiment_class",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train / val / test with stratification.

    Strategy: first split into (train+val) and test, then split (train+val) into train and val.
    No group-aware split (no grouping variable available — see DECISIONS.md).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"

    # Remove rows with missing stratify label
    df = df.dropna(subset=[stratify_col]).reset_index(drop=True)

    # Step 1: Split into (train+val) and test
    test_frac = test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(df, df[stratify_col]))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: Split (train+val) into train and val
    val_frac_of_trainval = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=seed)
    train_idx, val_idx = next(sss2.split(df_trainval, df_trainval[stratify_col]))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    log.info(
        "split_complete",
        train=len(df_train),
        val=len(df_val),
        test=len(df_test),
        stratify_col=stratify_col,
    )
    return df_train, df_val, df_test


def validate_split_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str = "survey_id",
) -> None:
    """Assert no survey_id appears in more than one split."""
    if id_col not in train.columns:
        return
    train_ids = set(train[id_col].dropna())
    val_ids = set(val[id_col].dropna())
    test_ids = set(test[id_col].dropna())

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids

    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Split leakage detected! "
            f"train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )


def get_split_distribution(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Return class distribution as fractions."""
    counts = df[col].value_counts(normalize=True)
    return counts.to_dict()
