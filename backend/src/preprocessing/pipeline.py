"""Full preprocessing pipeline — normalize, redact, add features."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.normalize import preprocess_text
from src.preprocessing.redact import redact_pii
from src.preprocessing.features import add_text_features
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_preprocessing(df: pd.DataFrame, text_col: str = "text_feedback") -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame:
    1. Unicode + punctuation normalization
    2. PII redaction
    3. Deterministic text feature computation

    Returns a new DataFrame with `text_processed` column and feature columns added.
    """
    log.info("preprocessing_start", rows=len(df))
    df = df.copy()

    # Step 1 & 2: normalize + redact
    def _process(text: str) -> str:
        normalized = preprocess_text(str(text) if pd.notna(text) else "")
        return redact_pii(normalized)

    df["text_processed"] = df[text_col].apply(_process)

    # Step 3: deterministic features
    df = add_text_features(df, text_col="text_processed")

    log.info("preprocessing_done", rows=len(df))
    return df


def save_preprocessed(df: pd.DataFrame, run_dir: Path) -> Path:
    """Save preprocessed DataFrame to parquet under run_dir/preprocessing/."""
    out_dir = run_dir / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessed.parquet"
    df.to_parquet(out_path, index=False)
    log.info("preprocessed_saved", path=str(out_path))
    return out_path
