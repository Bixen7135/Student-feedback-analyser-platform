"""Reusable preprocessing specification helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.preprocessing.pipeline import run_preprocessing

PREPROCESS_SPEC_ID = "preprocess_v1"


@dataclass(frozen=True)
class PreprocessSpec:
    """Stable preprocessing contract for deterministic text preparation."""

    id: str = PREPROCESS_SPEC_ID
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "normalize_unicode": True,
            "normalize_punctuation": True,
            "redact_pii": True,
            "text_features": True,
            "output_text_col": "text_processed",
        }
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "params": dict(self.params),
        }


DEFAULT_PREPROCESS_SPEC = PreprocessSpec()


def apply_preprocess(
    df: pd.DataFrame,
    text_col: str,
    spec: PreprocessSpec | None = None,
) -> pd.DataFrame:
    """Run preprocessing and stamp the resulting frame with the applied spec."""
    resolved_spec = spec or DEFAULT_PREPROCESS_SPEC
    processed = run_preprocessing(df, text_col=text_col)
    processed.attrs["preprocess_spec_id"] = resolved_spec.id
    processed.attrs["preprocess_spec"] = resolved_spec.as_dict()
    return processed
