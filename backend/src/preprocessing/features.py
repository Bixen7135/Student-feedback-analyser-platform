"""Deterministic text feature computation (character count, word count, etc.)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_WORD_RE = re.compile(r"\S+")


@dataclass
class TextFeatures:
    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float


def compute_text_features(text: str) -> TextFeatures:
    """
    Compute deterministic length-based features from text.
    These are computed AFTER preprocessing (unicode normalization, PII redaction).
    """
    if not text:
        return TextFeatures(char_count=0, word_count=0, sentence_count=0, avg_word_length=0.0)

    char_count = len(text)
    words = _WORD_RE.findall(text)
    word_count = len(words)
    # Simple sentence splitting on . ! ?
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(1, len(sentences))
    avg_word_length = (
        sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    )
    return TextFeatures(
        char_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_word_length=round(avg_word_length, 2),
    )


def add_text_features(df: pd.DataFrame, text_col: str = "text_feedback") -> pd.DataFrame:
    """Add text feature columns to a DataFrame. Mutates and returns the DataFrame."""
    features = df[text_col].apply(lambda t: compute_text_features(str(t) if pd.notna(t) else ""))
    df["char_count"] = features.apply(lambda f: f.char_count)
    df["word_count"] = features.apply(lambda f: f.word_count)
    df["sentence_count"] = features.apply(lambda f: f.sentence_count)
    df["avg_word_length"] = features.apply(lambda f: f.avg_word_length)
    return df
