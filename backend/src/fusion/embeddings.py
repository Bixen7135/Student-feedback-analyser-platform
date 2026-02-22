"""TF-IDF text embedding extraction for fusion experiments."""
from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from src.utils.logging import get_logger

log = get_logger(__name__)


def extract_tfidf_embeddings(
    train_texts: list[str],
    eval_texts: list[str],
    max_features: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Fit TF-IDF on train_texts and transform both train and eval sets.
    Returns dense arrays (memory permitting) for use in regression.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents=None,
        analyzer="word",
        token_pattern=r"(?u)\b\S+\b",
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    eval_matrix = vectorizer.transform(eval_texts)

    # Use sparse arrays — convert to dense only for small datasets or explicitly
    log.info(
        "embeddings_extracted",
        train_shape=train_matrix.shape,
        eval_shape=eval_matrix.shape,
    )
    return train_matrix, eval_matrix, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Path) -> TfidfVectorizer:
    return joblib.load(path)
