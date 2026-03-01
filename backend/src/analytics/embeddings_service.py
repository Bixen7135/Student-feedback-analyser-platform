"""Embedding generation and persistence for analysis results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from src.storage.models import ModelMeta
from src.text_tasks.char_ngram_classifier import CharNgramClassifier
from src.text_tasks.tfidf_classifier import TfidfClassifier

_KNOWN_LABEL_COLUMNS = {"language", "sentiment_class", "detail_level"}


def compute_or_load_embeddings(
    *,
    df: pd.DataFrame,
    text_col: str,
    analysis_dir: Path,
    model_meta: ModelMeta | None = None,
    reuse_cached: bool = True,
    max_features: int = 512,
) -> dict[str, Any]:
    """Compute 2D embeddings for analysis rows or reuse cached points."""
    embeddings_dir = analysis_dir / "analytics" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    points_path = embeddings_dir / "points.csv"
    meta_path = embeddings_dir / "metadata.json"

    if reuse_cached and points_path.exists():
        points_df = pd.read_csv(points_path, dtype=str, keep_default_na=False)
        metadata = {}
        if meta_path.exists():
            try:
                metadata = orjson.loads(meta_path.read_bytes())
            except Exception:
                metadata = {}
        return _payload(points_df, points_path, metadata)

    texts = df[text_col].fillna("").astype(str).tolist()
    matrix, source = _vectorize_texts(texts, model_meta=model_meta, max_features=max_features)
    n_rows = matrix.shape[0]
    n_features = matrix.shape[1] if len(matrix.shape) > 1 else 0

    if n_rows == 0:
        reduced = pd.DataFrame(columns=["x", "y"])
    else:
        n_components = min(2, n_rows, max(1, n_features))
        pca = PCA(n_components=n_components, random_state=42)
        coords = pca.fit_transform(matrix)
        reduced = pd.DataFrame(coords, columns=[f"pc{i+1}" for i in range(n_components)])
        if "pc1" not in reduced.columns:
            reduced["pc1"] = 0.0
        if "pc2" not in reduced.columns:
            reduced["pc2"] = 0.0
        reduced = reduced.rename(columns={"pc1": "x", "pc2": "y"})[["x", "y"]]

    points_df = pd.DataFrame(
        {
            "row_idx": df.index.to_numpy(dtype=int),
            "x": reduced["x"] if not reduced.empty else [],
            "y": reduced["y"] if not reduced.empty else [],
        }
    )
    for label_col in _label_columns(df):
        points_df[label_col] = df[label_col].fillna("").astype(str).tolist()

    points_df.to_csv(points_path, index=False, encoding="utf-8-sig")
    metadata = {
        "text_col": text_col,
        "vector_source": source,
        "n_rows": int(len(points_df)),
        "n_features": int(n_features),
    }
    meta_path.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))
    return _payload(points_df, points_path, metadata)


def _vectorize_texts(
    texts: list[str],
    *,
    model_meta: ModelMeta | None,
    max_features: int,
) -> tuple[Any, str]:
    if model_meta is not None:
        try:
            if model_meta.model_type == "tfidf":
                clf = TfidfClassifier.load(Path(model_meta.storage_path) / "model.joblib")
            elif model_meta.model_type == "char_ngram":
                clf = CharNgramClassifier.load(Path(model_meta.storage_path) / "model.joblib")
            else:
                clf = None
            pipeline = getattr(clf, "_pipeline", None) if clf is not None else None
            vectorizer = pipeline.named_steps.get("tfidf") if pipeline is not None else None
            if vectorizer is not None:
                return vectorizer.transform(texts).toarray(), f"model_vectorizer:{model_meta.id}"
        except Exception:
            pass

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\S+\b",
    )
    try:
        return vectorizer.fit_transform(texts).toarray(), "dataset_tfidf"
    except ValueError:
        return np.zeros((len(texts), 1), dtype=float), "dataset_tfidf_empty"


def _label_columns(df: pd.DataFrame) -> list[str]:
    labels = [col for col in df.columns if str(col) in _KNOWN_LABEL_COLUMNS]
    pred_cols = [str(col) for col in df.columns if str(col).endswith("_pred")]
    return list(dict.fromkeys(labels + pred_cols))


def _payload(points_df: pd.DataFrame, points_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_path": str(points_path),
        "count": int(len(points_df)),
        "columns": [str(col) for col in points_df.columns],
        "metadata": metadata,
        "points": points_df.fillna("").to_dict(orient="records"),
    }
