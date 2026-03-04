"""Global feature-importance helpers for linear text models."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.storage.model_registry import resolve_model_artifact_path
from src.storage.models import ModelMeta
from src.text_tasks.char_ngram_classifier import CharNgramClassifier
from src.text_tasks.tfidf_classifier import TfidfClassifier
from src.training.contract import MODEL_TYPE_XLM_ROBERTA


def load_linear_text_components(model_meta: ModelMeta) -> dict[str, Any]:
    """Load vectorizer, classifier, and ordered coefficient rows for a text model."""
    if model_meta.model_type == MODEL_TYPE_XLM_ROBERTA:
        raise ValueError("Feature importance is not supported for xlm_roberta models.")

    model_path = resolve_model_artifact_path(model_meta.storage_path)
    if model_meta.model_type == "tfidf":
        clf = TfidfClassifier.load(model_path)
    elif model_meta.model_type == "char_ngram":
        clf = CharNgramClassifier.load(model_path)
    else:
        raise ValueError(f"Unsupported model_type for explanations: {model_meta.model_type}")

    pipeline = getattr(clf, "_pipeline", None)
    if pipeline is None:
        raise ValueError("Model pipeline is unavailable.")

    vectorizer = pipeline.named_steps.get("tfidf")
    linear_model = pipeline.named_steps.get("clf")
    if vectorizer is None or linear_model is None:
        raise ValueError("Model does not contain the expected tfidf + linear classifier pipeline.")

    classes = [str(value) for value in getattr(linear_model, "classes_", [])]
    feature_names = _feature_names(vectorizer)
    coef = getattr(linear_model, "coef_", None)
    if coef is None:
        raise ValueError("Model does not expose linear coefficients.")

    coef_matrix = np.asarray(coef, dtype=float)
    if coef_matrix.ndim != 2:
        raise ValueError("Unexpected coefficient shape.")

    # Binary LogisticRegression exposes a single row for the positive class.
    if coef_matrix.shape[0] == 1 and len(classes) == 2:
        coef_matrix = np.vstack([-coef_matrix[0], coef_matrix[0]])

    if coef_matrix.shape[0] != len(classes):
        raise ValueError("Coefficient rows do not align with class labels.")

    if coef_matrix.shape[1] != len(feature_names):
        raise ValueError("Coefficient columns do not align with feature names.")

    return {
        "pipeline": pipeline,
        "vectorizer": vectorizer,
        "linear_model": linear_model,
        "classes": classes,
        "feature_names": feature_names,
        "coef_matrix": coef_matrix,
    }


def get_global_feature_importance(
    model_meta: ModelMeta,
    top_n: int = 20,
) -> dict[str, Any]:
    """Return top positively weighted features per class."""
    components = load_linear_text_components(model_meta)
    feature_names = components["feature_names"]
    coef_matrix = components["coef_matrix"]
    classes = components["classes"]

    per_class: dict[str, Any] = {}
    for class_name, weights in zip(classes, coef_matrix, strict=False):
        top_indices = np.argsort(weights)[::-1][:top_n]
        per_class[class_name] = [
            {
                "feature": feature_names[int(idx)],
                "weight": round(float(weights[int(idx)]), 6),
            }
            for idx in top_indices
        ]

    return {
        "model_id": model_meta.id,
        "task": model_meta.task,
        "model_type": model_meta.model_type,
        "classes": classes,
        "top_n": int(top_n),
        "per_class": per_class,
    }


def _feature_names(vectorizer: Any) -> list[str]:
    names = vectorizer.get_feature_names_out()
    return [str(value) for value in names]
