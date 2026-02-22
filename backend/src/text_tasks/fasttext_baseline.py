"""
fastText language identification baseline.
Compares fastText lid.176.bin predictions against provided language labels.
This is an optional comparison — skipped gracefully if the model is absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)

# Default path for fastText language model
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "lid.176.bin"


def _map_fasttext_to_local(label: str) -> str:
    """Map fastText language code to local label (ru/kz/mixed)."""
    label = label.replace("__label__", "").lower().strip()
    if label in ("ru", "rus"):
        return "ru"
    if label in ("kk", "kaz", "kz"):
        return "kz"
    return "mixed"


def run_fasttext_language_detection(
    texts: list[str],
    true_labels: list[str],
    model_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run fastText language identification and compare to provided labels.
    Returns agreement metrics dict. Skips gracefully if model is absent.
    """
    resolved_path = model_path or _DEFAULT_MODEL_PATH

    if not resolved_path.exists():
        log.warning(
            "fasttext_model_not_found",
            path=str(resolved_path),
            note="fastText baseline skipped. Download lid.176.bin to enable.",
        )
        return {"status": "skipped", "reason": f"Model not found: {resolved_path}"}

    try:
        import fasttext  # type: ignore
    except ImportError:
        log.warning("fasttext_not_installed", note="Install fasttext-wheel to enable.")
        return {"status": "skipped", "reason": "fasttext not installed"}

    try:
        model = fasttext.load_model(str(resolved_path))
        predictions = []
        for text in texts:
            clean = text.replace("\n", " ").strip()
            if not clean:
                predictions.append("mixed")
                continue
            label, _ = model.predict(clean, k=1)
            predictions.append(_map_fasttext_to_local(label[0]))

        preds = np.array(predictions)
        truth = np.array(true_labels)

        from src.evaluation.classification_metrics import compute_classification_metrics
        metrics = compute_classification_metrics(truth, preds, ["ru", "kz", "mixed"])

        return {
            "status": "completed",
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
            "per_class_f1": metrics.per_class_f1,
            "confusion_matrix": metrics.confusion_matrix.tolist(),
        }
    except Exception as e:
        log.error("fasttext_baseline_failed", error=str(e))
        return {"status": "error", "reason": str(e)}
