"""Training configuration dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    """Configuration for a single user-initiated training run."""

    # Split ratios — must sum to 1.0
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Class balancing strategy
    # "none"         — no balancing (raw class frequencies)
    # "class_weight" — pass class_weight='balanced' to LogisticRegression (default)
    # "oversample"   — random oversample minority classes to match majority
    class_balancing: str = "class_weight"

    # Optional classifier hyperparameter overrides (None → use class defaults)
    max_features: int | None = None
    C: float | None = None
    max_iter: int | None = None

    # Optional column overrides
    # None → auto-detect (text_feedback, text, feedback, comment, review, …)
    text_col: str | None = None
    # None → use TASK_LABEL_COLS[task] (language, sentiment_class, detail_level)
    label_col: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "class_balancing": self.class_balancing,
            "max_features": self.max_features,
            "C": self.C,
            "max_iter": self.max_iter,
            "text_col": self.text_col,
            "label_col": self.label_col,
        }
