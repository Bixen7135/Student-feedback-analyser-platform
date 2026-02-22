"""Abstract base class for text classifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ClassificationResult:
    """Output of a single trained text classifier."""
    task: str
    model_type: str
    predictions: np.ndarray
    probabilities: np.ndarray
    classes: list[str]
    model_path: Path
    hyperparameters: dict
    train_metrics: dict
    val_metrics: dict


class TextClassifier(ABC):
    """Abstract base for independently trained text classifiers."""

    @abstractmethod
    def fit(self, texts: list[str], labels: list[str], seed: int = 42) -> None:
        """Train the classifier. Each task has its own independent fit call."""
        ...

    @abstractmethod
    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict class labels for texts."""
        ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities for texts. Shape: (n_samples, n_classes)."""
        ...

    @property
    @abstractmethod
    def classes_(self) -> list[str]:
        """Return ordered list of class labels."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "TextClassifier":
        """Deserialize model from disk."""
        ...

    @abstractmethod
    def get_hyperparameters(self) -> dict:
        """Return hyperparameter dict for logging."""
        ...
