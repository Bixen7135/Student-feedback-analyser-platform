"""Character n-gram + Logistic Regression text classifier."""
from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from src.text_tasks.base import TextClassifier


class CharNgramClassifier(TextClassifier):
    """
    Character n-gram TF-IDF + Logistic Regression.
    Uses character-level features which are more robust to morphological variation
    in Russian/Kazakh text and better handle OOV words.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 5),
        max_features: int = 100_000,
        analyzer: str = "char_wb",  # char_wb respects word boundaries
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.analyzer = analyzer
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.max_iter = max_iter
        self._pipeline: Pipeline | None = None

    def fit(self, texts: list[str], labels: list[str], seed: int = 42) -> None:
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                sublinear_tf=self.sublinear_tf,
                strip_accents=None,
            )),
            ("clf", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=seed,
                class_weight="balanced",
                solver="saga",
            )),
        ])
        self._pipeline.fit(texts, labels)

    def predict(self, texts: list[str]) -> np.ndarray:
        assert self._pipeline is not None, "Call fit() first"
        return self._pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self._pipeline is not None, "Call fit() first"
        return self._pipeline.predict_proba(texts)

    @property
    def classes_(self) -> list[str]:
        assert self._pipeline is not None, "Call fit() first"
        return [str(c) for c in self._pipeline.named_steps["clf"].classes_]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "CharNgramClassifier":
        instance = cls.__new__(cls)
        instance._pipeline = joblib.load(path)
        tfidf = instance._pipeline.named_steps["tfidf"]
        clf = instance._pipeline.named_steps["clf"]
        instance.ngram_range = tfidf.ngram_range
        instance.max_features = tfidf.max_features
        instance.analyzer = tfidf.analyzer
        instance.sublinear_tf = tfidf.sublinear_tf
        instance.C = clf.C
        instance.max_iter = clf.max_iter
        return instance

    def get_hyperparameters(self) -> dict:
        return {
            "model_type": "char_ngram_logreg",
            "analyzer": self.analyzer,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": "balanced",
        }
