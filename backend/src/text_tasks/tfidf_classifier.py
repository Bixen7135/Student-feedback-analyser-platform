"""TF-IDF + Logistic Regression text classifier."""
from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from src.text_tasks.base import TextClassifier


class TfidfClassifier(TextClassifier):
    """
    TF-IDF (word n-gram) + Logistic Regression pipeline.
    Each task gets a fully independent instance with its own trained weights.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.max_iter = max_iter
        self._pipeline: Pipeline | None = None

    def fit(self, texts: list[str], labels: list[str], seed: int = 42) -> None:
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=self.sublinear_tf,
                strip_accents=None,
                analyzer="word",
                token_pattern=r"(?u)\b\S+\b",  # keep Cyrillic tokens
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
    def load(cls, path: Path) -> "TfidfClassifier":
        instance = cls.__new__(cls)
        instance._pipeline = joblib.load(path)
        # Restore hyperparams from pipeline
        tfidf = instance._pipeline.named_steps["tfidf"]
        clf = instance._pipeline.named_steps["clf"]
        instance.max_features = tfidf.max_features
        instance.ngram_range = tfidf.ngram_range
        instance.sublinear_tf = tfidf.sublinear_tf
        instance.C = clf.C
        instance.max_iter = clf.max_iter
        return instance

    def get_hyperparameters(self) -> dict:
        return {
            "model_type": "tfidf_logreg",
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "sublinear_tf": self.sublinear_tf,
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": "balanced",
            "solver": "saga",
        }

    def warm_start_from(
        self,
        base_clf: "TfidfClassifier",
        texts: list[str],
        labels: list[str],
        seed: int = 42,
    ) -> bool:
        """Re-fit the LR step initialised from base_clf's weights.

        Called after fit(). Transforms texts with the already-fitted TF-IDF
        and re-fits only the LR step, starting from the base model's coef_ /
        intercept_ (truncated / padded for dimension mismatch).

        Returns True if warm-start was applied (classes matched), False if
        class sets differ (caller should log a warning and keep fresh fit).
        """
        assert self._pipeline is not None, "Call fit() first"
        if base_clf._pipeline is None:
            return False

        base_lr = base_clf._pipeline.named_steps["clf"]
        new_lr = self._pipeline.named_steps["clf"]

        base_classes = set(str(c) for c in base_lr.classes_)
        new_classes_set = set(str(c) for c in new_lr.classes_)
        if base_classes != new_classes_set:
            return False

        base_coef = base_lr.coef_  # shape (n_coef_rows, n_feat_base)
        n_feat_new = new_lr.coef_.shape[1]
        n_feat_base = base_coef.shape[1]
        n_feat_copy = min(n_feat_new, n_feat_base)

        # Build coef_init aligned to class ordering that sklearn will assign
        # (np.unique(y) = lexicographic order; same for base since same classes).
        coef_init = np.zeros((base_coef.shape[0], n_feat_new))
        coef_init[:, :n_feat_copy] = base_coef[:, :n_feat_copy]
        intercept_init = base_lr.intercept_.copy()

        # Create warm-started LR pre-initialised with base weights
        warm_lr = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=seed,
            class_weight="balanced",
            solver="saga",
            warm_start=True,
        )
        # Pre-set so sklearn's warm_start picks them up
        warm_lr.coef_ = coef_init
        warm_lr.intercept_ = intercept_init
        warm_lr.classes_ = new_lr.classes_

        tfidf = self._pipeline.named_steps["tfidf"]
        X = tfidf.transform(texts)
        warm_lr.fit(X, labels)

        # Replace LR step in pipeline
        self._pipeline.steps[-1] = ("clf", warm_lr)
        return True
