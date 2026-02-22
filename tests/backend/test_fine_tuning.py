"""Tests for Phase 4: fine-tuning (warm-start) and model lineage."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import sys
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.storage.database import Database
from src.storage.dataset_manager import DatasetManager
from src.storage.model_registry import ModelRegistry
from src.text_tasks.tfidf_classifier import TfidfClassifier
from src.text_tasks.char_ngram_classifier import CharNgramClassifier
from src.training.config import TrainingConfig
from src.training.runner import LabelValidationError, run_training


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_df(n_per_class: int = 15) -> pd.DataFrame:
    rows = []
    for sentiment in ["positive", "neutral", "negative"]:
        for i in range(n_per_class):
            rows.append(
                {
                    "text_feedback": f"Some feedback text number {i} class {sentiment}",
                    "language": "ru" if i % 2 == 0 else "kz",
                    "sentiment_class": sentiment,
                    "detail_level": "short" if i < 5 else ("medium" if i < 10 else "long"),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df():
    return _make_df(15)


@pytest.fixture
def tmp_dirs(tmp_path):
    db = Database(tmp_path / "test.db")
    dm = DatasetManager(db, tmp_path / "datasets")
    reg = ModelRegistry(db, tmp_path / "models")
    arts = tmp_path / "training_runs"
    return dm, reg, arts


@pytest.fixture
def uploaded_dataset(sample_df, tmp_dirs):
    dm, reg, arts = tmp_dirs
    csv_bytes = sample_df.to_csv(index=False).encode()
    csv_path = Path(dm.datasets_dir) / "tmp_input.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_bytes)
    meta = dm.upload_dataset(csv_path, name="test_dataset")
    return meta.id


# ---------------------------------------------------------------------------
# warm_start_from — unit tests on classifiers directly
# ---------------------------------------------------------------------------


class TestWarmStartFrom:
    """Test TfidfClassifier.warm_start_from() directly."""

    def _make_texts_labels(self, n: int = 30):
        texts = [f"text sample {i}" for i in range(n)]
        labels = (["positive"] * (n // 3) + ["negative"] * (n // 3) + ["neutral"] * (n // 3))[:n]
        return texts, labels

    def test_warm_start_succeeds_matching_classes(self):
        texts, labels = self._make_texts_labels(30)
        # Train base model
        base = TfidfClassifier()
        base.fit(texts, labels, seed=0)
        # Train new model from scratch
        new = TfidfClassifier()
        new.fit(texts, labels, seed=1)
        # Warm-start from base
        ok = new.warm_start_from(base, texts, labels, seed=1)
        assert ok is True

    def test_warm_start_returns_false_on_class_mismatch(self):
        texts_base = [f"text {i}" for i in range(20)]
        labels_base = ["positive"] * 10 + ["negative"] * 10  # 2 classes
        base = TfidfClassifier()
        base.fit(texts_base, labels_base, seed=0)

        texts_new = [f"text {i}" for i in range(30)]
        labels_new = ["positive"] * 10 + ["negative"] * 10 + ["neutral"] * 10  # 3 classes
        new = TfidfClassifier()
        new.fit(texts_new, labels_new, seed=1)
        ok = new.warm_start_from(base, texts_new, labels_new, seed=1)
        assert ok is False

    def test_warm_start_preserves_prediction_capability(self):
        texts, labels = self._make_texts_labels(30)
        base = TfidfClassifier()
        base.fit(texts, labels, seed=0)
        new = TfidfClassifier()
        new.fit(texts, labels, seed=1)
        new.warm_start_from(base, texts, labels, seed=1)
        # After warm-start, predictions should still work
        preds = new.predict(["some random text"])
        assert len(preds) == 1
        assert preds[0] in ["positive", "negative", "neutral"]

    def test_char_ngram_warm_start_succeeds(self):
        texts, labels = self._make_texts_labels(30)
        base = CharNgramClassifier()
        base.fit(texts, labels, seed=0)
        new = CharNgramClassifier()
        new.fit(texts, labels, seed=1)
        ok = new.warm_start_from(base, texts, labels, seed=1)
        assert ok is True

    def test_warm_start_requires_fit_first(self):
        texts, labels = self._make_texts_labels(30)
        base = TfidfClassifier()
        base.fit(texts, labels, seed=0)
        new = TfidfClassifier()  # NOT fitted
        with pytest.raises(AssertionError):
            new.warm_start_from(base, texts, labels, seed=0)


# ---------------------------------------------------------------------------
# run_training with base_model_id
# ---------------------------------------------------------------------------


class TestFineTuning:
    def test_fine_tuning_succeeds_same_task_and_type(
        self, uploaded_dataset, tmp_dirs
    ):
        dm, reg, arts = tmp_dirs
        # Train base model
        base_result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
            name="base_model",
        )
        base_model_id = base_result["model_id"]

        # Fine-tune from base model
        ft_result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=1,
            name="fine_tuned_model",
            base_model_id=base_model_id,
        )
        assert ft_result["model_id"] is not None
        assert ft_result["model_id"] != base_model_id

    def test_fine_tuning_stores_base_model_id_in_registry(
        self, uploaded_dataset, tmp_dirs
    ):
        dm, reg, arts = tmp_dirs
        base_result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        ft_result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=1,
            base_model_id=base_result["model_id"],
        )
        ft_model = reg.get_model(ft_result["model_id"])
        assert ft_model is not None
        assert ft_model.base_model_id == base_result["model_id"]

    def test_fine_tuning_includes_comparison_metrics(
        self, uploaded_dataset, tmp_dirs
    ):
        dm, reg, arts = tmp_dirs
        base_result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        ft_result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=1,
            base_model_id=base_result["model_id"],
        )
        ft_info = ft_result["metrics"].get("fine_tuning")
        assert ft_info is not None
        assert ft_info["base_model_id"] == base_result["model_id"]
        assert "base_model_test" in ft_info
        assert "new_model_test" in ft_info
        assert "delta_macro_f1" in ft_info
        assert "delta_accuracy" in ft_info
        assert "warm_started" in ft_info

    def test_fine_tuning_via_config_base_model_id(
        self, uploaded_dataset, tmp_dirs
    ):
        """base_model_id can be passed via TrainingConfig as well."""
        dm, reg, arts = tmp_dirs
        base_result = run_training(
            dataset_id=uploaded_dataset,
            task="language",
            model_type="char_ngram",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        cfg = TrainingConfig(base_model_id=base_result["model_id"])
        ft_result = run_training(
            dataset_id=uploaded_dataset,
            task="language",
            model_type="char_ngram",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            config=cfg,
            seed=1,
        )
        ft_model = reg.get_model(ft_result["model_id"])
        assert ft_model.base_model_id == base_result["model_id"]

    def test_fine_tuning_raises_on_missing_base_model(
        self, uploaded_dataset, tmp_dirs
    ):
        dm, reg, arts = tmp_dirs
        with pytest.raises(ValueError, match="Base model not found"):
            run_training(
                dataset_id=uploaded_dataset,
                task="sentiment",
                model_type="tfidf",
                dataset_manager=dm,
                model_registry=reg,
                artifacts_dir=arts,
                base_model_id="nonexistent-model-id",
            )

    def test_fine_tuning_raises_on_task_mismatch(
        self, uploaded_dataset, tmp_dirs
    ):
        dm, reg, arts = tmp_dirs
        base_result = run_training(
            dataset_id=uploaded_dataset,
            task="language",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        with pytest.raises(ValueError, match="Base model task"):
            run_training(
                dataset_id=uploaded_dataset,
                task="sentiment",
                model_type="tfidf",
                dataset_manager=dm,
                model_registry=reg,
                artifacts_dir=arts,
                base_model_id=base_result["model_id"],
            )

    def test_fine_tuning_raises_on_model_type_mismatch(
        self, uploaded_dataset, tmp_dirs
    ):
        dm, reg, arts = tmp_dirs
        base_result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        with pytest.raises(ValueError, match="Base model type"):
            run_training(
                dataset_id=uploaded_dataset,
                task="sentiment",
                model_type="char_ngram",
                dataset_manager=dm,
                model_registry=reg,
                artifacts_dir=arts,
                base_model_id=base_result["model_id"],
            )

    def test_no_base_model_id_still_works(self, uploaded_dataset, tmp_dirs):
        """Backward compat: run_training without base_model_id is unchanged."""
        dm, reg, arts = tmp_dirs
        result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=42,
        )
        assert result["model_id"] is not None
        assert "fine_tuning" not in result["metrics"]
        model = reg.get_model(result["model_id"])
        assert model.base_model_id is None


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


class TestModelLineage:
    def test_lineage_single_model(self, uploaded_dataset, tmp_dirs):
        dm, reg, arts = tmp_dirs
        result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        chain = reg.get_lineage(result["model_id"])
        assert len(chain) == 1
        assert chain[0].id == result["model_id"]

    def test_lineage_two_generations(self, uploaded_dataset, tmp_dirs):
        dm, reg, arts = tmp_dirs
        base = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        fine_tuned = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=1,
            base_model_id=base["model_id"],
        )
        chain = reg.get_lineage(fine_tuned["model_id"])
        assert len(chain) == 2
        assert chain[0].id == fine_tuned["model_id"]
        assert chain[1].id == base["model_id"]
        assert chain[1].base_model_id is None

    def test_lineage_three_generations(self, uploaded_dataset, tmp_dirs):
        dm, reg, arts = tmp_dirs
        r1 = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        r2 = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=1,
            base_model_id=r1["model_id"],
        )
        r3 = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=2,
            base_model_id=r2["model_id"],
        )
        chain = reg.get_lineage(r3["model_id"])
        assert len(chain) == 3
        assert [m.id for m in chain] == [r3["model_id"], r2["model_id"], r1["model_id"]]

    def test_lineage_nonexistent_model_returns_empty(self, tmp_dirs):
        dm, reg, arts = tmp_dirs
        chain = reg.get_lineage("nonexistent-id")
        assert chain == []
