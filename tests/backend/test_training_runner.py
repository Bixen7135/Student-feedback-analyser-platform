"""Unit tests for the training runner — runner logic, validation, config."""
from __future__ import annotations

import io
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
from src.training.config import TrainingConfig
from src.training.runner import (
    LabelValidationError,
    _detect_text_col,
    _validate_labels,
    _apply_class_balancing,
    run_training,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n_per_class: int = 15) -> pd.DataFrame:
    """Return a small DataFrame with text + 3 label columns, each class balanced."""
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
    """Upload sample_df to DatasetManager, return dataset_id."""
    dm, reg, arts = tmp_dirs
    csv_bytes = sample_df.to_csv(index=False).encode()
    csv_path = Path(dm.datasets_dir) / "tmp_input.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_bytes)
    meta = dm.upload_dataset(csv_path, name="test_dataset")
    return meta.id


# ---------------------------------------------------------------------------
# _detect_text_col
# ---------------------------------------------------------------------------


class TestDetectTextCol:
    def test_finds_text_feedback(self):
        df = pd.DataFrame({"text_feedback": ["a"], "x": [1]})
        assert _detect_text_col(df) == "text_feedback"

    def test_finds_text(self):
        df = pd.DataFrame({"text": ["a"], "x": [1]})
        assert _detect_text_col(df) == "text"

    def test_priority_text_processed_over_text(self):
        df = pd.DataFrame({"text_processed": ["a"], "text": ["b"]})
        assert _detect_text_col(df) == "text_processed"

    def test_falls_back_to_first_object_col(self):
        df = pd.DataFrame({"col_a": ["a"], "col_b": [1]})
        assert _detect_text_col(df) == "col_a"

    def test_raises_if_no_text_col(self):
        df = pd.DataFrame({"num1": [1, 2], "num2": [3, 4]})
        with pytest.raises(LabelValidationError, match="No text column"):
            _detect_text_col(df)


# ---------------------------------------------------------------------------
# _validate_labels
# ---------------------------------------------------------------------------


class TestValidateLabels:
    def test_happy_path_returns_cleaned_df(self, sample_df):
        result = _validate_labels(sample_df, "sentiment_class", "sentiment")
        assert len(result) == len(sample_df)
        assert result["sentiment_class"].isna().sum() == 0

    def test_drops_nan_rows(self):
        # 6 positive + 6 negative = 12 valid rows + 1 null row; null should be dropped
        df = pd.DataFrame(
            {
                "text_feedback": [f"t{i}" for i in range(13)],
                "sentiment_class": ["positive"] * 6 + ["negative"] * 6 + [None],
            }
        )
        result = _validate_labels(df, "sentiment_class", "sentiment")
        assert len(result) == 12

    def test_raises_on_missing_column(self, sample_df):
        with pytest.raises(LabelValidationError, match="not found"):
            _validate_labels(sample_df, "nonexistent_col", "sentiment")

    def test_raises_when_all_labels_missing(self):
        df = pd.DataFrame(
            {"text_feedback": ["a", "b"], "sentiment_class": [None, None]}
        )
        with pytest.raises(LabelValidationError, match="No valid labels"):
            _validate_labels(df, "sentiment_class", "sentiment")

    def test_raises_when_only_one_class(self):
        df = pd.DataFrame(
            {
                "text_feedback": [f"t{i}" for i in range(20)],
                "sentiment_class": ["positive"] * 20,
            }
        )
        with pytest.raises(LabelValidationError, match="at least 2 distinct classes"):
            _validate_labels(df, "sentiment_class", "sentiment")

    def test_removes_small_classes_and_warns(self):
        # 3 classes: 'a' × 20, 'b' × 20, 'c' × 2 (below MIN_SAMPLES_PER_CLASS=5)
        df = pd.DataFrame(
            {
                "text_feedback": [f"t{i}" for i in range(42)],
                "sentiment_class": ["a"] * 20 + ["b"] * 20 + ["c"] * 2,
            }
        )
        result = _validate_labels(df, "sentiment_class", "sentiment")
        assert "c" not in result["sentiment_class"].values
        assert len(result) == 40

    def test_raises_after_small_class_removal_leaves_one_class(self):
        # After removing 'b' (3 samples), only 'a' remains → fail
        df = pd.DataFrame(
            {
                "text_feedback": [f"t{i}" for i in range(23)],
                "sentiment_class": ["a"] * 20 + ["b"] * 3,
            }
        )
        with pytest.raises(LabelValidationError, match="only 1 class"):
            _validate_labels(df, "sentiment_class", "sentiment")


# ---------------------------------------------------------------------------
# _apply_class_balancing
# ---------------------------------------------------------------------------


class TestApplyClassBalancing:
    def test_none_returns_unchanged(self, sample_df):
        result = _apply_class_balancing(sample_df, "sentiment_class", "none", 42)
        assert len(result) == len(sample_df)

    def test_class_weight_returns_unchanged(self, sample_df):
        result = _apply_class_balancing(
            sample_df, "sentiment_class", "class_weight", 42
        )
        assert len(result) == len(sample_df)

    def test_oversample_balances_classes(self):
        df = pd.DataFrame(
            {
                "text_feedback": [f"t{i}" for i in range(25)],
                "sentiment_class": ["positive"] * 10 + ["negative"] * 10 + ["neutral"] * 5,
            }
        )
        result = _apply_class_balancing(df, "sentiment_class", "oversample", 42)
        counts = result["sentiment_class"].value_counts()
        assert counts.max() == counts.min(), "All classes should be equal after oversampling"

    def test_invalid_strategy_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unknown class_balancing"):
            _apply_class_balancing(sample_df, "sentiment_class", "bogus", 42)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.train_ratio == 0.80
        assert cfg.val_ratio == 0.10
        assert cfg.class_balancing == "class_weight"

    def test_to_dict_round_trip(self):
        cfg = TrainingConfig(train_ratio=0.7, C=0.5, max_features=1000)
        d = cfg.to_dict()
        assert d["train_ratio"] == 0.7
        assert d["C"] == 0.5
        assert d["max_features"] == 1000
        assert d["label_col"] is None


# ---------------------------------------------------------------------------
# run_training — full integration
# ---------------------------------------------------------------------------


class TestRunTraining:
    def test_trains_sentiment_tfidf_and_registers(
        self, uploaded_dataset, tmp_dirs
    ):
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
        assert result["metrics"]["task"] == "sentiment"
        assert result["metrics"]["model_type"] == "tfidf"
        assert 0.0 <= result["metrics"]["val"]["macro_f1"] <= 1.0

        # Verify model registered in registry
        model = reg.get_model(result["model_id"])
        assert model is not None
        assert model.task == "sentiment"
        assert model.dataset_id == uploaded_dataset
        assert model.input_signature["text"]["source_column"] == "text_feedback"
        assert model.input_signature["text"]["model_input_column"] == "text_model_input"
        assert model.preprocess_spec["id"] == "preprocess_v1"
        assert model.training_profile["vocabulary_size"] > 0
        assert result["metrics"]["text_col"] == "text_model_input"
        assert result["metrics"]["preprocess_spec_id"] == "preprocess_v1"

    def test_trains_language_char_ngram(self, uploaded_dataset, tmp_dirs):
        dm, reg, arts = tmp_dirs
        result = run_training(
            dataset_id=uploaded_dataset,
            task="language",
            model_type="char_ngram",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=0,
        )
        assert result["metrics"]["task"] == "language"
        assert result["metrics"]["model_type"] == "char_ngram"

    def test_custom_name_stored_in_registry(self, uploaded_dataset, tmp_dirs):
        dm, reg, arts = tmp_dirs
        result = run_training(
            dataset_id=uploaded_dataset,
            task="detail_level",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            name="my_custom_model",
            seed=42,
        )
        assert result["model_name"] == "my_custom_model"

    def test_raises_on_invalid_task(self, uploaded_dataset, tmp_dirs):
        dm, reg, arts = tmp_dirs
        with pytest.raises(ValueError, match="Unknown task"):
            run_training(
                dataset_id=uploaded_dataset,
                task="bogus_task",
                model_type="tfidf",
                dataset_manager=dm,
                model_registry=reg,
                artifacts_dir=arts,
            )

    def test_raises_on_invalid_model_type(self, uploaded_dataset, tmp_dirs):
        dm, reg, arts = tmp_dirs
        with pytest.raises(ValueError, match="Unknown model_type"):
            run_training(
                dataset_id=uploaded_dataset,
                task="sentiment",
                model_type="random_forest",
                dataset_manager=dm,
                model_registry=reg,
                artifacts_dir=arts,
            )

    def test_raises_when_label_col_absent(self, tmp_dirs):
        """Dataset without sentiment_class should fail clearly."""
        dm, reg, arts = tmp_dirs
        df = pd.DataFrame(
            {
                "text_feedback": [f"text {i}" for i in range(30)],
                "language": ["ru"] * 30,
                # no sentiment_class column
            }
        )
        csv_bytes = df.to_csv(index=False).encode()
        csv_path = Path(dm.datasets_dir) / "no_sentiment.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_bytes(csv_bytes)
        meta = dm.upload_dataset(csv_path, name="no_sentiment_dataset")

        with pytest.raises(LabelValidationError, match="sentiment_class"):
            run_training(
                dataset_id=meta.id,
                task="sentiment",
                model_type="tfidf",
                dataset_manager=dm,
                model_registry=reg,
                artifacts_dir=arts,
            )

    def test_skips_missing_labels(self, tmp_dirs):
        """Rows with missing labels are dropped and training still succeeds."""
        dm, reg, arts = tmp_dirs
        rows = []
        for s in ["positive", "negative", "neutral"]:
            for i in range(12):
                rows.append({"text_feedback": f"text {i} {s}", "sentiment_class": s})
        # Add 5 rows with missing label
        for i in range(5):
            rows.append({"text_feedback": f"unlabelled {i}", "sentiment_class": None})
        df = pd.DataFrame(rows)
        csv_bytes = df.to_csv(index=False).encode()
        csv_path = Path(dm.datasets_dir) / "with_missing.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_bytes(csv_bytes)
        meta = dm.upload_dataset(csv_path, name="with_missing")

        result = run_training(
            dataset_id=meta.id,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=42,
        )
        # n_train + n_val + n_test should equal 36 (5 dropped)
        m = result["metrics"]
        assert m["n_train"] + m["n_val"] + m["n_test"] == 36

    def test_oversample_config_succeeds(self, uploaded_dataset, tmp_dirs):
        dm, reg, arts = tmp_dirs
        cfg = TrainingConfig(class_balancing="oversample")
        result = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            config=cfg,
            seed=42,
        )
        assert result["config"]["class_balancing"] == "oversample"

    def test_version_auto_increments_in_registry(self, uploaded_dataset, tmp_dirs):
        """Training twice on the same dataset+task+type should produce v1 then v2."""
        dm, reg, arts = tmp_dirs
        r1 = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=1,
        )
        r2 = run_training(
            dataset_id=uploaded_dataset,
            task="sentiment",
            model_type="tfidf",
            dataset_manager=dm,
            model_registry=reg,
            artifacts_dir=arts,
            seed=2,
        )
        assert r2["model_version"] == r1["model_version"] + 1
