"""Shared test fixtures for backend tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure backend/src is importable
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TINY_DATASET = FIXTURES_DIR / "tiny_dataset.csv"


@pytest.fixture(scope="session")
def tiny_dataset_path() -> Path:
    return TINY_DATASET


@pytest.fixture(scope="session")
def tiny_df():
    from src.ingest.loader import load_dataset
    return load_dataset(TINY_DATASET)


@pytest.fixture(scope="session")
def preprocessed_df(tiny_df):
    from src.preprocessing.pipeline import run_preprocessing
    return run_preprocessing(tiny_df)


@pytest.fixture(scope="session")
def factor_structure_path() -> Path:
    return BACKEND_DIR / "configs" / "factor_structure.yaml"


@pytest.fixture(scope="session")
def experiment_config_path() -> Path:
    return BACKEND_DIR / "configs" / "experiment.yaml"
