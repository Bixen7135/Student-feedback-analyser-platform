"""FastAPI dependency injection."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from src.utils.run_manager import RunManager

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def get_run_manager() -> RunManager:
    runs_dir = Path(os.environ.get("SFAP_RUNS_DIR", _BACKEND_DIR / "runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    return RunManager(runs_dir)


@lru_cache(maxsize=1)
def _get_db():
    from src.storage.database import Database
    db_path = Path(os.environ.get("SFAP_DB_PATH", _BACKEND_DIR / "data" / "platform.db"))
    return Database(db_path)


@lru_cache(maxsize=1)
def get_dataset_manager():
    from src.storage.dataset_manager import DatasetManager
    datasets_dir = Path(os.environ.get("SFAP_DATASETS_DIR", _BACKEND_DIR / "datasets"))
    return DatasetManager(_get_db(), datasets_dir)


@lru_cache(maxsize=1)
def get_model_registry():
    from src.storage.model_registry import ModelRegistry
    models_dir = Path(os.environ.get("SFAP_MODELS_DIR", _BACKEND_DIR / "models"))
    return ModelRegistry(_get_db(), models_dir)


def get_db():
    return _get_db()
