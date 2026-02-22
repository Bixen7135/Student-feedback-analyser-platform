"""Global project configuration — loaded once at startup."""
from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    data_path: Path
    runs_dir: Path
    factor_structure_path: Path
    experiment_config_path: Path
    random_seed: int
    config_hash: str
    raw_config: dict = field(hash=False, compare=False)


def compute_config_hash(config_path: Path) -> str:
    """Compute SHA256 of a config file for reproducibility tracking."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]


def load_config(
    experiment_config_path: Path,
    factor_structure_path: Path | None = None,
    data_path: Path | None = None,
    runs_dir: Path | None = None,
) -> ProjectConfig:
    """Load experiment configuration from YAML."""
    raw = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
    config_hash = compute_config_hash(experiment_config_path)

    resolved_data = data_path or Path(raw["data"]["input_path"])
    resolved_runs = runs_dir or (experiment_config_path.parent.parent / "runs")
    resolved_factors = factor_structure_path or (experiment_config_path.parent / "factor_structure.yaml")

    return ProjectConfig(
        data_path=resolved_data,
        runs_dir=resolved_runs,
        factor_structure_path=resolved_factors,
        experiment_config_path=experiment_config_path,
        random_seed=raw["reproducibility"]["random_seed"],
        config_hash=config_hash,
        raw_config=raw,
    )


def get_system_info() -> dict[str, Any]:
    """Collect system and library version info for reproducibility."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import pandas
        info["pandas_version"] = pandas.__version__
    except ImportError:
        pass
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    return info
