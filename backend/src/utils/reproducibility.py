"""Reproducibility utilities — seed setting and system info collection."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any


def set_all_seeds(seed: int) -> None:
    """Set seeds for all relevant random number generators."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_string(s: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def collect_library_versions() -> dict[str, Any]:
    """Collect versions of key libraries for reproducibility logging."""
    versions: dict[str, Any] = {}
    for lib in ["numpy", "pandas", "sklearn", "scipy", "semopy", "factor_analyzer", "fastapi", "pydantic"]:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not_installed"
    return versions
