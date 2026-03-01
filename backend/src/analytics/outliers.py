"""Outlier detection helpers for 2D embedding points."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pandas as pd
from sklearn.ensemble import IsolationForest  # type: ignore
from sklearn.neighbors import LocalOutlierFactor  # type: ignore


def detect_outliers(
    *,
    points_df: pd.DataFrame,
    analytics_dir: Path,
    method: str = "isolation_forest",
    contamination: float = 0.1,
    n_neighbors: int = 5,
) -> dict[str, Any]:
    """Detect outliers from 2D points and persist scored rows."""
    method = method.lower().strip() or "isolation_forest"
    if points_df.empty:
        empty_path = analytics_dir / f"outliers_{method}.csv"
        points_df.assign(
            is_outlier=pd.Series(dtype=bool),
            outlier_score=pd.Series(dtype=float),
        ).to_csv(
            empty_path,
            index=False,
            encoding="utf-8-sig",
        )
        metadata = {
            "method": method,
            "contamination": float(contamination),
            "n_neighbors": int(n_neighbors),
            "n_rows": 0,
        }
        (analytics_dir / f"outliers_{method}.json").write_bytes(
            orjson.dumps(metadata, option=orjson.OPT_INDENT_2)
        )
        return {
            "artifact_path": str(empty_path),
            "metadata": metadata,
            "outlier_count": 0,
            "rows": [],
        }

    coords = points_df[["x", "y"]].astype(float)

    if method == "lof":
        if len(points_df) < 3:
            method = "isolation_forest"
        safe_neighbors = max(2, min(int(n_neighbors), max(2, len(points_df) - 1)))
    if method == "lof":
        model = LocalOutlierFactor(
            n_neighbors=safe_neighbors,
            contamination=contamination,
        )
        pred = model.fit_predict(coords)
        scores = -model.negative_outlier_factor_
    else:
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
        )
        pred = model.fit_predict(coords)
        scores = -model.score_samples(coords)
        method = "isolation_forest"

    scored = points_df.copy()
    scored["is_outlier"] = [bool(value == -1) for value in pred]
    scored["outlier_score"] = [round(float(value), 6) for value in scores]

    output_path = analytics_dir / f"outliers_{method}.csv"
    meta_path = analytics_dir / f"outliers_{method}.json"
    scored.to_csv(output_path, index=False, encoding="utf-8-sig")
    metadata = {
        "method": method,
        "contamination": float(contamination),
        "n_neighbors": int(n_neighbors) if method == "lof" else None,
        "n_rows": int(len(scored)),
    }
    meta_path.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

    outlier_rows = scored[scored["is_outlier"]]
    return {
        "artifact_path": str(output_path),
        "metadata": metadata,
        "outlier_count": int(outlier_rows.shape[0]),
        "rows": scored.fillna("").to_dict(orient="records"),
    }
