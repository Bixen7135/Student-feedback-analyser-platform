"""Clustering helpers for 2D embedding points."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans  # type: ignore


def cluster_points(
    *,
    points_df: pd.DataFrame,
    analytics_dir: Path,
    method: str = "kmeans",
    k: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
) -> dict[str, Any]:
    """Cluster 2D points and persist assignments."""
    method = method.lower().strip() or "kmeans"
    if points_df.empty:
        empty_path = analytics_dir / f"clustering_{method}.csv"
        points_df.assign(cluster=pd.Series(dtype=int)).to_csv(
            empty_path,
            index=False,
            encoding="utf-8-sig",
        )
        metadata = {"method": method, "k": int(k), "eps": float(eps), "min_samples": int(min_samples), "n_rows": 0}
        (analytics_dir / f"clustering_{method}.json").write_bytes(
            orjson.dumps(metadata, option=orjson.OPT_INDENT_2)
        )
        return {
            "artifact_path": str(empty_path),
            "metadata": metadata,
            "clusters": [],
            "cluster_counts": {},
        }

    coords = points_df[["x", "y"]].astype(float)
    if method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(coords)
    else:
        safe_k = max(1, min(int(k), len(points_df)))
        model = KMeans(n_clusters=safe_k, random_state=42, n_init=10)
        labels = model.fit_predict(coords)
        method = "kmeans"

    clustered = points_df.copy()
    clustered["cluster"] = [int(label) for label in labels]

    output_path = analytics_dir / f"clustering_{method}.csv"
    meta_path = analytics_dir / f"clustering_{method}.json"
    clustered.to_csv(output_path, index=False, encoding="utf-8-sig")
    metadata = {
        "method": method,
        "k": int(k) if method == "kmeans" else None,
        "eps": float(eps) if method == "dbscan" else None,
        "min_samples": int(min_samples) if method == "dbscan" else None,
        "n_rows": int(len(clustered)),
    }
    meta_path.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

    counts = clustered["cluster"].value_counts().sort_index()
    return {
        "artifact_path": str(output_path),
        "metadata": metadata,
        "clusters": clustered.fillna("").to_dict(orient="records"),
        "cluster_counts": {str(label): int(count) for label, count in counts.items()},
    }
