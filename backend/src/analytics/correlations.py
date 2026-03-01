"""Association metrics for mixed data types."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def pearson_correlation(
    left: pd.Series | list[float] | np.ndarray,
    right: pd.Series | list[float] | np.ndarray,
) -> dict[str, float | int | str]:
    """Return Pearson's r for two numeric variables."""
    frame = _paired_numeric_frame(left, right)
    n = int(len(frame))
    if n < 2:
        return {"metric": "pearson", "value": 0.0, "n": n}
    value = float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method="pearson"))
    if np.isnan(value):
        value = 0.0
    return {"metric": "pearson", "value": round(value, 4), "n": n}


def spearman_correlation(
    left: pd.Series | list[float] | np.ndarray,
    right: pd.Series | list[float] | np.ndarray,
) -> dict[str, float | int | str]:
    """Return Spearman's rho for two numeric or ordinal variables."""
    frame = _paired_numeric_frame(left, right)
    n = int(len(frame))
    if n < 2:
        return {"metric": "spearman", "value": 0.0, "n": n}
    value = float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method="spearman"))
    if np.isnan(value):
        value = 0.0
    return {"metric": "spearman", "value": round(value, 4), "n": n}


def cramers_v(
    left: pd.Series | list[Any] | np.ndarray,
    right: pd.Series | list[Any] | np.ndarray,
) -> dict[str, float | int | str]:
    """Return bias-corrected Cramer's V for two categorical variables."""
    frame = _paired_text_frame(left, right)
    n = int(len(frame))
    if n == 0:
        return {"metric": "cramers_v", "value": 0.0, "n": 0}

    table = pd.crosstab(frame.iloc[:, 0], frame.iloc[:, 1])
    if table.empty:
        return {"metric": "cramers_v", "value": 0.0, "n": n}

    observed = table.to_numpy(dtype=float)
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)

    phi2 = chi2 / n if n else 0.0
    r, k = observed.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    value = sqrt_safe(phi2corr / denom) if denom > 0 else 0.0
    return {"metric": "cramers_v", "value": round(value, 4), "n": n}


def point_biserial_correlation(
    binary: pd.Series | list[Any] | np.ndarray,
    continuous: pd.Series | list[float] | np.ndarray,
) -> dict[str, float | int | str]:
    """Return the point-biserial correlation for a binary and continuous variable."""
    binary_series = pd.Series(binary)
    continuous_series = pd.to_numeric(pd.Series(continuous), errors="coerce")
    frame = pd.DataFrame({"binary": binary_series, "continuous": continuous_series}).dropna()
    if frame.empty:
        return {"metric": "point_biserial", "value": 0.0, "n": 0}

    labels = list(dict.fromkeys(frame["binary"].astype(str).tolist()))
    if len(labels) != 2:
        return {"metric": "point_biserial", "value": 0.0, "n": int(len(frame))}

    mapping = {labels[0]: 0.0, labels[1]: 1.0}
    encoded = frame["binary"].astype(str).map(mapping)
    value = float(encoded.corr(frame["continuous"], method="pearson"))
    if np.isnan(value):
        value = 0.0
    return {"metric": "point_biserial", "value": round(value, 4), "n": int(len(frame))}


def mixed_pairwise_correlations(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute pairwise associations using the best-fit metric per column pair."""
    selected = [col for col in (columns or list(df.columns)) if col in df.columns]
    results: list[dict[str, Any]] = []
    for left, right in combinations(selected, 2):
        result = _mixed_correlation(df[left], df[right])
        if result is None:
            continue
        results.append(
            {
                "left": left,
                "right": right,
                **result,
            }
        )
    return results


def _mixed_correlation(left: pd.Series, right: pd.Series) -> dict[str, Any] | None:
    left_info = _series_kind(left)
    right_info = _series_kind(right)

    if left_info["kind"] == "numeric" and right_info["kind"] == "numeric":
        return pearson_correlation(left, right)

    if left_info["kind"] == "binary" and right_info["kind"] == "numeric":
        return point_biserial_correlation(left, right)

    if left_info["kind"] == "numeric" and right_info["kind"] == "binary":
        return point_biserial_correlation(right, left)

    if left_info["kind"] in {"binary", "categorical"} and right_info["kind"] in {"binary", "categorical"}:
        return cramers_v(left, right)

    if left_info["kind"] == "numeric" and right_info["kind"] == "ordinal":
        return spearman_correlation(left, right)

    if left_info["kind"] == "ordinal" and right_info["kind"] == "numeric":
        return spearman_correlation(left, right)

    return None


def _series_kind(series: pd.Series) -> dict[str, Any]:
    text = series.fillna("").astype(str)
    non_empty = text[text.str.strip() != ""]
    if non_empty.empty:
        return {"kind": "empty", "n_unique": 0}

    numeric = pd.to_numeric(non_empty, errors="coerce")
    if numeric.notna().all():
        n_unique = int(numeric.nunique())
        if n_unique == 2:
            return {"kind": "binary", "n_unique": n_unique}
        if 2 < n_unique <= 7:
            return {"kind": "ordinal", "n_unique": n_unique}
        return {"kind": "numeric", "n_unique": n_unique}

    n_unique = int(non_empty.nunique())
    if n_unique == 2:
        return {"kind": "binary", "n_unique": n_unique}
    return {"kind": "categorical", "n_unique": n_unique}


def _paired_numeric_frame(
    left: pd.Series | list[float] | np.ndarray,
    right: pd.Series | list[float] | np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "left": pd.to_numeric(pd.Series(left), errors="coerce"),
            "right": pd.to_numeric(pd.Series(right), errors="coerce"),
        }
    ).dropna()
    return frame


def _paired_text_frame(
    left: pd.Series | list[Any] | np.ndarray,
    right: pd.Series | list[Any] | np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "left": pd.Series(left).fillna("").astype(str),
            "right": pd.Series(right).fillna("").astype(str),
        }
    )
    mask = (frame["left"].str.strip() != "") & (frame["right"].str.strip() != "")
    return frame[mask]


def sqrt_safe(value: float) -> float:
    return float(np.sqrt(max(0.0, value)))
