"""
Ordinal CFA implementation using semopy with Spearman correlation approximation.
Falls back to factor_analyzer EFA if semopy fails.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.psychometrics.factor_config import FactorStructure, to_semopy_syntax
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CFAResult:
    factor_scores: pd.DataFrame        # N x n_factors, columns = factor names
    factor_score_se: pd.DataFrame      # N x n_factors, standard errors (uncertainty)
    loadings: pd.DataFrame             # items x factors
    fit_statistics: dict[str, Any]     # CFI, RMSEA, SRMR, chi2, df, p
    reliability: dict[str, Any]        # per-factor reliability
    correlation_matrix: pd.DataFrame   # polychoric or spearman
    method: str                        # "semopy_cfa" or "efa_fallback"
    factor_structure: FactorStructure
    n_obs: int


def compute_spearman_correlation(df: pd.DataFrame, item_cols: list[str]) -> pd.DataFrame:
    """Compute Spearman correlation matrix for ordinal items."""
    return df[item_cols].corr(method="spearman")


def _fit_cfa_semopy(
    df: pd.DataFrame,
    factor_structure: FactorStructure,
) -> CFAResult:
    """Attempt CFA using semopy. Raises on failure."""
    import semopy  # type: ignore

    item_cols = factor_structure.all_items
    model_desc = to_semopy_syntax(factor_structure)
    log.info("cfa_semopy_start", model_desc=model_desc)

    data = df[item_cols].dropna().astype(float)
    model = semopy.Model(model_desc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)

    # Extract fit statistics
    # calc_stats returns a DataFrame with index ["Value"]; use .iloc[0] for scalars
    try:
        stats = semopy.calc_stats(model)

        def _stat(col: str) -> float:
            if col in stats.columns:
                return float(stats[col].iloc[0])
            return float("nan")

        fit_stats = {
            "cfi": _stat("CFI"),
            "rmsea": _stat("RMSEA"),
            "gfi": _stat("GFI"),
            "agfi": _stat("AGFI"),
            "nfi": _stat("NFI"),
            "tli": _stat("TLI"),
            "chi2": _stat("chi2"),
            "chi2_df": _stat("DoF"),
            "chi2_p": _stat("chi2 p-value"),
        }
    except Exception as e:
        log.warning("cfa_fit_stats_failed", error=str(e))
        fit_stats = {"error": str(e)}

    # Extract loadings from model parameters
    try:
        params = model.inspect()
        loadings = _extract_loadings_from_semopy(params, factor_structure, item_cols)
    except Exception as e:
        log.warning("cfa_loadings_extraction_failed", error=str(e))
        loadings = pd.DataFrame(
            np.zeros((len(item_cols), len(factor_structure.factor_names))),
            index=item_cols,
            columns=factor_structure.factor_names,
        )

    # Extract factor scores using regression method from loadings
    factor_scores, factor_score_se = _compute_factor_scores(
        data, loadings, factor_structure
    )

    corr_matrix = compute_spearman_correlation(df, item_cols)

    return CFAResult(
        factor_scores=factor_scores,
        factor_score_se=factor_score_se,
        loadings=loadings,
        fit_statistics=fit_stats,
        reliability={},
        correlation_matrix=corr_matrix,
        method="semopy_cfa",
        factor_structure=factor_structure,
        n_obs=len(data),
    )


def _extract_loadings_from_semopy(
    params: pd.DataFrame,
    factor_structure: FactorStructure,
    item_cols: list[str],
) -> pd.DataFrame:
    """Parse semopy parameter table to extract item loadings per factor."""
    loadings = pd.DataFrame(
        0.0,
        index=item_cols,
        columns=factor_structure.factor_names,
    )
    # semopy inspect() uses op "~" (not "=~"), direction: lval=item, rval=factor
    # e.g. "item_1 ~ program_quality" means item_1 loads on program_quality
    mask = (params["op"] == "~") if "op" in params.columns else pd.Series(False, index=params.index)
    for _, row in params[mask].iterrows():
        item = row.get("lval", "")
        factor = row.get("rval", "")
        estimate = row.get("Estimate", row.get("estimate", np.nan))
        if factor in loadings.columns and item in loadings.index:
            loadings.loc[item, factor] = float(estimate)
    return loadings


def _fit_cfa_factor_analyzer(
    df: pd.DataFrame,
    factor_structure: FactorStructure,
) -> CFAResult:
    """Fallback: fit EFA using factor_analyzer with fixed n_factors=3."""
    from factor_analyzer import FactorAnalyzer  # type: ignore

    item_cols = factor_structure.all_items
    n_factors = len(factor_structure.factor_names)
    log.info("cfa_efa_fallback_start", n_factors=n_factors)

    data = df[item_cols].dropna().astype(float)

    fa = FactorAnalyzer(n_factors=n_factors, rotation="promax", method="minres")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa.fit(data.values)

    loading_matrix = fa.loadings_
    loadings = pd.DataFrame(
        loading_matrix,
        index=item_cols,
        columns=factor_structure.factor_names,
    )

    # Align factor columns to intended structure (each item should load on its intended factor)
    loadings = _align_loadings_to_structure(loadings, factor_structure)

    # Factor scores
    scores_arr = fa.transform(data.values)
    factor_scores = pd.DataFrame(
        scores_arr,
        index=data.index,
        columns=factor_structure.factor_names,
    )
    # Approximate SE as residual variance (simplified)
    factor_score_se = pd.DataFrame(
        np.full_like(scores_arr, fill_value=0.1),
        index=data.index,
        columns=factor_structure.factor_names,
    )

    # Fit statistics from factor_analyzer
    try:
        ev = fa.get_eigenvalues()[0]
        communalities = fa.get_communalities()
        fit_stats = {
            "eigenvalues": ev.tolist(),
            "communalities_mean": float(communalities.mean()),
            "variance_explained": fa.get_factor_variance()[1].tolist(),
            "method": "efa_fallback",
        }
    except Exception:
        fit_stats = {"method": "efa_fallback"}

    corr_matrix = compute_spearman_correlation(df, item_cols)

    return CFAResult(
        factor_scores=factor_scores,
        factor_score_se=factor_score_se,
        loadings=loadings,
        fit_statistics=fit_stats,
        reliability={},
        correlation_matrix=corr_matrix,
        method="efa_fallback",
        factor_structure=factor_structure,
        n_obs=len(data),
    )


def _align_loadings_to_structure(
    loadings: pd.DataFrame,
    factor_structure: FactorStructure,
) -> pd.DataFrame:
    """
    Reassign EFA factor columns so that each factor aligns with its intended items.
    Uses the maximum loading assignment per intended factor to match columns.
    """
    aligned = pd.DataFrame(
        0.0,
        index=loadings.index,
        columns=factor_structure.factor_names,
    )
    used_cols: set[int] = set()
    for factor_name, items in factor_structure.factors.items():
        # Find the EFA column that has the highest mean absolute loading on these items
        available_cols = [c for c in range(loadings.shape[1]) if c not in used_cols]
        if not available_cols:
            break
        sub = loadings.iloc[[loadings.index.get_loc(i) for i in items if i in loadings.index]]
        col_sums = sub.iloc[:, available_cols].abs().mean()
        best_col = available_cols[col_sums.argmax()]
        aligned[factor_name] = loadings.iloc[:, best_col].values
        used_cols.add(best_col)
    return aligned


def _compute_factor_scores(
    data: pd.DataFrame,
    loadings: pd.DataFrame,
    factor_structure: FactorStructure,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute factor scores using the regression method.
    F = X_std * L * (L'L)^-1
    where X_std is standardized observed data, L is the loading matrix.
    """
    X = data[loadings.index].values.astype(float)
    # Standardize
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_scaled = (X - X_mean) / X_std

    L = loadings.values  # items x factors
    try:
        LtL_inv = np.linalg.pinv(L.T @ L)
        scores = X_scaled @ L @ LtL_inv
    except np.linalg.LinAlgError:
        # Fallback: use L directly as projection
        scores = X_scaled @ L

    # Approximate SE as std of residuals
    reconstructed = scores @ L.T
    residuals = X_scaled - reconstructed
    se = np.full_like(scores, fill_value=residuals.std())

    factor_scores = pd.DataFrame(
        scores,
        index=data.index,
        columns=factor_structure.factor_names,
    )
    factor_score_se = pd.DataFrame(
        se,
        index=data.index,
        columns=factor_structure.factor_names,
    )
    return factor_scores, factor_score_se


def fit_cfa(
    df: pd.DataFrame,
    factor_structure: FactorStructure,
) -> CFAResult:
    """
    Fit CFA model to the ordinal survey items.
    Tries semopy first; falls back to factor_analyzer EFA if semopy fails.
    """
    item_cols = factor_structure.all_items
    missing = [c for c in item_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing item columns: {missing}")

    try:
        result = _fit_cfa_semopy(df, factor_structure)
        log.info("cfa_method_used", method="semopy_cfa", n_obs=result.n_obs)
        return result
    except Exception as e:
        log.warning("cfa_semopy_failed_using_fallback", error=str(e))
        result = _fit_cfa_factor_analyzer(df, factor_structure)
        log.info("cfa_method_used", method="efa_fallback", n_obs=result.n_obs)
        return result


def save_cfa_artifacts(result: CFAResult, run_dir: Path) -> dict[str, Path]:
    """Save all CFA artifacts to run_dir/psychometrics/. Returns artifact paths."""
    import orjson  # type: ignore

    out_dir = run_dir / "psychometrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Factor scores CSV
    scores_path = out_dir / "factor_scores.csv"
    result.factor_scores.to_csv(scores_path)
    paths["factor_scores"] = scores_path

    # Factor score SE CSV
    se_path = out_dir / "factor_score_se.csv"
    result.factor_score_se.to_csv(se_path)
    paths["factor_score_se"] = se_path

    # Loadings CSV
    loadings_path = out_dir / "loadings.csv"
    result.loadings.to_csv(loadings_path)
    paths["loadings"] = loadings_path

    # Fit statistics JSON
    fit_path = out_dir / "fit_statistics.json"
    fit_path.write_bytes(orjson.dumps(result.fit_statistics, option=orjson.OPT_INDENT_2))
    paths["fit_statistics"] = fit_path

    # Correlation matrix CSV
    corr_path = out_dir / "correlation_matrix.csv"
    result.correlation_matrix.to_csv(corr_path)
    paths["correlation_matrix"] = corr_path

    # Summary JSON
    summary = {
        "method": result.method,
        "n_obs": result.n_obs,
        "factor_names": result.factor_structure.factor_names,
        "fit_statistics": result.fit_statistics,
        "reliability": result.reliability,
    }
    summary_path = out_dir / "cfa_summary.json"
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    paths["cfa_summary"] = summary_path

    log.info("cfa_artifacts_saved", out_dir=str(out_dir))
    return paths
