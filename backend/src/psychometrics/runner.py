"""Psychometrics pipeline runner."""
from __future__ import annotations

from pathlib import Path

import orjson  # type: ignore
import pandas as pd

from src.psychometrics.factor_config import load_factor_structure
from src.psychometrics.ordinal_cfa import fit_cfa, save_cfa_artifacts, CFAResult
from src.psychometrics.reliability import compute_reliability
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_psychometrics(
    df: pd.DataFrame,
    factor_structure_path: Path,
    run_dir: Path,
) -> CFAResult:
    """
    Full psychometrics pipeline:
    1. Load factor structure config
    2. Fit CFA (semopy or fallback)
    3. Compute reliability metrics
    4. Save all artifacts to run_dir/psychometrics/

    Returns CFAResult with factor scores, loadings, fit statistics, reliability.
    """
    log.info("psychometrics_start", n_rows=len(df))

    # 1. Load config
    factor_structure = load_factor_structure(factor_structure_path)
    log.info("factor_structure_loaded", factors=factor_structure.factor_names)

    # 2. Fit CFA
    result = fit_cfa(df, factor_structure)

    # 3. Reliability
    reliability = compute_reliability(
        df,
        factor_structure.factors,
        result.loadings,
    )
    result.reliability.update({
        "cronbach_alpha": reliability.cronbach_alpha,
        "mcdonald_omega": reliability.mcdonald_omega,
        "alpha_if_item_deleted": reliability.per_item_alpha_if_deleted,
    })

    # 4. Save artifacts
    save_cfa_artifacts(result, run_dir)

    # Save reliability separately
    rel_path = run_dir / "psychometrics" / "reliability.json"
    rel_path.write_bytes(orjson.dumps(result.reliability, option=orjson.OPT_INDENT_2))

    log.info("psychometrics_done", method=result.method, n_obs=result.n_obs)
    return result


def load_factor_scores(run_dir: Path) -> pd.DataFrame:
    """Load factor scores from a completed psychometrics run."""
    path = run_dir / "psychometrics" / "factor_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Factor scores not found: {path}")
    return pd.read_csv(path, index_col=0)
