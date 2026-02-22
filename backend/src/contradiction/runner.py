"""Contradiction monitoring pipeline runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson  # type: ignore
import pandas as pd

from src.contradiction.detector import detect_contradictions, ContradictionResult
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_contradiction_monitoring(
    df: pd.DataFrame,
    factor_scores: pd.DataFrame,
    run_dir: Path,
    sentiment_col: str = "sentiment_class",
    high_percentile: float = 90.0,
    low_percentile: float = 10.0,
) -> ContradictionResult:
    """
    Run contradiction monitoring and save results to run_dir/contradiction/.
    Monitoring only — predictions are never altered.
    """
    log.info("contradiction_start", n_rows=len(df))

    result = detect_contradictions(
        df=df,
        factor_scores=factor_scores,
        sentiment_col=sentiment_col,
        high_percentile=high_percentile,
        low_percentile=low_percentile,
    )

    # Save results
    out_dir = run_dir / "contradiction"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "overall_rate": result.overall_rate,
        "n_total": result.n_total,
        "n_contradictions": result.n_contradictions,
        "by_type": result.by_contradiction_type,
        "stratified_by_language": result.stratified_by_language,
        "stratified_by_detail_level": result.stratified_by_length,
        "thresholds": {
            "high_percentile": high_percentile,
            "low_percentile": low_percentile,
        },
        "disclaimer": result.note,
    }
    results_path = out_dir / "results.json"
    results_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    # Save flags CSV
    flags_path = out_dir / "flags.csv"
    result.flags.to_csv(flags_path)

    log.info(
        "contradiction_done",
        n_contradictions=result.n_contradictions,
        overall_rate=round(result.overall_rate, 4),
    )
    return result
