"""Statistical analytics helpers for descriptive summaries and diagnostics."""

from src.analytics.correlations import (
    cramers_v,
    pearson_correlation,
    point_biserial_correlation,
    spearman_correlation,
)
from src.analytics.descriptive import (
    bootstrap_interval,
    categorical_frequency,
    descriptive_summary,
    numeric_summary,
    text_length_stats,
    t_interval_mean,
    wilson_interval,
)
from src.analytics.diagnostics import (
    classification_diagnostics,
    regression_diagnostics,
)

__all__ = [
    "bootstrap_interval",
    "categorical_frequency",
    "classification_diagnostics",
    "cramers_v",
    "descriptive_summary",
    "numeric_summary",
    "pearson_correlation",
    "point_biserial_correlation",
    "regression_diagnostics",
    "spearman_correlation",
    "t_interval_mean",
    "text_length_stats",
    "wilson_interval",
]
