"""Unit tests for psychometrics (CFA)."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np


def test_factor_structure_loads_from_yaml(factor_structure_path):
    from src.psychometrics.factor_config import load_factor_structure
    fs = load_factor_structure(factor_structure_path)
    assert len(fs.factor_names) == 3
    assert all(len(items) == 3 for items in fs.factors.values())


def test_semopy_syntax_generation(factor_structure_path):
    from src.psychometrics.factor_config import load_factor_structure, to_semopy_syntax
    fs = load_factor_structure(factor_structure_path)
    syntax = to_semopy_syntax(fs)
    assert "=~" in syntax
    assert "item_1" in syntax
    assert "item_9" in syntax


def test_cfa_output_shape(preprocessed_df, factor_structure_path):
    from src.psychometrics.ordinal_cfa import fit_cfa
    from src.psychometrics.factor_config import load_factor_structure
    fs = load_factor_structure(factor_structure_path)
    result = fit_cfa(preprocessed_df, fs)
    assert result.factor_scores.shape == (len(preprocessed_df.dropna(subset=fs.all_items)), 3)
    assert list(result.factor_scores.columns) == fs.factor_names


def test_cfa_factor_score_uncertainty(preprocessed_df, factor_structure_path):
    from src.psychometrics.ordinal_cfa import fit_cfa
    from src.psychometrics.factor_config import load_factor_structure
    fs = load_factor_structure(factor_structure_path)
    result = fit_cfa(preprocessed_df, fs)
    assert result.factor_score_se.shape == result.factor_scores.shape
    assert not result.factor_score_se.isnull().all().all()


def test_cfa_fit_statistics_present(preprocessed_df, factor_structure_path):
    from src.psychometrics.ordinal_cfa import fit_cfa
    from src.psychometrics.factor_config import load_factor_structure
    fs = load_factor_structure(factor_structure_path)
    result = fit_cfa(preprocessed_df, fs)
    assert isinstance(result.fit_statistics, dict)
    assert len(result.fit_statistics) > 0


def test_cfa_loadings_shape(preprocessed_df, factor_structure_path):
    from src.psychometrics.ordinal_cfa import fit_cfa
    from src.psychometrics.factor_config import load_factor_structure
    fs = load_factor_structure(factor_structure_path)
    result = fit_cfa(preprocessed_df, fs)
    assert result.loadings.shape == (9, 3)


def test_cronbach_alpha_range(preprocessed_df, factor_structure_path):
    from src.psychometrics.reliability import cronbach_alpha
    alpha = cronbach_alpha(preprocessed_df, [f"item_{i}" for i in range(1, 10)])
    if not np.isnan(alpha):
        assert -1.0 <= alpha <= 1.0


def test_mcdonald_omega_range(preprocessed_df, factor_structure_path):
    from src.psychometrics.ordinal_cfa import fit_cfa
    from src.psychometrics.factor_config import load_factor_structure
    from src.psychometrics.reliability import mcdonald_omega
    fs = load_factor_structure(factor_structure_path)
    result = fit_cfa(preprocessed_df, fs)
    for factor_name in fs.factor_names:
        if factor_name in result.loadings.columns:
            omega = mcdonald_omega(result.loadings[factor_name])
            if not np.isnan(omega):
                assert 0.0 <= omega <= 1.0
