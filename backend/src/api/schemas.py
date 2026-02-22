"""Pydantic schemas for API request/response models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class StageStatusResponse(BaseModel):
    name: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


class RunSummaryResponse(BaseModel):
    run_id: str
    created_at: str
    config_hash: str
    data_snapshot_id: str
    random_seed: int
    stages: dict[str, StageStatusResponse]


class RunDetailResponse(RunSummaryResponse):
    git_commit: str | None = None
    system_info: dict[str, Any] | None = None


class CreateRunRequest(BaseModel):
    data_path: str | None = None
    config_path: str | None = None
    factor_structure_path: str | None = None
    seed: int = 42


class StartStageRequest(BaseModel):
    stage: str


class ArtifactResponse(BaseModel):
    name: str
    path: str
    type: str
    stage: str
    size_bytes: int | None = None
    created_at: str | None = None


class ArtifactManifestResponse(BaseModel):
    run_id: str
    artifacts: list[ArtifactResponse]


class PsychometricsMetricsResponse(BaseModel):
    method: str
    n_obs: int
    factor_names: list[str]
    fit_statistics: dict[str, Any]
    reliability: dict[str, Any]
    loadings: dict[str, dict[str, float]] | None = None


class ClassificationMetricsResponse(BaseModel):
    task: str
    model_type: str
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    classes: list[str]
    n_samples: int


class FusionMetricsResponse(BaseModel):
    factor_names: list[str]
    survey_only: dict[str, Any]
    text_only: dict[str, Any]
    late_fusion: dict[str, Any]
    delta_mae: dict[str, float]
    delta_r2: dict[str, float]
    ablations: dict[str, Any] | None = None


class ContradictionMetricsResponse(BaseModel):
    overall_rate: float
    n_total: int
    n_contradictions: int
    by_type: dict[str, float]
    stratified_by_language: dict[str, float]
    stratified_by_detail_level: dict[str, float]
    disclaimer: str
