"""Pydantic models for storage entities."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    """Schema info for a single column."""
    name: str
    dtype: str
    n_unique: int = 0
    n_null: int = 0
    sample_values: list[str] = Field(default_factory=list)


class DatasetMeta(BaseModel):
    """Dataset metadata record."""
    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    created_at: str
    current_version: int = 1
    schema_info: list[ColumnSchema] = Field(default_factory=list)
    row_count: int = 0
    file_size_bytes: int = 0
    sha256: str = ""
    status: str = "active"
    default_branch_id: str | None = None


class DatasetBranch(BaseModel):
    """A named branch within a dataset's version history."""
    id: str
    dataset_id: str
    name: str
    description: str = ""
    base_version_id: str | None = None
    head_version_id: str | None = None
    author: str = ""
    created_at: str
    is_default: bool = False
    is_deleted: bool = False


class DatasetVersion(BaseModel):
    """A specific version of a dataset."""
    id: str
    dataset_id: str
    version: int
    created_at: str
    author: str = ""
    reason: str = "initial upload"
    sha256: str = ""
    row_count: int = 0
    file_size_bytes: int = 0
    storage_path: str = ""
    branch_id: str | None = None
    column_roles: dict[str, str] = Field(default_factory=dict)
    is_fork: bool = False  # True if this version was created as a branch fork


class ModelMeta(BaseModel):
    """Trained model metadata record."""
    id: str
    name: str
    task: str
    model_type: str
    version: int = 1
    dataset_id: str | None = None
    dataset_version: int | None = None
    config: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    created_at: str
    status: str = "active"
    storage_path: str = ""
    run_id: str | None = None
    base_model_id: str | None = None  # set when fine-tuned from an existing model


class AnalysisRunMeta(BaseModel):
    """Analysis run metadata record."""
    id: str
    name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    comments: str = ""
    dataset_id: str | None = None
    dataset_version: int | None = None
    model_ids: list[str] = Field(default_factory=list)
    created_at: str
    status: str = "pending"
    run_id: str = ""
    result_summary: dict = Field(default_factory=dict)


class SavedFilter(BaseModel):
    """Saved filter preset."""
    id: str
    name: str
    entity_type: str
    filter_config: dict = Field(default_factory=dict)
    created_at: str
