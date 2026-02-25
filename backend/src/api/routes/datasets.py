"""Dataset management API routes."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field

from src.api.dependencies import get_dataset_manager
from src.ingest.loader import COLUMN_MAP
from src.storage.dataset_manager import DatasetManager, DatasetValidationError

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class DatasetUploadResponse(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    author: str
    created_at: str
    current_version: int
    row_count: int
    file_size_bytes: int
    sha256: str
    schema_info: list[dict[str, Any]]
    default_branch_id: str | None = None


class DatasetListResponse(BaseModel):
    datasets: list[dict[str, Any]]
    total: int
    page: int
    per_page: int


class DatasetDetailResponse(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    author: str
    created_at: str
    current_version: int
    row_count: int
    file_size_bytes: int
    sha256: str
    status: str
    schema_info: list[dict[str, Any]]
    default_branch_id: str | None = None


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    version: int
    columns: list[str]
    total_rows: int
    offset: int
    limit: int
    rows: list[list[Any]]


class DatasetUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None


class SubsetRequest(BaseModel):
    name: str
    description: str = ""
    author: str = ""
    version: int | None = None
    filter_config: dict[str, Any] = Field(default_factory=dict)


class DatasetVersionResponse(BaseModel):
    id: str
    dataset_id: str
    version: int
    created_at: str
    author: str
    reason: str
    sha256: str
    row_count: int
    file_size_bytes: int
    branch_id: str | None = None
    column_roles: dict[str, str] = Field(default_factory=dict)
    is_fork: bool = False


class DatasetDeleteResponse(BaseModel):
    deleted: bool
    reason: str | None = None
    dependencies: dict[str, int] | None = None


class CellChange(BaseModel):
    row_idx: int
    col: str
    value: str


class BatchCellUpdateRequest(BaseModel):
    changes: list[CellChange]
    reason: str = "cell edits"
    author: str = ""
    branch_id: str | None = None


class AddRowsRequest(BaseModel):
    rows: list[dict[str, str]]
    reason: str = "added rows"
    author: str = ""
    branch_id: str | None = None


class DeleteRowsRequest(BaseModel):
    row_indices: list[int]
    reason: str = "deleted rows"
    author: str = ""
    branch_id: str | None = None


class RenameColumnsRequest(BaseModel):
    renames: dict[str, str]
    reason: str = "rename columns"
    author: str = ""
    branch_id: str | None = None


class CreateEmptyDatasetRequest(BaseModel):
    name: str
    columns: list[str]
    description: str = ""
    tags: list[str] = []
    author: str = ""


# Branch schemas
class BranchCreateRequest(BaseModel):
    name: str
    description: str = ""
    from_version_id: str | None = None
    author: str = ""


class BranchUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class BranchResponse(BaseModel):
    id: str
    dataset_id: str
    name: str
    description: str
    base_version_id: str | None
    head_version_id: str | None = None
    author: str
    created_at: str
    is_default: bool
    is_deleted: bool = False


class VersionUpdateRequest(BaseModel):
    reason: str


# ---------------------------------------------------------------------------
# Endpoints — IMPORTANT: static paths must be registered BEFORE /{dataset_id}/...
# ---------------------------------------------------------------------------

@router.post("/empty", response_model=DatasetUploadResponse)
async def create_empty_dataset(
    req: CreateEmptyDatasetRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Create a new empty dataset with a defined column schema (no CSV upload needed)."""
    try:
        meta = mgr.create_empty_dataset(
            name=req.name,
            columns=req.columns,
            description=req.description,
            tags=req.tags,
            author=req.author,
        )
        return DatasetUploadResponse(
            id=meta.id,
            name=meta.name,
            description=meta.description,
            tags=meta.tags,
            author=meta.author,
            created_at=meta.created_at,
            current_version=meta.current_version,
            row_count=meta.row_count,
            file_size_bytes=meta.file_size_bytes,
            sha256=meta.sha256,
            schema_info=[s.model_dump() for s in meta.schema_info],
            default_branch_id=meta.default_branch_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    tags: str = Form("[]"),
    author: str = Form(""),
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Upload a new CSV dataset with metadata."""
    import orjson

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        tags_list = orjson.loads(tags) if tags else []
    except Exception:
        tags_list = []

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        meta = mgr.upload_dataset(
            file_path=tmp_path,
            name=name,
            description=description,
            tags=tags_list,
            author=author,
        )
        return DatasetUploadResponse(
            id=meta.id,
            name=meta.name,
            description=meta.description,
            tags=meta.tags,
            author=meta.author,
            created_at=meta.created_at,
            current_version=meta.current_version,
            row_count=meta.row_count,
            file_size_bytes=meta.file_size_bytes,
            sha256=meta.sha256,
            schema_info=[s.model_dump() for s in meta.schema_info],
            default_branch_id=meta.default_branch_id,
        )
    except DatasetValidationError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    search: str | None = Query(None),
    tags: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """List datasets with search, filter, and pagination."""
    import orjson

    tags_list = orjson.loads(tags) if tags else None

    datasets, total = mgr.list_datasets(
        search=search,
        tags=tags_list,
        sort=sort,
        order=order,
        page=page,
        per_page=per_page,
    )
    return DatasetListResponse(
        datasets=[d.model_dump() for d in datasets],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(
    dataset_id: str,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Get full metadata for a dataset."""
    ds = mgr.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return DatasetDetailResponse(
        id=ds.id,
        name=ds.name,
        description=ds.description,
        tags=ds.tags,
        author=ds.author,
        created_at=ds.created_at,
        current_version=ds.current_version,
        row_count=ds.row_count,
        file_size_bytes=ds.file_size_bytes,
        sha256=ds.sha256,
        status=ds.status,
        schema_info=[s.model_dump() for s in ds.schema_info],
        default_branch_id=ds.default_branch_id,
    )


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def get_dataset_preview(
    dataset_id: str,
    version: int | None = Query(None),
    branch_id: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Get paginated rows from a dataset."""
    try:
        preview = mgr.get_dataset_preview(
            dataset_id, version=version, branch_id=branch_id,
            offset=offset, limit=limit,
        )
        return DatasetPreviewResponse(**preview)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/schema")
async def get_dataset_schema(
    dataset_id: str,
    version_id: str | None = Query(None, description="Version UUID; omit for current version"),
    version: int | None = Query(None, description="Version number; optional alternative to version_id"),
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Get column schema information for a dataset."""
    try:
        resolved_version = version
        if version_id and resolved_version is None:
            row = mgr.db.fetchone(
                """SELECT version FROM dataset_versions
                   WHERE id = ? AND dataset_id = ? AND is_deleted = 0""",
                (version_id, dataset_id),
            )
            if row is None:
                raise ValueError(f"Dataset version not found: {version_id}")
            resolved_version = int(row["version"])

        if version_id is None and resolved_version is None:
            schema = mgr.get_dataset_schema(dataset_id)
        else:
            df = mgr.get_dataframe(dataset_id, version=resolved_version)
            schema = []
            for col in df.columns:
                sample_values = (
                    df[col].dropna().head(5).astype(str).tolist()
                    if col in df.columns
                    else []
                )
                schema.append({
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "n_unique": int(df[col].nunique()),
                    "n_null": int(df[col].isna().sum()),
                    "sample_values": sample_values,
                })

        cols = []
        for s in schema:
            d = s.model_dump() if hasattr(s, "model_dump") else dict(s)
            d["name"] = COLUMN_MAP.get(d["name"].strip(), d["name"].strip())
            cols.append(d)
        return {
            "dataset_id": dataset_id,
            "version_id": version_id,
            "version": resolved_version,
            "columns": cols,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/column-roles")
async def get_column_roles(
    dataset_id: str,
    version_id: str | None = Query(None, description="Version UUID; omit for current version"),
    version: int | None = Query(None, description="Version number; optional alternative to version_id"),
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Get column role assignments for a dataset version."""
    try:
        roles = mgr.get_column_roles(dataset_id, version_id=version_id, version=version)
        return {
            "dataset_id": dataset_id,
            "version_id": version_id,
            "version": version,
            "column_roles": roles,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{dataset_id}", response_model=DatasetDetailResponse)
async def update_dataset_metadata(
    dataset_id: str,
    update: DatasetUpdateRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Update dataset metadata (name, description, tags)."""
    ds = mgr.update_metadata(
        dataset_id,
        name=update.name,
        description=update.description,
        tags=update.tags,
    )
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return DatasetDetailResponse(
        id=ds.id,
        name=ds.name,
        description=ds.description,
        tags=ds.tags,
        author=ds.author,
        created_at=ds.created_at,
        current_version=ds.current_version,
        row_count=ds.row_count,
        file_size_bytes=ds.file_size_bytes,
        sha256=ds.sha256,
        status=ds.status,
        schema_info=[s.model_dump() for s in ds.schema_info],
        default_branch_id=ds.default_branch_id,
    )


@router.delete("/{dataset_id}", response_model=DatasetDeleteResponse)
async def delete_dataset(
    dataset_id: str,
    force: bool = Query(False),
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Delete a dataset (soft delete). Returns dependency warnings if force=False."""
    try:
        result = mgr.delete_dataset(dataset_id, force=force)
        return DatasetDeleteResponse(
            deleted=result["deleted"],
            reason=result.get("reason"),
            dependencies=result.get("dependencies"),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dataset_id}/subset", response_model=DatasetUploadResponse)
async def create_subset(
    dataset_id: str,
    req: SubsetRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Create a new dataset from a filtered subset."""
    try:
        meta = mgr.create_subset(
            dataset_id=dataset_id,
            filter_config=req.filter_config,
            name=req.name,
            description=req.description,
            author=req.author,
            version=req.version,
        )
        return DatasetUploadResponse(
            id=meta.id,
            name=meta.name,
            description=meta.description,
            tags=meta.tags,
            author=meta.author,
            created_at=meta.created_at,
            current_version=meta.current_version,
            row_count=meta.row_count,
            file_size_bytes=meta.file_size_bytes,
            sha256=meta.sha256,
            schema_info=[s.model_dump() for s in meta.schema_info],
            default_branch_id=meta.default_branch_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/versions", response_model=list[DatasetVersionResponse])
async def get_dataset_versions(
    dataset_id: str,
    branch_id: str | None = Query(None),
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Get version history for a dataset, optionally filtered by branch."""
    ds = mgr.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    versions = mgr.get_dataset_versions(dataset_id, branch_id=branch_id)
    return [DatasetVersionResponse(**v.model_dump()) for v in versions]


@router.patch("/{dataset_id}/versions/{version_id}", response_model=DatasetVersionResponse)
async def update_version_metadata(
    dataset_id: str,
    version_id: str,
    req: VersionUpdateRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Update the reason/description of a specific version."""
    try:
        ver = mgr.update_version_metadata(dataset_id, version_id, req.reason)
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{dataset_id}/versions/{version_id}")
async def delete_version(
    dataset_id: str,
    version_id: str,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Delete a specific version (soft delete). Cannot delete the only version on a branch."""
    try:
        return mgr.delete_version(dataset_id, version_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class MoveVersionRequest(BaseModel):
    target_branch_id: str
    author: str = ""


@router.post("/{dataset_id}/versions/{version_id}/move")
async def move_version_to_branch(
    dataset_id: str,
    version_id: str,
    req: MoveVersionRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Move a version to a different branch."""
    try:
        ver = mgr.move_version_to_branch(
            dataset_id,
            version_id,
            req.target_branch_id,
            author=req.author,
        )
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class CopyVersionRequest(BaseModel):
    new_reason: str
    author: str = ""
    branch_id: str | None = None


@router.post("/{dataset_id}/versions/{version_id}/copy", response_model=DatasetVersionResponse)
async def copy_version(
    dataset_id: str,
    version_id: str,
    req: CopyVersionRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Create a copy of a version with modified metadata."""
    try:
        ver = mgr.copy_version(
            dataset_id,
            version_id,
            req.new_reason,
            author=req.author,
            branch_id=req.branch_id,
        )
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class RestoreVersionRequest(BaseModel):
    reason: str = ""
    author: str = ""


@router.post("/{dataset_id}/versions/{version_id}/restore", response_model=DatasetVersionResponse)
async def restore_version(
    dataset_id: str,
    version_id: str,
    req: RestoreVersionRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Restore a historical version as a new latest version on its branch."""
    try:
        ver = mgr.restore_version(
            dataset_id,
            version_id,
            reason=req.reason,
            author=req.author,
        )
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class SetVersionDefaultRequest(BaseModel):
    author: str = ""


@router.post("/{dataset_id}/versions/{version_id}/set-default", response_model=DatasetVersionResponse)
async def set_version_as_default(
    dataset_id: str,
    version_id: str,
    req: SetVersionDefaultRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Set a version as dataset default (switch branch if needed and make it current)."""
    try:
        ver = mgr.set_version_as_default(dataset_id, version_id, author=req.author)
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/branches", response_model=list[BranchResponse])
async def list_branches(
    dataset_id: str,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """List all branches for a dataset."""
    ds = mgr.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    # Ensure main branch exists (migration helper for old datasets)
    mgr._get_or_create_main_branch(dataset_id)
    branches = mgr.list_branches(dataset_id)
    return [BranchResponse(**b.model_dump()) for b in branches]


@router.post("/{dataset_id}/branches", response_model=BranchResponse)
async def create_branch(
    dataset_id: str,
    req: BranchCreateRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Create a new branch from a specific version."""
    ds = mgr.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    try:
        branch = mgr.create_branch(
            dataset_id=dataset_id,
            name=req.name,
            from_version_id=req.from_version_id,
            author=req.author,
            description=req.description,
        )
        return BranchResponse(**branch.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/branches/{branch_id}", response_model=BranchResponse)
async def get_branch(
    dataset_id: str,
    branch_id: str,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Get a specific branch."""
    branch = mgr.get_branch(branch_id)
    if branch is None or branch.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail=f"Branch not found: {branch_id}")
    return BranchResponse(**branch.model_dump())


@router.patch("/{dataset_id}/branches/{branch_id}", response_model=BranchResponse)
async def update_branch(
    dataset_id: str,
    branch_id: str,
    req: BranchUpdateRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Update branch name and/or description."""
    try:
        branch = mgr.update_branch(
            dataset_id=dataset_id,
            branch_id=branch_id,
            name=req.name,
            description=req.description,
        )
        return BranchResponse(**branch.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{dataset_id}/branches/{branch_id}")
async def delete_branch(
    dataset_id: str,
    branch_id: str,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Delete a branch (soft delete). Cannot delete the default branch."""
    try:
        return mgr.delete_branch(dataset_id, branch_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/branches/{branch_id}/set-default")
async def set_default_branch(
    dataset_id: str,
    branch_id: str,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Set a branch as the default (content source) for this dataset."""
    try:
        return mgr.set_default_branch(dataset_id, branch_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Data editing (creates new versions)
# ---------------------------------------------------------------------------

@router.patch("/{dataset_id}/cells", response_model=DatasetVersionResponse)
async def update_dataset_cells(
    dataset_id: str,
    req: BatchCellUpdateRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Apply cell-level edits; saves a new immutable version."""
    try:
        ver = mgr.update_cells(
            dataset_id,
            changes=[ch.model_dump() for ch in req.changes],
            reason=req.reason,
            author=req.author,
            branch_id=req.branch_id,
        )
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/rows", response_model=DatasetVersionResponse)
async def add_dataset_rows(
    dataset_id: str,
    req: AddRowsRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Append new rows; saves a new immutable version."""
    try:
        ver = mgr.add_rows(
            dataset_id,
            new_rows=req.rows,
            reason=req.reason,
            author=req.author,
            branch_id=req.branch_id,
        )
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{dataset_id}/rows", response_model=DatasetVersionResponse)
async def delete_dataset_rows(
    dataset_id: str,
    req: DeleteRowsRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Delete rows by index; saves a new immutable version."""
    try:
        ver = mgr.delete_rows(
            dataset_id,
            row_indices=req.row_indices,
            reason=req.reason,
            author=req.author,
            branch_id=req.branch_id,
        )
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/{dataset_id}/columns", response_model=DatasetVersionResponse)
async def rename_dataset_columns(
    dataset_id: str,
    req: RenameColumnsRequest,
    mgr: DatasetManager = Depends(get_dataset_manager),
):
    """Rename one or more columns; saves a new immutable version with updated column roles."""
    try:
        ver = mgr.rename_columns(
            dataset_id,
            renames=req.renames,
            reason=req.reason,
            author=req.author,
            branch_id=req.branch_id,
        )
        return DatasetVersionResponse(**ver.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
