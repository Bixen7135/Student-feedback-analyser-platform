"""Saved filters API routes — CRUD for named filter configurations."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_db
from src.analysis.saved_filters import (
    create_saved_filter,
    delete_saved_filter,
    get_saved_filter,
    list_saved_filters,
    update_saved_filter,
)

router = APIRouter(prefix="/api/saved-filters", tags=["saved-filters"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CreateFilterRequest(BaseModel):
    name: str = Field(..., max_length=200)
    entity_type: str = Field(
        "analysis_results",
        max_length=100,
        description="Scope: analysis_results | datasets | models",
    )
    filter_config: dict = Field(
        default_factory=dict,
        description="Arbitrary filter configuration stored as JSON",
    )


class UpdateFilterRequest(BaseModel):
    name: str | None = Field(None, max_length=200)
    filter_config: dict | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=dict, status_code=201)
async def create_filter(body: CreateFilterRequest, db=Depends(get_db)):
    """Create a new saved filter configuration."""
    return create_saved_filter(
        db=db,
        name=body.name,
        entity_type=body.entity_type,
        filter_config=body.filter_config,
    )


@router.get("", response_model=list)
async def list_filters(
    entity_type: str | None = Query(
        None, description="Filter by entity_type (e.g. analysis_results)"
    ),
    db=Depends(get_db),
):
    """List all saved filters, optionally scoped to an entity_type."""
    return list_saved_filters(db=db, entity_type=entity_type)


@router.get("/{filter_id}", response_model=dict)
async def get_filter(filter_id: str, db=Depends(get_db)):
    """Fetch a single saved filter by ID."""
    record = get_saved_filter(db, filter_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Saved filter not found: {filter_id}")
    return record


@router.put("/{filter_id}", response_model=dict)
async def update_filter(filter_id: str, body: UpdateFilterRequest, db=Depends(get_db)):
    """Update a saved filter's name and/or filter_config."""
    result = update_saved_filter(
        db=db,
        filter_id=filter_id,
        name=body.name,
        filter_config=body.filter_config,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Saved filter not found: {filter_id}")
    return result


@router.delete("/{filter_id}")
async def delete_filter(filter_id: str, db=Depends(get_db)):
    """Delete a saved filter."""
    deleted = delete_saved_filter(db, filter_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Saved filter not found: {filter_id}")
    return {"deleted": True, "filter_id": filter_id}
