"""FastAPI application entry point."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable
_backend_dir = Path(__file__).parent.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.runs import router as runs_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.artifacts import router as artifacts_router
from src.api.routes.datasets import router as datasets_router
from src.api.routes.models import router as models_router
from src.api.routes.training import router as training_router
from src.api.routes.analyses import router as analyses_router
from src.api.routes.saved_filters import router as saved_filters_router
from src.api.routes.summary import router as summary_router
from src.utils.logging import configure_logging

configure_logging()

app = FastAPI(
    title="Student Feedback Analysis Platform API",
    version="0.1.0",
    description=(
        "Backend API for the Multilingual Student Feedback Analysis Platform. "
        "Batch-only pipeline. Not for real-time or individual-level decisions."
    ),
)

# CORS — allow Next.js frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(runs_router)
app.include_router(metrics_router)
app.include_router(artifacts_router)
app.include_router(datasets_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(analyses_router)
app.include_router(saved_filters_router)
app.include_router(summary_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/")
async def root():
    return JSONResponse({
        "name": "Student Feedback Analysis Platform API",
        "version": "0.1.0",
        "docs": "/docs",
        "disclaimer": "Not for individual-level decisions. No causal claims.",
    })
