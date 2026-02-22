"""Structured logging setup using structlog."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure structlog for structured, consistent logging."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_file:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(log_file, encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(_fh)


def get_logger(name: str) -> Any:
    """Return a structlog logger bound with the given name."""
    return structlog.get_logger(name)
