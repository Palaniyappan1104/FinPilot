"""Application logging configuration."""

import logging
import sys

from app.core.config import get_settings


def setup_logging() -> None:
    """Configure structured console logging for the application."""
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Set external libraries log levels to avoid noise
    logging.getLogger("uvicorn.access").handlers = []


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
