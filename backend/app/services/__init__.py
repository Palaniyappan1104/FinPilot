"""Services package for FinPilot.

Structural foundation for domain services, data providers,
and business logic.
"""

from app.services.document_validator import (
    DocumentCorruptedFormatError,
    DocumentEmptyError,
    DocumentFileSizeExceededError,
    DocumentUnsupportedTypeError,
    DocumentValidationError,
    InvalidTickerError,
    validate_document_file,
    validate_ticker,
)
from app.services.fundamental_metrics import calculate_fundamental_metrics
from app.services.news_processor import NewsProcessor
from app.services.technical_metrics import calculate_technical_metrics

__all__ = [
    "DocumentCorruptedFormatError",
    "DocumentEmptyError",
    "DocumentFileSizeExceededError",
    "DocumentUnsupportedTypeError",
    "DocumentValidationError",
    "InvalidTickerError",
    "NewsProcessor",
    "calculate_fundamental_metrics",
    "calculate_technical_metrics",
    "validate_document_file",
    "validate_ticker",
]
