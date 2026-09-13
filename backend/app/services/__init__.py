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
from app.services.text_extraction import (
    DocumentTextExtractor,
    PdfTextExtractor,
    PlainTextExtractor,
    clean_extracted_text,
    get_document_extractor,
)
from app.services.text_extraction_exceptions import (
    DocumentExtractionError,
    DocumentNotFoundExtractionError,
    EncryptedDocumentError,
    MalformedDocumentError,
    ScannedDocumentError,
    UnsupportedDocumentFormatError,
)

__all__ = [
    "DocumentCorruptedFormatError",
    "DocumentEmptyError",
    "DocumentExtractionError",
    "DocumentFileSizeExceededError",
    "DocumentNotFoundExtractionError",
    "DocumentTextExtractor",
    "DocumentUnsupportedTypeError",
    "DocumentValidationError",
    "EncryptedDocumentError",
    "InvalidTickerError",
    "MalformedDocumentError",
    "NewsProcessor",
    "PdfTextExtractor",
    "PlainTextExtractor",
    "ScannedDocumentError",
    "UnsupportedDocumentFormatError",
    "calculate_fundamental_metrics",
    "calculate_technical_metrics",
    "clean_extracted_text",
    "get_document_extractor",
    "validate_document_file",
    "validate_ticker",
]
