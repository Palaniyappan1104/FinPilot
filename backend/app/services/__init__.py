"""Services package for FinPilot.

Structural foundation for domain services, data providers,
and business logic.
"""

from app.services.document_text_validator import (
    CorruptedDocumentError,
    DocumentTextValidationError,
    EmptyExtractedTextError,
    ExtractedDocumentValidator,
    MalformedExtractionResultError,
    UnsupportedDocumentError,
    UnusableExtractedTextError,
    validate_document_content,
    validate_extracted_document,
    validate_stored_document,
)
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
    "CorruptedDocumentError",
    "DocumentCorruptedFormatError",
    "DocumentEmptyError",
    "DocumentExtractionError",
    "DocumentFileSizeExceededError",
    "DocumentNotFoundExtractionError",
    "DocumentTextExtractor",
    "DocumentTextValidationError",
    "DocumentUnsupportedTypeError",
    "DocumentValidationError",
    "EmptyExtractedTextError",
    "EncryptedDocumentError",
    "ExtractedDocumentValidator",
    "InvalidTickerError",
    "MalformedDocumentError",
    "MalformedExtractionResultError",
    "NewsProcessor",
    "PdfTextExtractor",
    "PlainTextExtractor",
    "ScannedDocumentError",
    "UnusableExtractedTextError",
    "UnsupportedDocumentError",
    "UnsupportedDocumentFormatError",
    "calculate_fundamental_metrics",
    "calculate_technical_metrics",
    "clean_extracted_text",
    "get_document_extractor",
    "validate_document_content",
    "validate_document_file",
    "validate_extracted_document",
    "validate_stored_document",
    "validate_ticker",
]
