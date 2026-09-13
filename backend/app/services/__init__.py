"""Services package for FinPilot.

Structural foundation for domain services, data providers,
and business logic.
"""

from app.services.document_chunker import (
    ChunkingConfig,
    ChunkingError,
    DocumentChunker,
    EmptyDocumentChunkingError,
    InvalidChunkConfigurationError,
    chunk_document,
)
from app.services.document_metadata import (
    InvalidMetadataError,
    MetadataError,
    MissingSourceProvenanceError,
    attach_metadata_to_chunked_document,
    build_document_chunks_metadata,
    create_chunk_metadata,
)
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
    "ChunkingConfig",
    "ChunkingError",
    "CorruptedDocumentError",
    "DocumentChunker",
    "DocumentCorruptedFormatError",
    "DocumentEmptyError",
    "DocumentExtractionError",
    "DocumentFileSizeExceededError",
    "DocumentNotFoundExtractionError",
    "DocumentTextExtractor",
    "DocumentTextValidationError",
    "DocumentUnsupportedTypeError",
    "DocumentValidationError",
    "EmptyDocumentChunkingError",
    "EmptyExtractedTextError",
    "EncryptedDocumentError",
    "ExtractedDocumentValidator",
    "InvalidChunkConfigurationError",
    "InvalidMetadataError",
    "InvalidTickerError",
    "MalformedDocumentError",
    "MalformedExtractionResultError",
    "MetadataError",
    "MissingSourceProvenanceError",
    "NewsProcessor",
    "PdfTextExtractor",
    "PlainTextExtractor",
    "ScannedDocumentError",
    "UnusableExtractedTextError",
    "UnsupportedDocumentError",
    "UnsupportedDocumentFormatError",
    "attach_metadata_to_chunked_document",
    "build_document_chunks_metadata",
    "calculate_fundamental_metrics",
    "calculate_technical_metrics",
    "chunk_document",
    "clean_extracted_text",
    "create_chunk_metadata",
    "get_document_extractor",
    "validate_document_content",
    "validate_document_file",
    "validate_extracted_document",
    "validate_stored_document",
    "validate_ticker",
]
