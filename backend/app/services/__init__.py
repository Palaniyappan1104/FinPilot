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
from app.services.document_embedder import (
    DocumentEmbedder,
    chunk_to_embedding_request,
    embed_chunked_document,
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
from app.services.document_vector_service import (
    chunk_embedding_to_vector_record,
    store_chunk_embeddings,
    store_document_embeddings,
)
from app.services.fundamental_metrics import calculate_fundamental_metrics
from app.services.news_processor import NewsProcessor
from app.services.query_embedder import QueryEmbedder, embed_query
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
    "DocumentEmbedder",
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
    "QueryEmbedder",
    "ScannedDocumentError",
    "UnusableExtractedTextError",
    "UnsupportedDocumentError",
    "UnsupportedDocumentFormatError",
    "attach_metadata_to_chunked_document",
    "build_document_chunks_metadata",
    "calculate_fundamental_metrics",
    "calculate_technical_metrics",
    "chunk_document",
    "chunk_embedding_to_vector_record",
    "chunk_to_embedding_request",
    "clean_extracted_text",
    "create_chunk_metadata",
    "embed_chunked_document",
    "embed_query",
    "get_document_extractor",
    "store_chunk_embeddings",
    "store_document_embeddings",
    "validate_document_content",
    "validate_document_file",
    "validate_extracted_document",
    "validate_stored_document",
    "validate_ticker",
]
