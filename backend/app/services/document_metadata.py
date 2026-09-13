"""Document Metadata Service (Phase 9.5).

Builds, validates, and attaches structured provenance metadata to
DocumentChunk models for later RAG vector search, query embedding,
and verifiable citation.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Union

from app.models.documents import (
    ChunkedDocument,
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    StoredDocument,
    ValidatedDocument,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# TYPED EXCEPTIONS (Phase 9.5)
# ===========================================================================


class MetadataError(Exception):
    """Base exception for all document chunk metadata errors."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
        code: str = "METADATA_ERROR",
    ) -> None:
        self.message = message
        self.document_id = document_id
        self.code = code
        super().__init__(f"[{code}] (doc={document_id}) {message}")


class InvalidMetadataError(MetadataError):
    """Raised when metadata values are invalid, malformed, or out of bounds."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code="INVALID_METADATA",
        )


class MissingSourceProvenanceError(MetadataError):
    """Raised when required source provenance attributes are missing."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code="MISSING_SOURCE_PROVENANCE",
        )


# ===========================================================================
# METADATA BUILDER FUNCTIONS (Phase 9.5)
# ===========================================================================


def create_chunk_metadata(
    chunk: DocumentChunk,
    source_document: str,
    upload_date: Optional[datetime] = None,
    document_id: Optional[str] = None,
    ticker: Optional[str] = None,
    document_type: Optional[Union[DocumentType, str]] = None,
) -> ChunkMetadata:
    """Create a validated ChunkMetadata record for a DocumentChunk.

    Args:
        chunk: Source DocumentChunk instance.
        source_document: Original document filename or source reference.
        upload_date: UTC timestamp when document was uploaded (defaults to now).
        document_id: Document ID override if different from chunk.
        ticker: Ticker symbol override if different from chunk.
        document_type: DocumentType override if different from chunk.

    Returns:
        ChunkMetadata: Fully validated metadata record.

    Raises:
        MissingSourceProvenanceError: If source_document, document_id, or ticker
            is missing or blank.
        InvalidMetadataError: If document_type, page numbers, or dates are invalid.
    """
    if chunk is None or not isinstance(chunk, DocumentChunk):
        raise InvalidMetadataError(
            "Expected a valid DocumentChunk instance.",
            document_id=document_id or "unknown",
        )

    # 1. Resolve & validate doc_id
    doc_id = (document_id or chunk.document_id or "").strip()
    if not doc_id:
        raise MissingSourceProvenanceError(
            "document_id is required and cannot be empty.",
            document_id="unknown",
        )

    # 2. Resolve & validate source_document
    clean_source = (source_document or "").strip()
    if not clean_source:
        raise MissingSourceProvenanceError(
            "source_document (filename) is required and cannot be empty.",
            document_id=doc_id,
        )

    # 3. Resolve & validate ticker
    clean_ticker = (ticker or chunk.ticker or "").strip().upper()
    if not clean_ticker:
        raise MissingSourceProvenanceError(
            "ticker is required and cannot be empty.",
            document_id=doc_id,
        )

    # 4. Resolve & validate document_type
    raw_doc_type = document_type or chunk.document_type
    if isinstance(raw_doc_type, str):
        try:
            resolved_doc_type = DocumentType(raw_doc_type.lower())
        except ValueError as e:
            raise InvalidMetadataError(
                f"Unsupported document type: '{raw_doc_type}'.",
                document_id=doc_id,
            ) from e
    elif isinstance(raw_doc_type, DocumentType):
        resolved_doc_type = raw_doc_type
    else:
        raise InvalidMetadataError(
            f"Invalid document type format: {raw_doc_type}.",
            document_id=doc_id,
        )

    # 5. Validate page metadata
    if not chunk.page_numbers or len(chunk.page_numbers) == 0:
        raise InvalidMetadataError(
            "page_numbers must contain at least one valid page number.",
            document_id=doc_id,
        )

    for p in chunk.page_numbers:
        if p <= 0:
            raise InvalidMetadataError(
                f"Page numbers must be positive 1-indexed integers, got {p}.",
                document_id=doc_id,
            )

    start_p = chunk.start_page if chunk.start_page > 0 else min(chunk.page_numbers)
    end_p = chunk.end_page if chunk.end_page >= start_p else max(chunk.page_numbers)
    if end_p < start_p:
        raise InvalidMetadataError(
            f"end_page ({end_p}) cannot be less than start_page ({start_p}).",
            document_id=doc_id,
        )

    # 6. Resolve upload_date
    resolved_upload_date = upload_date or datetime.now(timezone.utc)
    if not isinstance(resolved_upload_date, datetime):
        raise InvalidMetadataError(
            f"upload_date must be a datetime instance, got "
            f"{type(resolved_upload_date).__name__}.",
            document_id=doc_id,
        )
    if resolved_upload_date.tzinfo is None:
        resolved_upload_date = resolved_upload_date.replace(tzinfo=timezone.utc)

    return ChunkMetadata(
        chunk_id=chunk.chunk_id,
        document_id=doc_id,
        source_document=clean_source,
        ticker=clean_ticker,
        document_type=resolved_doc_type,
        page_numbers=list(chunk.page_numbers),
        start_page=start_p,
        end_page=end_p,
        upload_date=resolved_upload_date,
        chunk_index=chunk.chunk_index,
        section_name=chunk.section_name,
        character_count=chunk.character_count,
        word_count=chunk.word_count,
    )


def build_document_chunks_metadata(
    chunks: List[DocumentChunk],
    source_document: str,
    upload_date: Optional[datetime] = None,
    document_id: Optional[str] = None,
    ticker: Optional[str] = None,
    document_type: Optional[Union[DocumentType, str]] = None,
) -> List[ChunkMetadata]:
    """Build a list of ChunkMetadata records for multiple document chunks.

    All chunks share document-level provenance (source_document, ticker,
    upload_date, document_type, document_id) while retaining chunk-level
    specificity (chunk_id, page_numbers, chunk_index, section_name, counts).
    """
    if not chunks:
        return []

    return [
        create_chunk_metadata(
            chunk=c,
            source_document=source_document,
            upload_date=upload_date,
            document_id=document_id,
            ticker=ticker,
            document_type=document_type,
        )
        for c in chunks
    ]


def attach_metadata_to_chunked_document(
    chunked_doc: ChunkedDocument,
    source_document: Optional[str] = None,
    upload_date: Optional[datetime] = None,
    stored_doc: Optional[StoredDocument] = None,
    validated_doc: Optional[ValidatedDocument] = None,
) -> ChunkedDocument:
    """Attach ChunkMetadata to each DocumentChunk in a ChunkedDocument.

    Resolves provenance in order of precedence:
    1. Explicit arguments (source_document, upload_date)
    2. StoredDocument instance (if provided)
    3. ValidatedDocument instance (original_filename, uploaded_at)
    4. Default fallback to document_id for source filename, now() for upload_date

    Returns a new ChunkedDocument with enriched DocumentChunk objects.
    """
    if chunked_doc is None or not isinstance(chunked_doc, ChunkedDocument):
        raise InvalidMetadataError(
            "Expected a valid ChunkedDocument instance.",
            document_id="unknown",
        )

    # Determine source_document
    resolved_source: Optional[str] = source_document
    if not resolved_source and stored_doc:
        resolved_source = stored_doc.original_filename
    if not resolved_source and validated_doc and validated_doc.original_filename:
        resolved_source = validated_doc.original_filename
    if not resolved_source:
        resolved_source = f"{chunked_doc.document_id}.pdf"

    # Determine upload_date
    resolved_date: Optional[datetime] = upload_date
    if not resolved_date and stored_doc:
        resolved_date = stored_doc.uploaded_at
    if not resolved_date and validated_doc and validated_doc.uploaded_at:
        resolved_date = validated_doc.uploaded_at
    if not resolved_date:
        resolved_date = chunked_doc.chunked_at

    enriched_chunks: List[DocumentChunk] = []
    for c in chunked_doc.chunks:
        meta = create_chunk_metadata(
            chunk=c,
            source_document=resolved_source,
            upload_date=resolved_date,
            document_id=chunked_doc.document_id,
            ticker=chunked_doc.ticker,
            document_type=chunked_doc.document_type,
        )
        # Rebuild frozen DocumentChunk with metadata attached
        enriched_c = DocumentChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            ticker=c.ticker,
            document_type=c.document_type,
            chunk_index=c.chunk_index,
            text=c.text,
            character_count=c.character_count,
            word_count=c.word_count,
            page_numbers=c.page_numbers,
            start_page=c.start_page,
            end_page=c.end_page,
            section_name=c.section_name,
            metadata=meta,
            created_at=c.created_at,
        )
        enriched_chunks.append(enriched_c)

    return ChunkedDocument(
        document_id=chunked_doc.document_id,
        ticker=chunked_doc.ticker,
        document_type=chunked_doc.document_type,
        total_chunks=len(enriched_chunks),
        total_characters=chunked_doc.total_characters,
        chunk_size=chunked_doc.chunk_size,
        chunk_overlap=chunked_doc.chunk_overlap,
        chunks=enriched_chunks,
        chunked_at=chunked_doc.chunked_at,
    )
