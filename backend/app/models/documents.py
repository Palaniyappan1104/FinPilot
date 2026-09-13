"""Domain models and schemas for Document Upload (Phase 9.1).

Defines the DocumentType enumeration, the StoredDocument metadata model,
and the public DocumentUploadResponse API schema.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    """Controlled enumeration of supported financial document types."""

    ANNUAL_REPORT = "annual_report"
    INVESTOR_PRESENTATION = "investor_presentation"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    COMPANY_REPORT = "company_report"
    OTHER_SUPPORTED = "other_supported"


class StoredDocument(BaseModel):
    """Metadata record for a document saved into storage."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Unique server-generated document identifier")
    ticker: str = Field(description="Uppercase stock ticker symbol")
    document_type: DocumentType = Field(description="Classified document category")
    original_filename: str = Field(
        description="Original sanitized filename submitted by client"
    )
    stored_filename: str = Field(description="Server-side storage filename")
    storage_path: str = Field(description="Abstracted storage path or reference key")
    file_size_bytes: int = Field(ge=0, description="Size of file in bytes")
    sha256_checksum: str = Field(description="SHA-256 hash of file contents")
    mime_type: str = Field(default="application/pdf", description="MIME content type")
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of document upload",
    )


class DocumentUploadResponse(BaseModel):
    """FastAPI API response schema returned after successful document upload."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Unique document identifier")
    ticker: str = Field(description="Associated stock ticker symbol")
    document_type: DocumentType = Field(description="Document type category")
    original_filename: str = Field(description="Sanitized client filename")
    storage_path: str = Field(description="Relative storage reference key")
    file_size_bytes: int = Field(description="Size of stored file in bytes")
    upload_status: str = Field(
        default="stored", description="Current upload lifecycle status"
    )
    uploaded_at: datetime = Field(description="UTC timestamp of upload")


class ExtractedPage(BaseModel):
    """Cleaned text extracted from a single physical document page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1, description="1-indexed physical page number")
    text: str = Field(description="Cleaned extracted text content of page")
    character_count: int = Field(ge=0, description="Length of extracted page text")


class DocumentExtractionResult(BaseModel):
    """Structured result of document text extraction."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Unique document identifier")
    ticker: str = Field(description="Associated stock ticker symbol")
    document_type: DocumentType = Field(description="Category of document")
    total_pages: int = Field(ge=1, description="Total physical pages in document")
    total_characters: int = Field(
        ge=0, description="Total extracted text character count"
    )
    pages: list[ExtractedPage] = Field(
        description="List of extracted page contents in sequential order"
    )
    extraction_status: str = Field(
        default="completed", description="Lifecycle status of extraction"
    )
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of extraction completion",
    )


class ValidatedPage(BaseModel):
    """Validated text extracted from a single physical document page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1, description="1-indexed physical page number")
    text: str = Field(description="Cleaned extracted text content of page")
    character_count: int = Field(ge=0, description="Length of page text")
    word_count: int = Field(ge=0, description="Word count of page text")
    is_usable: bool = Field(
        default=True, description="Whether page contains usable text content"
    )


class ValidatedDocument(BaseModel):
    """Structured result of document text validation (Phase 9.3)."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Unique document identifier")
    ticker: str = Field(description="Associated stock ticker symbol")
    document_type: DocumentType = Field(description="Category of document")
    total_pages: int = Field(ge=1, description="Total physical pages in document")
    usable_pages: int = Field(ge=0, description="Count of pages containing usable text")
    total_characters: int = Field(
        ge=0, description="Total characters across usable pages"
    )
    total_words: int = Field(ge=0, description="Total words across usable pages")
    pages: list[ValidatedPage] = Field(
        description="List of validated page models in sequential order"
    )
    validation_status: str = Field(
        default="valid", description="Lifecycle validation status"
    )
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of validation completion",
    )

    def get_usable_pages(self) -> list[ValidatedPage]:
        """Return only pages flagged as containing usable text."""
        return [p for p in self.pages if p.is_usable]

    def get_full_text(self) -> str:
        """Return concatenated text of all usable pages separated by newlines."""
        return "\n\n".join(p.text for p in self.pages if p.is_usable and p.text)


class DocumentChunk(BaseModel):
    """Structured text chunk extracted from a validated document (Phase 9.4)."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique deterministic chunk identifier")
    document_id: str = Field(description="Source document identifier")
    ticker: str = Field(description="Associated stock ticker symbol")
    document_type: DocumentType = Field(description="Category of document")
    chunk_index: int = Field(ge=1, description="1-indexed sequential chunk position")
    text: str = Field(description="Extracted chunk text content")
    character_count: int = Field(ge=1, description="Length of chunk text in characters")
    word_count: int = Field(ge=1, description="Word count of chunk text")
    page_numbers: list[int] = Field(
        description="List of 1-indexed source pages spanning this chunk"
    )
    start_page: int = Field(ge=1, description="Initial source page for chunk text")
    end_page: int = Field(ge=1, description="Ending source page for chunk text")
    section_name: Optional[str] = Field(
        default=None,
        description="Reliable detected section title or heading if present",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of chunk creation",
    )


class ChunkedDocument(BaseModel):
    """Container holding all chunks generated from a ValidatedDocument."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Source document identifier")
    ticker: str = Field(description="Associated stock ticker symbol")
    document_type: DocumentType = Field(description="Category of document")
    total_chunks: int = Field(ge=0, description="Total number of chunks produced")
    total_characters: int = Field(
        ge=0, description="Total characters across all chunks"
    )
    chunk_size: int = Field(ge=1, description="Configured target chunk size")
    chunk_overlap: int = Field(ge=0, description="Configured chunk overlap")
    chunks: list[DocumentChunk] = Field(
        description="Ordered list of DocumentChunk objects"
    )
    chunked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when chunking completed",
    )
