"""Domain models and schemas for Document Upload (Phase 9.1).

Defines the DocumentType enumeration, the StoredDocument metadata model,
and the public DocumentUploadResponse API schema.
"""

from datetime import datetime, timezone
from enum import Enum

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
