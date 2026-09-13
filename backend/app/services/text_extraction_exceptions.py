"""Exceptions for document text extraction (Phase 9.2)."""


class DocumentExtractionError(Exception):
    """Base exception for all document text extraction errors."""

    def __init__(self, message: str, document_id: str = "unknown") -> None:
        super().__init__(message)
        self.message = message
        self.document_id = document_id

    def __str__(self) -> str:
        return f"[{self.document_id}] {self.message}"


class ScannedDocumentError(DocumentExtractionError):
    """Raised when a PDF contains no extractable text (e.g. scanned or image-only)."""


class EncryptedDocumentError(DocumentExtractionError):
    """Raised when a document is password-protected or encrypted."""


class MalformedDocumentError(DocumentExtractionError):
    """Raised when document data is corrupted or cannot be parsed."""


class UnsupportedDocumentFormatError(DocumentExtractionError):
    """Raised when the document format is not supported for text extraction."""


class DocumentNotFoundExtractionError(DocumentExtractionError):
    """Raised when the document to extract is missing from storage."""
