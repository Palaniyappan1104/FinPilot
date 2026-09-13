"""Validation service for Document Upload (Phase 9.1).

Enforces:
- Ticker symbol validation and normalization.
- Supported file extension validation (PDF primary).
- Configurable file size limits (rejects empty and oversized files).
- File format sanity checks (PDF magic bytes validation).
"""

import os
import re
from typing import List, Optional

from app.core.config import get_settings


class DocumentValidationError(Exception):
    """Base exception for document upload validation failures."""

    def __init__(self, message: str, code: str = "VALIDATION_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return self.message


class DocumentEmptyError(DocumentValidationError):
    """Raised when uploaded file contains 0 bytes."""

    def __init__(self, message: str = "Uploaded file is empty (0 bytes).") -> None:
        super().__init__(message=message, code="EMPTY_FILE")


class DocumentFileSizeExceededError(DocumentValidationError):
    """Raised when uploaded file exceeds maximum configured size limit."""

    def __init__(self, file_size: int, max_size: int) -> None:
        message = (
            f"File size ({file_size} bytes) exceeds maximum allowed limit "
            f"({max_size} bytes)."
        )
        super().__init__(message=message, code="PAYLOAD_TOO_LARGE")
        self.file_size = file_size
        self.max_size = max_size


class DocumentUnsupportedTypeError(DocumentValidationError):
    """Raised when file extension or content type is not supported."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="UNSUPPORTED_FILE_TYPE")


class DocumentCorruptedFormatError(DocumentValidationError):
    """Raised when file content fails basic format sanity checks."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="CORRUPTED_FORMAT")


class InvalidTickerError(DocumentValidationError):
    """Raised when ticker format is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="INVALID_TICKER")


def validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker symbol.

    Args:
        ticker: Input stock ticker string.

    Returns:
        str: Normalized uppercase ticker symbol.

    Raises:
        InvalidTickerError: If ticker is empty, malformed, or invalid characters.
    """
    if not ticker or not ticker.strip():
        raise InvalidTickerError("Ticker symbol cannot be empty or whitespace.")

    clean = ticker.strip().upper()
    if not re.match(r"^[A-Z0-9.\-]{1,10}$", clean):
        raise InvalidTickerError(
            f"Invalid ticker symbol '{ticker}'. Expected 1-10 uppercase alphanumeric "
            f"characters (dots and dashes allowed)."
        )
    return clean


def validate_document_file(
    filename: Optional[str],
    content: bytes,
    content_type: Optional[str] = None,
    allowed_extensions: Optional[List[str]] = None,
    allowed_mime_types: Optional[List[str]] = None,
    max_size_bytes: Optional[int] = None,
) -> None:
    """Validate uploaded document file properties and sanity checks.

    Args:
        filename: Client-submitted filename.
        content: Raw binary file content.
        content_type: MIME content-type header if available.
        allowed_extensions: Allowed file extensions (defaults from settings).
        allowed_mime_types: Allowed MIME types (defaults from settings).
        max_size_bytes: Maximum allowed file size (defaults from settings).

    Raises:
        DocumentEmptyError: If content is empty.
        DocumentFileSizeExceededError: If content exceeds size limit.
        DocumentUnsupportedTypeError: If file extension or MIME is unsupported.
        DocumentCorruptedFormatError: If magic bytes check fails.
    """
    settings = get_settings()

    exts = allowed_extensions or settings.ALLOWED_DOCUMENT_EXTENSIONS
    mimes = allowed_mime_types or settings.ALLOWED_DOCUMENT_MIME_TYPES
    max_size = (
        max_size_bytes if max_size_bytes is not None else settings.MAX_UPLOAD_SIZE_BYTES
    )

    # 1. Filename validation
    if not filename or not filename.strip():
        raise DocumentUnsupportedTypeError("Filename cannot be empty.")

    clean_filename = filename.strip()
    _, ext = os.path.splitext(clean_filename)
    lower_ext = ext.lower()

    normalized_allowed_exts = [e.lower() for e in exts]
    if lower_ext not in normalized_allowed_exts:
        raise DocumentUnsupportedTypeError(
            f"File extension '{lower_ext or '(none)'}' is unsupported. "
            f"Allowed extensions: {normalized_allowed_exts}."
        )

    # 2. File size validation
    size = len(content)
    if size == 0:
        raise DocumentEmptyError("Uploaded file is empty (0 bytes).")

    if size > max_size:
        raise DocumentFileSizeExceededError(file_size=size, max_size=max_size)

    # 3. MIME type validation (when specified and non-generic)
    if content_type:
        clean_mime = content_type.split(";")[0].strip().lower()
        # Allow generic octet-stream from simple clients, but block explicit non-PDFs
        generic_types = {"application/octet-stream", "binary/octet-stream"}
        allowed_mimes = [m.lower() for m in mimes]
        if clean_mime not in allowed_mimes and clean_mime not in generic_types:
            raise DocumentUnsupportedTypeError(
                f"Content-Type '{clean_mime}' is unsupported. "
                f"Allowed MIME types: {allowed_mimes}."
            )

    # 4. Format sanity checks (PDF magic bytes)
    if lower_ext == ".pdf":
        if size < 8:
            raise DocumentCorruptedFormatError(
                "File is too small to be a valid PDF document."
            )
        # Standard PDF files must contain %PDF- in header (first 1024 bytes)
        header_sample = content[:1024]
        if b"%PDF-" not in header_sample:
            raise DocumentCorruptedFormatError(
                "File failed PDF sanity check: missing '%PDF-' header magic bytes. "
                "The file may be corrupted or disguised."
            )
