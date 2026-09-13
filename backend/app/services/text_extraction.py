"""Document text extraction service for Phase 9.2.

Provides:
- DocumentTextExtractor: Abstract base class for text extraction backends.
- PdfTextExtractor: Deterministic PDF extraction using pypdf.
- clean_extracted_text: Safe whitespace and artifact normalization
  preserving financial numbers.
"""

import io
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import pypdf
from pypdf.errors import PdfReadError, PdfStreamError

from app.core.logging import get_logger
from app.models.documents import (
    DocumentExtractionResult,
    DocumentType,
    ExtractedPage,
    StoredDocument,
)
from app.services.text_extraction_exceptions import (
    DocumentNotFoundExtractionError,
    EncryptedDocumentError,
    MalformedDocumentError,
    ScannedDocumentError,
    UnsupportedDocumentFormatError,
)
from app.storage.base import DocumentStorage
from app.storage.exceptions import StorageNotFoundError

logger = get_logger("app.services.text_extraction")


def clean_extracted_text(text: str) -> str:
    """Perform safe, deterministic normalization on extracted document text.

    Preserves exact financial figures, numbers, punctuation, and structural layout
    while normalizing line endings, trailing spaces, and PDF control artifacts.

    Rules:
    - Replaces Windows (CRLF) and legacy Mac (CR) line breaks with standard LF.
    - Replaces non-breaking spaces (\\xa0) with standard ASCII space.
    - Strips non-printable ASCII control characters (excluding tab and newline).
    - Strips trailing whitespace per line.
    - Collapses runs of 3 or more newlines into double newlines.
    - Does NOT mutate words, numbers, currencies, or symbols.
    """
    if not text:
        return ""

    # Normalize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize non-breaking and special spaces
    normalized = normalized.replace("\xa0", " ").replace("\u200b", "")

    # Remove non-printable control characters except \n and \t
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)

    # Strip trailing whitespace on each line while preserving indentation
    lines = [line.rstrip() for line in normalized.split("\n")]
    result = "\n".join(lines)

    # Collapse excessive blank lines (>= 3 newlines -> 2 newlines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


class DocumentTextExtractor(ABC):
    """Abstract interface for extracting structured text from stored documents."""

    @abstractmethod
    def extract_text(
        self,
        document: StoredDocument,
        storage: DocumentStorage,
    ) -> DocumentExtractionResult:
        """Extract page-by-page text from a stored document.

        Args:
            document: Metadata of the stored document.
            storage: DocumentStorage backend holding the document file.

        Returns:
            DocumentExtractionResult: Page-by-page extracted content and metadata.

        Raises:
            DocumentNotFoundExtractionError: If document is missing in storage.
            ScannedDocumentError: If document contains no extractable text.
            EncryptedDocumentError: If document is password-protected.
            MalformedDocumentError: If document is corrupt or invalid.
            UnsupportedDocumentFormatError: If document format cannot be extracted.
        """
        pass

    @abstractmethod
    def extract_from_bytes(
        self,
        content: bytes,
        document_id: str,
        ticker: str = "UNKNOWN",
        document_type: DocumentType = DocumentType.OTHER_SUPPORTED,
    ) -> DocumentExtractionResult:
        """Extract text directly from binary content in memory."""
        pass


class PdfTextExtractor(DocumentTextExtractor):
    """Deterministic PDF text extractor powered by pypdf.

    Extracts text page-by-page, preserves exact 1-indexed page boundaries,
    normalizes safe whitespace artifacts without altering financial figures,
    and detects scanned/image-only PDFs deterministically.
    """

    def extract_from_bytes(
        self,
        content: bytes,
        document_id: str,
        ticker: str = "UNKNOWN",
        document_type: DocumentType = DocumentType.OTHER_SUPPORTED,
    ) -> DocumentExtractionResult:
        """Extract page-by-page text from raw PDF bytes.

        Args:
            content: Raw PDF file bytes.
            document_id: Unique document identifier.
            ticker: Associated stock ticker symbol.
            document_type: Document category.

        Returns:
            DocumentExtractionResult: Extracted pages and aggregate character counts.

        Raises:
            MalformedDocumentError: If bytes are empty or unparseable.
            EncryptedDocumentError: If PDF is encrypted/password-protected.
            ScannedDocumentError: If all pages contain zero extractable text.
        """
        if not content or len(content) == 0:
            raise MalformedDocumentError(
                "Cannot extract text from empty (0 bytes) PDF payload.",
                document_id=document_id,
            )

        try:
            stream = io.BytesIO(content)
            reader = pypdf.PdfReader(stream)
        except (PdfReadError, PdfStreamError, ValueError) as e:
            logger.error("Failed to parse PDF '%s': %s", document_id, e)
            raise MalformedDocumentError(
                f"PDF parsing error: {e}", document_id=document_id
            ) from e
        except Exception as e:
            logger.error("Unexpected error opening PDF '%s': %s", document_id, e)
            raise MalformedDocumentError(
                f"Unexpected error parsing PDF: {e}", document_id=document_id
            ) from e

        # Handle encryption
        if reader.is_encrypted:
            try:
                # Attempt empty password decryption (handles standard encryption
                # with blank password)
                decryption_result = reader.decrypt("")
                # If decrypt returns 0 or False, document remains encrypted
                if decryption_result == 0:
                    raise EncryptedDocumentError(
                        "PDF document is password-protected or encrypted.",
                        document_id=document_id,
                    )
            except EncryptedDocumentError:
                raise
            except Exception as e:
                raise EncryptedDocumentError(
                    f"PDF document is password-protected or encrypted: {e}",
                    document_id=document_id,
                ) from e

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise MalformedDocumentError(
                "PDF document contains zero pages.",
                document_id=document_id,
            )

        extracted_pages: List[ExtractedPage] = []
        total_characters = 0
        has_any_extractable_text = False

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1
            try:
                raw_page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(
                    "Error extracting text from page %d of document '%s': %s",
                    page_number,
                    document_id,
                    e,
                )
                raw_page_text = ""

            cleaned_page_text = clean_extracted_text(raw_page_text)
            if cleaned_page_text:
                has_any_extractable_text = True

            char_count = len(cleaned_page_text)
            total_characters += char_count

            extracted_pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text=cleaned_page_text,
                    character_count=char_count,
                )
            )

        # Deterministic detection of scanned / image-only PDFs
        if not has_any_extractable_text or total_characters == 0:
            logger.warning(
                "PDF '%s' opened with %d page(s) but has no extractable text "
                "(likely scanned).",
                document_id,
                total_pages,
            )
            raise ScannedDocumentError(
                f"PDF document '{document_id}' contains {total_pages} page(s) "
                "but no extractable text. The file appears to be a scanned or "
                "image-only document. Optical Character Recognition (OCR) is "
                "required but not currently supported.",
                document_id=document_id,
            )

        logger.info(
            "Extracted text from doc '%s' (ticker='%s'): %d pages, %d chars",
            document_id,
            ticker,
            total_pages,
            total_characters,
        )

        return DocumentExtractionResult(
            document_id=document_id,
            ticker=ticker,
            document_type=document_type,
            total_pages=total_pages,
            total_characters=total_characters,
            pages=extracted_pages,
            extraction_status="completed",
        )

    def extract_text(
        self,
        document: StoredDocument,
        storage: DocumentStorage,
    ) -> DocumentExtractionResult:
        """Extract text from a StoredDocument retrieved via DocumentStorage.

        Args:
            document: StoredDocument metadata.
            storage: DocumentStorage provider.

        Returns:
            DocumentExtractionResult: Page-by-page extracted text and metadata.
        """
        # Validate format
        orig_name = document.original_filename.lower()
        if not orig_name.endswith(".pdf") and document.mime_type != "application/pdf":
            raise UnsupportedDocumentFormatError(
                f"Unsupported format for PDF text extraction: "
                f"'{document.original_filename}' (mime_type: "
                f"'{document.mime_type}'). Only PDF documents are supported.",
                document_id=document.document_id,
            )

        try:
            content = storage.get_bytes(document.storage_path)
        except StorageNotFoundError as e:
            raise DocumentNotFoundExtractionError(
                f"Document file missing from storage at '{document.storage_path}'.",
                document_id=document.document_id,
            ) from e

        return self.extract_from_bytes(
            content=content,
            document_id=document.document_id,
            ticker=document.ticker,
            document_type=document.document_type,
        )


class PlainTextExtractor(DocumentTextExtractor):
    """Deterministic text extractor for UTF-8 / ASCII plain text files."""

    def extract_from_bytes(
        self,
        content: bytes,
        document_id: str,
        ticker: str = "UNKNOWN",
        document_type: DocumentType = DocumentType.OTHER_SUPPORTED,
    ) -> DocumentExtractionResult:
        """Extract text from raw plain text bytes."""
        if not content or len(content) == 0:
            raise MalformedDocumentError(
                "Cannot extract text from empty (0 bytes) text payload.",
                document_id=document_id,
            )

        try:
            raw_text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = content.decode("latin-1")
            except Exception as e:
                raise MalformedDocumentError(
                    f"Failed to decode text document: {e}",
                    document_id=document_id,
                ) from e

        cleaned = clean_extracted_text(raw_text)
        if not cleaned:
            raise ScannedDocumentError(
                f"Text document '{document_id}' contains no usable text content.",
                document_id=document_id,
            )

        return DocumentExtractionResult(
            document_id=document_id,
            ticker=ticker,
            document_type=document_type,
            total_pages=1,
            total_characters=len(cleaned),
            pages=[
                ExtractedPage(
                    page_number=1,
                    text=cleaned,
                    character_count=len(cleaned),
                )
            ],
            extraction_status="completed",
        )

    def extract_text(
        self,
        document: StoredDocument,
        storage: DocumentStorage,
    ) -> DocumentExtractionResult:
        """Extract text from stored plain text document."""
        try:
            content = storage.get_bytes(document.storage_path)
        except StorageNotFoundError as e:
            raise DocumentNotFoundExtractionError(
                f"Document file missing from storage at '{document.storage_path}'.",
                document_id=document.document_id,
            ) from e

        return self.extract_from_bytes(
            content=content,
            document_id=document.document_id,
            ticker=document.ticker,
            document_type=document.document_type,
        )


def get_document_extractor(
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> DocumentTextExtractor:
    """Factory returning appropriate DocumentTextExtractor for given document format."""
    clean_mime = (mime_type or "").strip().lower()
    clean_name = (filename or "").strip().lower()

    if clean_mime == "application/pdf" or clean_name.endswith(".pdf"):
        return PdfTextExtractor()

    if clean_mime in ("text/plain", "application/text") or clean_name.endswith(".txt"):
        return PlainTextExtractor()

    if not clean_mime and not clean_name:
        return PdfTextExtractor()

    raise UnsupportedDocumentFormatError(
        f"No text extractor registered for format '{mime_type or filename}'. "
        "PDF is the primary supported extraction format for Phase 9.2."
    )
