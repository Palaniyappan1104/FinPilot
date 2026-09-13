"""Document text validation service for Phase 9.3.

Provides:
- DocumentTextValidationError and typed subclasses:
  EmptyExtractedTextError, UnusableExtractedTextError,
  MalformedExtractionResultError, UnsupportedDocumentError,
  CorruptedDocumentError.
- ExtractedDocumentValidator: Deterministic validator for extracted text.
- validate_extracted_document: Helper validating DocumentExtractionResult.
- validate_document_content: Pipeline helper extracting and validating bytes.
- validate_stored_document: Storage helper for StoredDocument instances.
"""

import re
from typing import Any, List, Optional

from app.core.logging import get_logger
from app.models.documents import (
    DocumentExtractionResult,
    DocumentType,
    ExtractedPage,
    StoredDocument,
    ValidatedDocument,
    ValidatedPage,
)
from app.services.text_extraction import get_document_extractor
from app.services.text_extraction_exceptions import (
    DocumentExtractionError,
    EncryptedDocumentError,
    MalformedDocumentError,
    ScannedDocumentError,
    UnsupportedDocumentFormatError,
)
from app.storage.local import DocumentStorage

logger = get_logger(__name__)


# ===========================================================================
# TYPED DOMAIN EXCEPTIONS (Phase 9.3)
# ===========================================================================


class DocumentTextValidationError(Exception):
    """Base exception for document text validation failures (Phase 9.3)."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
        code: str = "VALIDATION_FAILED",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.document_id = document_id
        self.code = code

    def __str__(self) -> str:
        return f"[{self.document_id}] {self.message}"


class EmptyExtractedTextError(DocumentTextValidationError):
    """Raised when extracted text is completely empty or whitespace-only."""

    def __init__(
        self,
        message: str = "Extracted document text is empty or whitespace-only.",
        document_id: str = "unknown",
    ) -> None:
        super().__init__(message=message, document_id=document_id, code="EMPTY_TEXT")


class UnusableExtractedTextError(DocumentTextValidationError):
    """Raised when extracted text contains insufficient usable content."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
    ) -> None:
        super().__init__(message=message, document_id=document_id, code="UNUSABLE_TEXT")


class MalformedExtractionResultError(DocumentTextValidationError):
    """Raised when extraction result structure is malformed or inconsistent."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code="MALFORMED_EXTRACTION_RESULT",
        )


class UnsupportedDocumentError(DocumentTextValidationError):
    """Raised when document format, MIME type, or type is unsupported."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code="UNSUPPORTED_DOCUMENT",
        )


class CorruptedDocumentError(DocumentTextValidationError):
    """Raised when document content is corrupted, unparseable, or garbage."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code="CORRUPTED_DOCUMENT",
        )


# ===========================================================================
# EXTRACTED DOCUMENT VALIDATOR (Phase 9.3.1 & 9.3.2)
# ===========================================================================


class ExtractedDocumentValidator:
    """Deterministic validator for extracted document text and structure."""

    def __init__(
        self,
        min_total_characters: int = 20,
        min_words: int = 5,
        max_unprintable_ratio: float = 0.15,
        min_page_characters: int = 1,
    ) -> None:
        """Initialize validator with quality thresholds.

        Args:
            min_total_characters: Minimum non-whitespace characters required.
            min_words: Minimum word count required across the document.
            max_unprintable_ratio: Maximum allowable ratio of corrupted or
                unprintable characters.
            min_page_characters: Character threshold below which a specific
                page is marked as non-usable.
        """
        self.min_total_characters = min_total_characters
        self.min_words = min_words
        self.max_unprintable_ratio = max_unprintable_ratio
        self.min_page_characters = min_page_characters

    def validate(self, extraction_result: Any) -> ValidatedDocument:
        """Validate an extracted document for well-formedness and usable text.

        Args:
            extraction_result: DocumentExtractionResult instance to validate.

        Returns:
            ValidatedDocument: Strongly-typed validated document model.

        Raises:
            MalformedExtractionResultError: If result structure is inconsistent.
            EmptyExtractedTextError: If text is completely empty or whitespace.
            UnusableExtractedTextError: If text lacks sufficient words/length.
            CorruptedDocumentError: If text has excessive corrupted characters.
            UnsupportedDocumentError: If document type is unsupported.
        """
        # 1. Type and integrity checks
        if extraction_result is None:
            raise MalformedExtractionResultError(
                "Extraction result cannot be None.", document_id="unknown"
            )

        if not isinstance(extraction_result, DocumentExtractionResult):
            raise MalformedExtractionResultError(
                f"Expected DocumentExtractionResult, got "
                f"{type(extraction_result).__name__}.",
                document_id="unknown",
            )

        doc_id = extraction_result.document_id
        if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
            raise MalformedExtractionResultError(
                "Extraction result document_id must be a non-empty string.",
                document_id="unknown",
            )

        if (
            not extraction_result.ticker
            or not isinstance(extraction_result.ticker, str)
            or not extraction_result.ticker.strip()
        ):
            raise MalformedExtractionResultError(
                "Extraction result ticker must be a non-empty string.",
                document_id=doc_id,
            )

        if not isinstance(extraction_result.document_type, DocumentType):
            raise UnsupportedDocumentError(
                f"Document type '{extraction_result.document_type}' is "
                f"unsupported.",
                document_id=doc_id,
            )

        # 2. Structural consistency checks
        pages = extraction_result.pages
        if not isinstance(pages, list) or len(pages) == 0:
            raise MalformedExtractionResultError(
                "Extraction result contains no pages.", document_id=doc_id
            )

        if extraction_result.total_pages <= 0:
            raise MalformedExtractionResultError(
                f"Invalid total_pages count ({extraction_result.total_pages}). "
                f"Must be at least 1.",
                document_id=doc_id,
            )

        if len(pages) != extraction_result.total_pages:
            raise MalformedExtractionResultError(
                f"total_pages ({extraction_result.total_pages}) does not match "
                f"actual page count ({len(pages)}).",
                document_id=doc_id,
            )

        seen_pages = set()
        for page in pages:
            if not isinstance(page, ExtractedPage):
                raise MalformedExtractionResultError(
                    f"Page item is not an ExtractedPage instance: {type(page)}",
                    document_id=doc_id,
                )
            if page.page_number <= 0:
                raise MalformedExtractionResultError(
                    f"Invalid non-positive page number: {page.page_number}",
                    document_id=doc_id,
                )
            if page.page_number in seen_pages:
                raise MalformedExtractionResultError(
                    f"Duplicate page number detected: {page.page_number}",
                    document_id=doc_id,
                )
            seen_pages.add(page.page_number)
            if page.character_count < 0:
                raise MalformedExtractionResultError(
                    f"Negative character count on page {page.page_number}: "
                    f"{page.character_count}",
                    document_id=doc_id,
                )

        # 3. Aggregate text validation
        combined_text = "".join(page.text for page in pages)
        stripped_combined = combined_text.strip()

        if not stripped_combined:
            raise EmptyExtractedTextError(
                f"Document '{doc_id}' contains no extractable text "
                f"(empty or whitespace-only across all {len(pages)} pages).",
                document_id=doc_id,
            )

        # 4. Check for corruption / excessive unprintable characters
        unprintable_count = sum(
            1
            for c in combined_text
            if (ord(c) < 32 and c not in ("\n", "\r", "\t")) or c == "\ufffd"
        )
        total_len = max(len(combined_text), 1)
        unprintable_ratio = unprintable_count / total_len
        if unprintable_ratio > self.max_unprintable_ratio:
            raise CorruptedDocumentError(
                f"Extracted document text contains excessive corrupted or "
                f"unprintable characters ({unprintable_count}/{total_len} = "
                f"{unprintable_ratio:.1%}, exceeds threshold "
                f"{self.max_unprintable_ratio:.1%}).",
                document_id=doc_id,
            )

        # 5. Token / word density and usability check
        words: List[str] = re.findall(r"\b\w+\b", stripped_combined)
        if len(stripped_combined) < self.min_total_characters:
            raise UnusableExtractedTextError(
                f"Document text is too short ({len(stripped_combined)} chars "
                f"< minimum {self.min_total_characters} chars).",
                document_id=doc_id,
            )

        if len(words) < self.min_words:
            raise UnusableExtractedTextError(
                f"Document text contains insufficient words ({len(words)} "
                f"words < minimum {self.min_words} words).",
                document_id=doc_id,
            )

        # 6. Page-level usability evaluation
        validated_pages: List[ValidatedPage] = []
        for page in pages:
            page_clean = page.text.strip()
            page_word_count = len(re.findall(r"\b\w+\b", page_clean))
            page_char_count = len(page.text)
            page_is_usable = bool(
                page_clean
                and page_char_count >= self.min_page_characters
                and page_word_count > 0
            )

            validated_pages.append(
                ValidatedPage(
                    page_number=page.page_number,
                    text=page.text,
                    character_count=page_char_count,
                    word_count=page_word_count,
                    is_usable=page_is_usable,
                )
            )

        usable_page_count = sum(1 for p in validated_pages if p.is_usable)
        if usable_page_count == 0:
            raise UnusableExtractedTextError(
                f"Document '{doc_id}' contains no pages with usable content.",
                document_id=doc_id,
            )

        total_usable_chars = sum(
            p.character_count for p in validated_pages if p.is_usable
        )
        total_usable_words = sum(p.word_count for p in validated_pages if p.is_usable)

        logger.info(
            f"Successfully validated document '{doc_id}' ({extraction_result.ticker}): "
            f"{usable_page_count}/{len(pages)} usable pages, "
            f"{total_usable_chars} chars, {total_usable_words} words."
        )

        return ValidatedDocument(
            document_id=doc_id,
            ticker=extraction_result.ticker,
            document_type=extraction_result.document_type,
            total_pages=extraction_result.total_pages,
            usable_pages=usable_page_count,
            total_characters=total_usable_chars,
            total_words=total_usable_words,
            pages=validated_pages,
            validation_status="valid",
        )


# ===========================================================================
# CONVENIENCE VALIDATION FUNCTIONS (Phase 9.3)
# ===========================================================================


def validate_extracted_document(
    extraction_result: DocumentExtractionResult,
    validator: Optional[ExtractedDocumentValidator] = None,
) -> ValidatedDocument:
    """Validate an existing DocumentExtractionResult.

    Args:
        extraction_result: Extraction result from Phase 9.2.
        validator: Optional custom validator instance.

    Returns:
        ValidatedDocument: Validated document model.
    """
    v = validator or ExtractedDocumentValidator()
    return v.validate(extraction_result)


def validate_document_content(
    content: bytes,
    original_filename: str,
    ticker: str,
    document_type: DocumentType,
    mime_type: Optional[str] = None,
    document_id: Optional[str] = None,
    validator: Optional[ExtractedDocumentValidator] = None,
) -> ValidatedDocument:
    """Extract and validate raw document bytes in an end-to-end pipeline.

    Detects and rejects:
    - Empty content (0 bytes) -> CorruptedDocumentError
    - Unsupported file extensions/MIME -> UnsupportedDocumentError
    - Scanned/image-only PDFs -> EmptyExtractedTextError
    - Corrupted or encrypted PDFs -> CorruptedDocumentError
    - Insufficient or malformed text -> UnusableExtractedTextError

    Args:
        content: Raw binary document bytes.
        original_filename: Original filename of uploaded document.
        ticker: Associated stock ticker symbol.
        document_type: Category of document.
        mime_type: Optional declared MIME type.
        document_id: Optional document identifier.
        validator: Optional custom validator instance.

    Returns:
        ValidatedDocument: Successfully extracted and validated document.
    """
    doc_id = document_id or f"doc_{ticker}_{original_filename}"

    if not content or len(content) == 0:
        raise CorruptedDocumentError(
            "Document content is empty (0 bytes).", document_id=doc_id
        )

    # 1. Resolve extractor
    try:
        extractor = get_document_extractor(
            mime_type=mime_type, filename=original_filename
        )
    except UnsupportedDocumentFormatError as e:
        raise UnsupportedDocumentError(str(e), document_id=doc_id) from e

    # 2. Perform extraction with format-specific error mapping
    try:
        extraction_result = extractor.extract_from_bytes(
            content=content,
            document_id=doc_id,
            ticker=ticker,
            document_type=document_type,
        )
    except ScannedDocumentError as e:
        raise EmptyExtractedTextError(str(e), document_id=doc_id) from e
    except EncryptedDocumentError as e:
        raise CorruptedDocumentError(str(e), document_id=doc_id) from e
    except MalformedDocumentError as e:
        raise CorruptedDocumentError(str(e), document_id=doc_id) from e
    except UnsupportedDocumentFormatError as e:
        raise UnsupportedDocumentError(str(e), document_id=doc_id) from e
    except DocumentExtractionError as e:
        raise CorruptedDocumentError(str(e), document_id=doc_id) from e

    # 3. Run validation
    v = validator or ExtractedDocumentValidator()
    return v.validate(extraction_result)


def validate_stored_document(
    stored_document: StoredDocument,
    storage: DocumentStorage,
    validator: Optional[ExtractedDocumentValidator] = None,
) -> ValidatedDocument:
    """Retrieve, extract, and validate a StoredDocument from storage.

    Args:
        stored_document: StoredDocument metadata object from Phase 9.1.
        storage: DocumentStorage instance where file is stored.
        validator: Optional custom validator instance.

    Returns:
        ValidatedDocument: Validated document model.
    """
    try:
        content = storage.get_bytes(stored_document.storage_path)
    except Exception as e:
        raise CorruptedDocumentError(
            f"Failed to retrieve document file from storage: {e}",
            document_id=stored_document.document_id,
        ) from e

    return validate_document_content(
        content=content,
        original_filename=stored_document.original_filename,
        ticker=stored_document.ticker,
        document_type=stored_document.document_type,
        mime_type=stored_document.mime_type,
        document_id=stored_document.document_id,
        validator=validator,
    )
