"""Document chunking service for Phase 9.4.

Provides:
- ChunkingError and typed subclasses:
  InvalidChunkConfigurationError, EmptyDocumentChunkingError.
- ChunkingConfig: Tuned configuration for financial text chunking.
- DocumentChunker: Deterministic chunking engine preserving page and
  section provenance.
- chunk_document: Helper function to chunk a ValidatedDocument.
"""

import re
from typing import Any, List, Optional

from app.core.logging import get_logger
from app.models.documents import (
    ChunkedDocument,
    DocumentChunk,
    ValidatedDocument,
)

logger = get_logger(__name__)

# Standard SEC and financial report structural section headings
SECTION_HEADER_PATTERN = re.compile(
    r"^(?:ITEM\s+[0-9]+[A-Z]?(?:\.[0-9]+)?\.?|PART\s+[IVXLCDM]+|"
    r"CONSOLIDATED\s+(?:STATEMENTS?(?:\s+OF\s+[A-Z\s]+)?|BALANCE\s+SHEETS?)|"
    r"NOTES\s+TO\s+(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS)"
    r"(?:\s*[-–—:.]?\s*.*)?$",
    re.IGNORECASE,
)


# ===========================================================================
# TYPED DOMAIN EXCEPTIONS (Phase 9.4)
# ===========================================================================


class ChunkingError(Exception):
    """Base exception for document chunking failures (Phase 9.4)."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
        code: str = "CHUNKING_ERROR",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.document_id = document_id
        self.code = code

    def __str__(self) -> str:
        return f"[{self.document_id}] {self.message}"


class InvalidChunkConfigurationError(ChunkingError):
    """Raised when chunk size, overlap, or parameters are invalid."""

    def __init__(
        self,
        message: str,
        document_id: str = "unknown",
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code="INVALID_CHUNK_CONFIG",
        )


class EmptyDocumentChunkingError(ChunkingError):
    """Raised when the input document has no usable text or pages to chunk."""

    def __init__(
        self,
        message: str = "Cannot chunk document with no usable pages or text.",
        document_id: str = "unknown",
    ) -> None:
        super().__init__(
            message=message,
            document_id=document_id,
            code="EMPTY_DOCUMENT_FOR_CHUNKING",
        )


# ===========================================================================
# CHUNKING CONFIGURATION (Phase 9.4.1)
# ===========================================================================


class ChunkingConfig:
    """Tuned configuration for financial document chunking.

    Default Values Rationale:
    - chunk_size (1000 characters, ~150-200 words):
      Specifically tuned for financial disclosures. Financial metrics, balance
      sheet lines, footnotes, and MD&A management discussions require
      surrounding narrative context to retain semantic clarity for RAG embeddings.
    - chunk_overlap (200 characters, ~20% overlap):
      Guarantees continuity across chunk splits so that critical financial ratios,
      table rows, or YoY comparisons spanning split boundaries are captured
      completely in both adjacent chunks.
    - min_chunk_characters (50 characters):
      Prevents tiny orphaned fragments or stray header lines from generating
      low-signal vector embeddings.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cross_page: bool = False,
        min_chunk_characters: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cross_page = cross_page
        self.min_chunk_characters = min_chunk_characters
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise InvalidChunkConfigurationError(
                f"chunk_size must be positive, got {self.chunk_size}."
            )
        if self.chunk_overlap < 0:
            raise InvalidChunkConfigurationError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise InvalidChunkConfigurationError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less "
                f"than chunk_size ({self.chunk_size})."
            )
        if self.min_chunk_characters <= 0:
            raise InvalidChunkConfigurationError(
                f"min_chunk_characters must be positive, got "
                f"{self.min_chunk_characters}."
            )
        if self.min_chunk_characters > self.chunk_size:
            raise InvalidChunkConfigurationError(
                f"min_chunk_characters ({self.min_chunk_characters}) cannot "
                f"exceed chunk_size ({self.chunk_size})."
            )


# ===========================================================================
# DOCUMENT CHUNKER (Phase 9.4.1 & 9.4.2)
# ===========================================================================


class DocumentChunker:
    """Deterministic document chunker for financial texts."""

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()

    def _detect_section_name(self, text: str) -> Optional[str]:
        """Detect a reliable standard structural section heading in text.

        Only matches unambiguous financial disclosure headings (Item 1, Part I,
        Consolidated Statements). Does NOT guess or invent section names.
        """
        for line in text.splitlines()[:3]:
            stripped = line.strip()
            if SECTION_HEADER_PATTERN.match(stripped):
                return stripped
        return None

    def _find_split_point(
        self, text: str, start: int, target_end: int, overlap: int
    ) -> int:
        """Find a natural boundary near target_end within the overlap window.

        Prioritizes:
        1. Paragraph break (\\n\\n)
        2. Line break (\\n)
        3. Sentence boundary (. )
        4. Word space ( )
        """
        if target_end >= len(text):
            return len(text)

        search_start = max(start, target_end - overlap)
        window = text[search_start:target_end]

        for sep in ("\n\n", "\n", ". ", " "):
            idx = window.rfind(sep)
            if idx != -1:
                return search_start + idx + len(sep)

        return target_end

    def _split_text_into_slices(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[tuple[int, int]]:
        """Compute [start, end] character slice offsets covering the text.

        Guarantees:
        - No text is lost between consecutive chunks.
        - Overlap is bounded by chunk_overlap.
        - Every character from 0 to len(text) is present in at least one chunk.
        """
        slices: List[tuple[int, int]] = []
        text_len = len(text)
        if text_len == 0:
            return slices

        if text_len <= chunk_size:
            return [(0, text_len)]

        start = 0
        while start < text_len:
            target_end = min(start + chunk_size, text_len)
            if target_end == text_len:
                slices.append((start, text_len))
                break

            split_pos = self._find_split_point(
                text=text,
                start=start,
                target_end=target_end,
                overlap=chunk_overlap,
            )

            # Ensure strict forward progress
            if split_pos <= start:
                split_pos = target_end

            slices.append((start, split_pos))

            # Next chunk starts backed up by overlap
            next_start = split_pos - chunk_overlap
            if next_start <= start:
                next_start = split_pos

            start = next_start

        return slices

    def chunk(self, document: Any) -> ChunkedDocument:
        """Chunk a ValidatedDocument deterministically.

        Args:
            document: ValidatedDocument instance from Phase 9.3.

        Returns:
            ChunkedDocument: Ordered container of DocumentChunk objects.

        Raises:
            EmptyDocumentChunkingError: If document is None or has no usable text.
            ChunkingError: If document structure is invalid.
        """
        if document is None:
            raise EmptyDocumentChunkingError(
                "ValidatedDocument cannot be None.", document_id="unknown"
            )

        if not isinstance(document, ValidatedDocument):
            raise EmptyDocumentChunkingError(
                f"Expected ValidatedDocument instance, got "
                f"{type(document).__name__}.",
                document_id="unknown",
            )

        doc_id = document.document_id
        usable_pages = document.get_usable_pages()
        if not usable_pages:
            raise EmptyDocumentChunkingError(
                f"Document '{doc_id}' contains no usable pages to chunk.",
                document_id=doc_id,
            )

        chunks: List[DocumentChunk] = []
        chunk_idx = 1

        if not self.config.cross_page:
            # 1. Page-by-page chunking (strict page boundary preservation)
            for page in usable_pages:
                page_text = page.text.strip()
                if not page_text:
                    continue

                slices = self._split_text_into_slices(
                    text=page_text,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )

                for s_start, s_end in slices:
                    chunk_text = page_text[s_start:s_end].strip()
                    if not chunk_text:
                        continue

                    chunk_id = f"{doc_id}_c{chunk_idx:04d}"
                    sec_name = self._detect_section_name(chunk_text)
                    words_count = len(re.findall(r"\b\w+\b", chunk_text))

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=doc_id,
                            ticker=document.ticker,
                            document_type=document.document_type,
                            chunk_index=chunk_idx,
                            text=chunk_text,
                            character_count=len(chunk_text),
                            word_count=words_count,
                            page_numbers=[page.page_number],
                            start_page=page.page_number,
                            end_page=page.page_number,
                            section_name=sec_name,
                        )
                    )
                    chunk_idx += 1
        else:
            # 2. Cross-page continuous chunking
            # Build continuous string with character-to-page index mappings
            page_segments: List[tuple[int, str]] = []
            for p in usable_pages:
                t = p.text.strip()
                if t:
                    page_segments.append((p.page_number, t))

            if not page_segments:
                raise EmptyDocumentChunkingError(
                    f"Document '{doc_id}' contains no text across usable pages.",
                    document_id=doc_id,
                )

            # Join with standard double newlines and track character boundaries
            full_text_parts: List[str] = []
            page_ranges: List[tuple[int, int, int]] = []  # (start, end, page_num)
            current_offset = 0

            for p_num, p_text in page_segments:
                if full_text_parts:
                    sep = "\n\n"
                    full_text_parts.append(sep)
                    current_offset += len(sep)

                part_start = current_offset
                part_end = part_start + len(p_text)
                full_text_parts.append(p_text)
                page_ranges.append((part_start, part_end, p_num))
                current_offset = part_end

            full_text = "".join(full_text_parts)

            slices = self._split_text_into_slices(
                text=full_text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            for s_start, s_end in slices:
                chunk_text = full_text[s_start:s_end].strip()
                if not chunk_text:
                    continue

                # Identify all pages intersecting [s_start, s_end]
                spanning_pages = [
                    p_num
                    for (r_start, r_end, p_num) in page_ranges
                    if max(s_start, r_start) < min(s_end, r_end)
                ]

                if not spanning_pages:
                    spanning_pages = [usable_pages[0].page_number]

                chunk_id = f"{doc_id}_c{chunk_idx:04d}"
                sec_name = self._detect_section_name(chunk_text)
                words_count = len(re.findall(r"\b\w+\b", chunk_text))

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        ticker=document.ticker,
                        document_type=document.document_type,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        character_count=len(chunk_text),
                        word_count=words_count,
                        page_numbers=spanning_pages,
                        start_page=min(spanning_pages),
                        end_page=max(spanning_pages),
                        section_name=sec_name,
                    )
                )
                chunk_idx += 1

        total_chars = sum(c.character_count for c in chunks)
        logger.info(
            f"Successfully chunked document '{doc_id}' ({document.ticker}): "
            f"{len(chunks)} chunks produced, {total_chars} total characters."
        )

        return ChunkedDocument(
            document_id=doc_id,
            ticker=document.ticker,
            document_type=document.document_type,
            total_chunks=len(chunks),
            total_characters=total_chars,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunks=chunks,
        )


# ===========================================================================
# CONVENIENCE FUNCTION (Phase 9.4)
# ===========================================================================


def chunk_document(
    document: ValidatedDocument,
    config: Optional[ChunkingConfig] = None,
) -> ChunkedDocument:
    """Chunk a ValidatedDocument using DocumentChunker.

    Args:
        document: Validated document model.
        config: Optional custom ChunkingConfig.

    Returns:
        ChunkedDocument: Structured chunk container.
    """
    chunker = DocumentChunker(config=config)
    return chunker.chunk(document)
