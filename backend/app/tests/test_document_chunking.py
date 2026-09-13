"""Unit and integration tests for Phase 9.4: Document Chunking.

Tests cover:
1. Single-page document shorter than chunk size.
2. Single-page document requiring multiple chunks.
3. Multi-page document chunking.
4. Configurable chunk size.
5. Configurable overlap.
6. Correct overlap behavior.
7. No text loss across chunks.
8. No unintended duplication beyond configured overlap.
9. Chunk ordering and index continuity.
10. Chunk IDs are deterministic and unique.
11. Document/ticker/document-type metadata preservation.
12. Page provenance preservation.
13. Multiple-page provenance when chunks span pages (cross_page=True).
14. Blank/non-usable page handling.
15. Invalid chunk configuration rejection.
16. Empty/invalid validated document input rejection.
17. Regression compatibility with Phase 9.1–9.3 storage and validation.
18. Section header detection without LLM fabrication.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.documents import (
    ChunkedDocument,
    DocumentChunk,
    DocumentType,
    StoredDocument,
    ValidatedDocument,
    ValidatedPage,
)
from app.services.document_chunker import (
    ChunkingConfig,
    ChunkingError,
    DocumentChunker,
    EmptyDocumentChunkingError,
    InvalidChunkConfigurationError,
    chunk_document,
)
from app.services.document_text_validator import validate_stored_document
from app.storage.local import LocalDocumentStorage

# ===========================================================================
# HELPERS FOR TEST FIXTURES
# ===========================================================================


def build_validated_doc(
    pages_text: list[str],
    document_id: str = "doc_test_10k",
    ticker: str = "AAPL",
    document_type: DocumentType = DocumentType.ANNUAL_REPORT,
    unusable_page_indices: list[int] = None,
) -> ValidatedDocument:
    """Construct a ValidatedDocument fixture for chunking tests."""
    unusable_set = set(unusable_page_indices or [])
    pages = []
    for idx, text in enumerate(pages_text, start=1):
        is_usable = idx not in unusable_set and bool(text.strip())
        words = len(text.split())
        pages.append(
            ValidatedPage(
                page_number=idx,
                text=text,
                character_count=len(text),
                word_count=words,
                is_usable=is_usable,
            )
        )

    usable_pages = [p for p in pages if p.is_usable]
    total_chars = sum(p.character_count for p in usable_pages)
    total_words = sum(p.word_count for p in usable_pages)

    return ValidatedDocument(
        document_id=document_id,
        ticker=ticker,
        document_type=document_type,
        total_pages=len(pages),
        usable_pages=len(usable_pages),
        total_characters=total_chars,
        total_words=total_words,
        pages=pages,
        validation_status="valid",
        validated_at=datetime.now(timezone.utc),
    )


def build_minimal_pdf(pages: list[str]) -> bytes:
    """Generate a valid binary PDF for end-to-end integration tests."""
    total_pages = len(pages)
    page_obj_nums = [4 + i * 2 + 1 for i in range(total_pages)]
    content_obj_nums = [4 + i * 2 for i in range(total_pages)]

    obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    kids = " ".join(f"{p} 0 R" for p in page_obj_nums)
    obj2 = f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {total_pages} >>\nendobj\n"
    obj3 = "3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    objects = [obj1, obj2, obj3]
    for c_num, p_num, text in zip(content_obj_nums, page_obj_nums, pages):
        stream = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET\n"
        objects.append(
            f"{c_num} 0 obj\n<< /Length {len(stream.encode('ascii'))} >>\n"
            f"stream\n{stream}endstream\nendobj\n"
        )
        objects.append(
            f"{p_num} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {c_num} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n"
        )

    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf.encode("ascii")))
        pdf += obj

    xref_pos = len(pdf.encode("ascii"))
    total_objs = len(objects) + 1
    pdf += f"xref\n0 {total_objs}\n0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n"
    pdf += (
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )
    return pdf.encode("ascii")


# ===========================================================================
# 1. SINGLE-PAGE DOCUMENT CHUNKING (9.4.1)
# ===========================================================================


class TestSinglePageChunking:
    """Tests for single-page documents shorter and longer than chunk size."""

    def test_single_page_shorter_than_chunk_size(self):
        """Single-page text shorter than chunk size produces exactly one chunk."""
        text = (
            "Apple Inc. Q3 FY24 Financial Summary: Total net sales reached "
            "$85.8 billion with diluted EPS of $1.40. Operating cash flow "
            "generated $28.9 billion during the period."
        )
        doc = build_validated_doc([text])
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
        result = chunk_document(doc, config)

        assert isinstance(result, ChunkedDocument)
        assert result.total_chunks == 1
        assert len(result.chunks) == 1

        chunk = result.chunks[0]
        assert isinstance(chunk, DocumentChunk)
        assert chunk.chunk_index == 1
        assert chunk.chunk_id == "doc_test_10k_c0001"
        assert chunk.document_id == "doc_test_10k"
        assert chunk.ticker == "AAPL"
        assert chunk.document_type == DocumentType.ANNUAL_REPORT
        assert chunk.page_numbers == [1]
        assert chunk.start_page == 1
        assert chunk.end_page == 1
        assert chunk.text == text
        assert chunk.character_count == len(text)
        assert chunk.word_count == len(re.findall(r"\b\w+\b", text))

    def test_single_page_requiring_multiple_chunks(self):
        """Single-page text longer than chunk size splits into multiple chunks."""
        paragraphs = [
            f"Paragraph {i}: Net income increased by 12.5% YoY to $25,{i:03d}M "
            f"due to operational efficiencies across regional market segments."
            for i in range(1, 15)
        ]
        text = "\n\n".join(paragraphs)
        doc = build_validated_doc([text])

        config = ChunkingConfig(chunk_size=400, chunk_overlap=100)
        result = chunk_document(doc, config)

        assert result.total_chunks > 1
        assert len(result.chunks) == result.total_chunks

        for i, chunk in enumerate(result.chunks, start=1):
            assert chunk.chunk_index == i
            assert chunk.chunk_id == f"doc_test_10k_c{i:04d}"
            assert chunk.page_numbers == [1]
            assert chunk.character_count > 0


# ===========================================================================
# 2. CONFIGURABLE CHUNK SIZE AND OVERLAP (9.4.1)
# ===========================================================================


class TestConfigurableChunkParameters:
    """Tests verifying configurable chunk size and overlap behaviors."""

    def test_varying_chunk_size_changes_chunk_count(self):
        """Increasing chunk size reduces total chunk count."""
        paragraphs = [
            f"Financial disclosure paragraph {i}: Detailed breakdown of capital "
            f"expenditures totaling $5,{i}00M in hardware infrastructure."
            for i in range(1, 20)
        ]
        text = "\n\n".join(paragraphs)
        doc = build_validated_doc([text])

        result_small = chunk_document(
            doc, ChunkingConfig(chunk_size=300, chunk_overlap=50)
        )
        result_large = chunk_document(
            doc, ChunkingConfig(chunk_size=1000, chunk_overlap=50)
        )

        assert result_small.total_chunks > result_large.total_chunks
        assert result_small.chunk_size == 300
        assert result_large.chunk_size == 1000

    def test_correct_overlap_between_adjacent_chunks(self):
        """Consecutive chunks share expected overlapping text substring."""
        p1 = "Operating segment results: Hardware revenue was $50.0B."
        p2 = "Cloud infrastructure revenue reached $35.0B, up 22% YoY."
        p3 = "Operating margin expanded by 150 basis points to 31.5%."
        text = f"{p1}\n\n{p2}\n\n{p3}"

        doc = build_validated_doc([text])
        config = ChunkingConfig(chunk_size=90, chunk_overlap=30)
        result = chunk_document(doc, config)

        assert len(result.chunks) >= 2
        for idx in range(len(result.chunks) - 1):
            c_curr = result.chunks[idx]
            c_next = result.chunks[idx + 1]

            # Find common overlapping content: end of c_curr and start of c_next
            tail_curr = c_curr.text[-20:]
            assert (
                tail_curr in c_next.text
            ), f"Overlap missing between chunk {idx+1} and {idx+2}"

    def test_no_text_loss_across_chunks(self):
        """All unique sentences from source text are preserved across chunks."""
        sentences = [
            "Revenue reached $100B in 2024.",
            "Operating expenses rose 5% to $40B.",
            "Net income came in at $25B.",
            "Free cash flow exceeded $20B.",
            "Dividend payout ratio remained at 30%.",
        ]
        text = " ".join(sentences)
        doc = build_validated_doc([text])

        config = ChunkingConfig(chunk_size=80, chunk_overlap=25)
        result = chunk_document(doc, config)

        combined_chunk_text = " ".join(c.text for c in result.chunks)
        for s in sentences:
            assert s in combined_chunk_text, f"Sentence '{s}' was lost!"

    def test_no_unintended_excessive_duplication(self):
        """Chunks do not duplicate content beyond the configured overlap limit."""
        text = (
            "Section Alpha: General market conditions. " * 5
            + "Section Beta: Performance metrics. " * 5
        )
        doc = build_validated_doc([text])
        config = ChunkingConfig(chunk_size=200, chunk_overlap=40)
        result = chunk_document(doc, config)

        # Verify that each chunk advances forward
        for idx in range(len(result.chunks) - 1):
            assert result.chunks[idx].text != result.chunks[idx + 1].text


# ===========================================================================
# 2B. CHUNK ORDERING & METADATA PRESERVATION (9.4.1 & 9.4.2)
# ===========================================================================


class TestChunkOrderingAndMetadata:
    """Tests verifying sequential chunk ordering, deterministic IDs, and metadata."""

    def test_chunk_ordering_strictly_monotonic(self):
        """Chunk indices are 1-indexed, strictly monotonic, and consecutive."""
        paragraphs = [
            f"Paragraph {i}: Content for ordering verification." for i in range(1, 10)
        ]
        doc = build_validated_doc(["\n\n".join(paragraphs)])
        config = ChunkingConfig(chunk_size=120, chunk_overlap=30)
        result = chunk_document(doc, config)

        expected_indices = list(range(1, len(result.chunks) + 1))
        actual_indices = [c.chunk_index for c in result.chunks]
        assert actual_indices == expected_indices

    def test_deterministic_and_unique_chunk_ids(self):
        """Chunk IDs follow {document_id}_c{index:04d}, are unique and reproducible."""
        paragraphs = [
            f"Data block {i}: Revenue and EBITDA metrics." for i in range(1, 8)
        ]
        doc = build_validated_doc(
            ["\n\n".join(paragraphs)], document_id="doc_reproducible"
        )
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)

        result_1 = chunk_document(doc, config)
        result_2 = chunk_document(doc, config)

        ids_1 = [c.chunk_id for c in result_1.chunks]
        ids_2 = [c.chunk_id for c in result_2.chunks]

        # Uniqueness
        assert len(ids_1) == len(set(ids_1))
        # Reproducibility
        assert ids_1 == ids_2
        # Deterministic formatting
        for idx, cid in enumerate(ids_1, start=1):
            assert cid == f"doc_reproducible_c{idx:04d}"

    def test_document_metadata_preservation_across_all_chunks(self):
        """All chunks retain document_id, ticker, and document_type faithfully."""
        doc = build_validated_doc(
            ["Paragraph 1: Alpha\n\nParagraph 2: Beta\n\nParagraph 3: Gamma"],
            document_id="doc_meta_check",
            ticker="NVDA",
            document_type=DocumentType.INVESTOR_PRESENTATION,
        )
        config = ChunkingConfig(
            chunk_size=40, chunk_overlap=10, min_chunk_characters=20
        )
        result = chunk_document(doc, config)

        assert result.total_chunks > 1
        for chunk in result.chunks:
            assert chunk.document_id == "doc_meta_check"
            assert chunk.ticker == "NVDA"
            assert chunk.document_type == DocumentType.INVESTOR_PRESENTATION
            assert isinstance(chunk.created_at, datetime)


# ===========================================================================
# 3. MULTI-PAGE & PROVENANCE PRESERVATION (9.4.2)
# ===========================================================================


class TestMultiPageProvenance:
    """Tests preserving page and section provenance across multiple pages."""

    def test_multi_page_document_strict_page_boundaries(self):
        """Default cross_page=False isolates chunks strictly to their source page."""
        p1_text = "Page 1: Microsoft FY24 Executive Overview and Strategic Milestones."
        p2_text = "Page 2: Intelligent Cloud Segment Revenue: $105.4 billion."
        p3_text = "Page 3: More Personal Computing Revenue: $54.7 billion."

        doc = build_validated_doc([p1_text, p2_text, p3_text], ticker="MSFT")
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=200, cross_page=False)
        result = chunk_document(doc, config)

        assert result.total_chunks == 3
        assert result.chunks[0].page_numbers == [1]
        assert result.chunks[0].start_page == 1
        assert result.chunks[0].end_page == 1
        assert "Page 1" in result.chunks[0].text

        assert result.chunks[1].page_numbers == [2]
        assert result.chunks[1].start_page == 2
        assert result.chunks[1].end_page == 2
        assert "Page 2" in result.chunks[1].text

        assert result.chunks[2].page_numbers == [3]
        assert result.chunks[2].start_page == 3
        assert result.chunks[2].end_page == 3
        assert "Page 3" in result.chunks[2].text

    def test_multi_page_document_cross_page_provenance(self):
        """cross_page=True tracks multiple-page provenance when a chunk spans pages."""
        p1_text = "Short Page 1 content."
        p2_text = "Short Page 2 content."

        doc = build_validated_doc([p1_text, p2_text], ticker="GOOGL")
        # With chunk_size=500, both pages easily fit into 1 chunk
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50, cross_page=True)
        result = chunk_document(doc, config)

        assert result.total_chunks == 1
        chunk = result.chunks[0]
        assert chunk.page_numbers == [1, 2]
        assert chunk.start_page == 1
        assert chunk.end_page == 2
        assert "Page 1" in chunk.text
        assert "Page 2" in chunk.text

    def test_blank_non_usable_pages_are_omitted(self):
        """Blank or divider pages marked is_usable=False generate no chunks."""
        p1 = "First page: Executive summary and highlights."
        p2 = ""  # Blank divider page
        p3 = "Third page: Consolidated statements of income."

        doc = build_validated_doc(
            [p1, p2, p3], unusable_page_indices=[2], ticker="AMZN"
        )
        config = ChunkingConfig(cross_page=False)
        result = chunk_document(doc, config)

        assert result.total_chunks == 2
        assert result.chunks[0].page_numbers == [1]
        assert result.chunks[1].page_numbers == [3]
        for c in result.chunks:
            assert 2 not in c.page_numbers


# ===========================================================================
# 4. SECTION HEADING DETECTION (9.4.2)
# ===========================================================================


class TestSectionDetection:
    """Tests detecting standard financial report headings without LLM guessing."""

    def test_standard_item_header_detected(self):
        """Standard SEC 'ITEM 1A. RISK FACTORS' is detected deterministically."""
        text = (
            "ITEM 1A. RISK FACTORS\n"
            "Our operations are subject to cybersecurity vulnerabilities and "
            "global supply chain disruptions."
        )
        doc = build_validated_doc([text])
        result = chunk_document(doc)

        assert len(result.chunks) == 1
        assert result.chunks[0].section_name == "ITEM 1A. RISK FACTORS"

    def test_consolidated_statements_header_detected(self):
        """Standard 'CONSOLIDATED STATEMENTS OF OPERATIONS' is detected."""
        text = (
            "CONSOLIDATED STATEMENTS OF OPERATIONS\n"
            "Total net sales: $383,285 million\n"
            "Cost of sales: $210,352 million"
        )
        doc = build_validated_doc([text])
        result = chunk_document(doc)

        assert result.chunks[0].section_name == "CONSOLIDATED STATEMENTS OF OPERATIONS"

    def test_arbitrary_text_has_none_section_name(self):
        """Arbitrary narrative text does not invent or fabricate section names."""
        text = (
            "The company continued its expansion into European renewable "
            "energy markets during the fiscal third quarter."
        )
        doc = build_validated_doc([text])
        result = chunk_document(doc)

        assert result.chunks[0].section_name is None


# ===========================================================================
# 5. ERROR HANDLING AND VALIDATION (9.4)
# ===========================================================================


class TestChunkingErrorHandling:
    """Tests verifying error handling for invalid configs and invalid inputs."""

    def test_overlap_greater_than_or_equal_to_chunk_size_raises_error(self):
        """chunk_overlap >= chunk_size raises InvalidChunkConfigurationError."""
        with pytest.raises(
            InvalidChunkConfigurationError,
            match="strictly less than chunk_size",
        ):
            ChunkingConfig(chunk_size=200, chunk_overlap=200)

        with pytest.raises(
            InvalidChunkConfigurationError,
            match="strictly less than chunk_size",
        ):
            ChunkingConfig(chunk_size=200, chunk_overlap=250)

    def test_negative_or_zero_chunk_size_raises_error(self):
        """chunk_size <= 0 raises InvalidChunkConfigurationError."""
        with pytest.raises(
            InvalidChunkConfigurationError, match="chunk_size must be positive"
        ):
            ChunkingConfig(chunk_size=0, chunk_overlap=0)

        with pytest.raises(
            InvalidChunkConfigurationError, match="chunk_size must be positive"
        ):
            ChunkingConfig(chunk_size=-100, chunk_overlap=0)

    def test_negative_overlap_raises_error(self):
        """Negative chunk_overlap raises InvalidChunkConfigurationError."""
        with pytest.raises(
            InvalidChunkConfigurationError,
            match="chunk_overlap must be non-negative",
        ):
            ChunkingConfig(chunk_size=500, chunk_overlap=-10)

    def test_none_document_raises_empty_document_error(self):
        """Passing None to chunker raises EmptyDocumentChunkingError."""
        chunker = DocumentChunker()
        with pytest.raises(EmptyDocumentChunkingError, match="cannot be None"):
            chunker.chunk(None)

    def test_invalid_document_type_raises_empty_document_error(self):
        """Passing non-ValidatedDocument object raises EmptyDocumentChunkingError."""
        chunker = DocumentChunker()
        with pytest.raises(
            EmptyDocumentChunkingError, match="Expected ValidatedDocument"
        ):
            chunker.chunk({"document_id": "test"})

    def test_document_with_no_usable_pages_raises_empty_error(self):
        """ValidatedDocument with 0 usable pages raises EmptyDocumentChunkingError."""
        doc = build_validated_doc(["", ""], unusable_page_indices=[1, 2])
        chunker = DocumentChunker()
        with pytest.raises(EmptyDocumentChunkingError, match="no usable pages"):
            chunker.chunk(doc)

    def test_exception_inheritance(self):
        """Chunking exceptions inherit from ChunkingError."""
        assert issubclass(InvalidChunkConfigurationError, ChunkingError)
        assert issubclass(EmptyDocumentChunkingError, ChunkingError)


# ===========================================================================
# 6. END-TO-END REGRESSION COMPATIBILITY (9.1 - 9.4)
# ===========================================================================


class TestEndToEndChunkingPipeline:
    """End-to-end integration test: LocalDocumentStorage -> Validate -> Chunk."""

    def test_storage_to_validated_to_chunked_pipeline(self, tmp_path: Path):
        """Full pipeline from storage to validated document to structured chunks."""
        storage = LocalDocumentStorage(base_directory=tmp_path)
        pdf_bytes = build_minimal_pdf(
            [
                (
                    "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS\n"
                    "Amazon Web Services delivered strong revenue of $26.3B, "
                    "reflecting accelerating cloud adoption and enterprise migration."
                ),
                (
                    "CONSOLIDATED BALANCE SHEETS\n"
                    "Cash and cash equivalents totaled $45.2 billion as of "
                    "June 30, 2024, demonstrating liquidity resilience."
                ),
            ]
        )

        # 1. Store document (Phase 9.1)
        stored_doc = storage.save(
            content=pdf_bytes,
            original_filename="amzn_10q.pdf",
            ticker="AMZN",
            document_type=DocumentType.COMPANY_REPORT,
        )
        assert isinstance(stored_doc, StoredDocument)

        # 2. Validate document (Phase 9.3 using Phase 9.2 extraction)
        validated_doc = validate_stored_document(stored_doc, storage)
        assert isinstance(validated_doc, ValidatedDocument)
        assert validated_doc.usable_pages == 2

        # 3. Chunk document (Phase 9.4)
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
        chunked = chunk_document(validated_doc, config)

        assert isinstance(chunked, ChunkedDocument)
        assert chunked.document_id == stored_doc.document_id
        assert chunked.ticker == "AMZN"
        assert chunked.document_type == DocumentType.COMPANY_REPORT
        assert chunked.total_chunks == 2

        c1 = chunked.chunks[0]
        assert c1.chunk_index == 1
        assert c1.start_page == 1
        assert c1.end_page == 1
        assert c1.section_name == "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS"
        assert "Amazon Web Services" in c1.text

        c2 = chunked.chunks[1]
        assert c2.chunk_index == 2
        assert c2.start_page == 2
        assert c2.end_page == 2
        assert c2.section_name == "CONSOLIDATED BALANCE SHEETS"
        assert "Cash and cash equivalents" in c2.text
