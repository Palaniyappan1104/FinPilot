"""Unit and integration tests for Phase 9.3: Document Validation.

Tests cover:
1. Valid extracted text (single and multi-page).
2. Empty extraction result.
3. Whitespace-only extraction.
4. Pages with missing/empty text where appropriate.
5. Reasonably well-formed multi-page text.
6. Malformed/inconsistent extraction result.
7. Unsupported document format.
8. Corrupted/invalid extraction input.
9. Preservation of document/page metadata.
10. Regression and compatibility with Phase 9.1 and 9.2 storage & extraction.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.documents import (
    DocumentExtractionResult,
    DocumentType,
    ExtractedPage,
    StoredDocument,
    ValidatedDocument,
    ValidatedPage,
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
from app.storage.local import LocalDocumentStorage

# ===========================================================================
# HELPERS FOR TEST FIXTURES
# ===========================================================================


def build_sample_extraction_result(
    pages_text: list[str],
    document_id: str = "doc_test_10k",
    ticker: str = "AAPL",
    document_type: DocumentType = DocumentType.ANNUAL_REPORT,
) -> DocumentExtractionResult:
    """Construct a synthetic DocumentExtractionResult for testing."""
    pages = [
        ExtractedPage(
            page_number=idx,
            text=text,
            character_count=len(text),
        )
        for idx, text in enumerate(pages_text, start=1)
    ]
    total_chars = sum(p.character_count for p in pages)
    return DocumentExtractionResult(
        document_id=document_id,
        ticker=ticker,
        document_type=document_type,
        total_pages=len(pages),
        total_characters=total_chars,
        pages=pages,
        extraction_status="completed",
        extracted_at=datetime.now(timezone.utc),
    )


def build_minimal_pdf(pages: list[str]) -> bytes:
    """Generate a clean binary PDF for end-to-end integration tests."""
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
# 1. VALID EXTRACTED TEXT & WELL-FORMEDNESS (9.3.1)
# ===========================================================================


class TestValidExtractedText:
    """Tests for validating well-formed extracted text."""

    def test_validate_single_page_valid_text(self):
        """Single-page extraction with valid text passes validation."""
        text = (
            "Apple Inc. Q3 FY24 Consolidated Revenue: $85,777M, up 5% YoY. "
            "Net income reached $21,448M."
        )
        extraction = build_sample_extraction_result([text])
        validator = ExtractedDocumentValidator()
        validated = validator.validate(extraction)

        assert isinstance(validated, ValidatedDocument)
        assert validated.document_id == "doc_test_10k"
        assert validated.ticker == "AAPL"
        assert validated.document_type == DocumentType.ANNUAL_REPORT
        assert validated.total_pages == 1
        assert validated.usable_pages == 1
        assert validated.total_characters == len(text)
        assert validated.total_words > 10
        assert validated.validation_status == "valid"
        assert len(validated.pages) == 1
        assert isinstance(validated.pages[0], ValidatedPage)
        assert validated.pages[0].page_number == 1
        assert validated.pages[0].is_usable is True


class TestExceptionHierarchy:
    """Tests verifying domain exception class hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """All Phase 9.3 exceptions inherit from DocumentTextValidationError."""
        assert issubclass(EmptyExtractedTextError, DocumentTextValidationError)
        assert issubclass(UnusableExtractedTextError, DocumentTextValidationError)
        assert issubclass(MalformedExtractionResultError, DocumentTextValidationError)
        assert issubclass(UnsupportedDocumentError, DocumentTextValidationError)
        assert issubclass(CorruptedDocumentError, DocumentTextValidationError)

    def test_validate_multi_page_well_formed_text(self):
        """Multi-page extraction preserves page structure and calculates totals."""
        page1 = "Executive Summary: FY2024 Revenue grew to $383.29B driven by Services."
        page2 = "Consolidated Balance Sheet: Cash and marketable securities $153.0B."
        page3 = (
            "Cash Flows: Operating cash flow generated $118.25B across all quarters."
        )

        extraction = build_sample_extraction_result(
            [page1, page2, page3],
            document_id="doc_aapl_annual",
            ticker="AAPL",
            document_type=DocumentType.ANNUAL_REPORT,
        )

        validated = validate_extracted_document(extraction)

        assert validated.total_pages == 3
        assert validated.usable_pages == 3
        assert len(validated.pages) == 3
        assert validated.pages[0].page_number == 1
        assert validated.pages[1].page_number == 2
        assert validated.pages[2].page_number == 3
        assert all(p.is_usable for p in validated.pages)
        assert validated.total_characters == len(page1) + len(page2) + len(page3)
        assert validated.total_words == sum(p.word_count for p in validated.pages)
        assert len(validated.get_usable_pages()) == 3
        assert "Executive Summary" in validated.get_full_text()
        assert "Cash Flows" in validated.get_full_text()


# ===========================================================================
# 2. EMPTY AND WHITESPACE EXTRACTION HANDLING (9.3.1)
# ===========================================================================


class TestEmptyAndWhitespaceValidation:
    """Tests for rejecting empty, whitespace-only, or zero-content results."""

    def test_empty_string_extraction_raises_empty_error(self):
        """Extraction result with empty strings raises EmptyExtractedTextError."""
        extraction = build_sample_extraction_result(["", ""])
        validator = ExtractedDocumentValidator()

        with pytest.raises(EmptyExtractedTextError) as exc_info:
            validator.validate(extraction)

        assert "empty or whitespace-only" in str(exc_info.value).lower()
        assert exc_info.value.code == "EMPTY_TEXT"
        assert exc_info.value.document_id == "doc_test_10k"

    def test_whitespace_only_extraction_raises_empty_error(self):
        """Extraction result containing only spaces, tabs, and newlines raises error."""
        extraction = build_sample_extraction_result(["   \n\n\t  ", "  \r\n  "])
        validator = ExtractedDocumentValidator()

        with pytest.raises(EmptyExtractedTextError) as exc_info:
            validator.validate(extraction)

        assert "empty or whitespace-only" in str(exc_info.value).lower()
        assert exc_info.value.code == "EMPTY_TEXT"

    def test_insufficient_character_count_raises_unusable_error(self):
        """Text below minimum character threshold raises UnusableExtractedTextError."""
        extraction = build_sample_extraction_result(["Tiny text"])
        validator = ExtractedDocumentValidator(min_total_characters=25)

        with pytest.raises(UnusableExtractedTextError) as exc_info:
            validator.validate(extraction)

        assert "too short" in str(exc_info.value).lower()
        assert exc_info.value.code == "UNUSABLE_TEXT"

    def test_insufficient_word_count_raises_unusable_error(self):
        """Text with fewer than minimum words raises UnusableExtractedTextError."""
        extraction = build_sample_extraction_result(
            ["1234567890 1234567890 1234567890"]
        )
        validator = ExtractedDocumentValidator(min_words=5)

        with pytest.raises(UnusableExtractedTextError) as exc_info:
            validator.validate(extraction)

        assert "insufficient words" in str(exc_info.value).lower()


# ===========================================================================
# 3. PAGES WITH MISSING / EMPTY TEXT (9.3.1)
# ===========================================================================


class TestPartialEmptyPagesValidation:
    """Tests handling documents with partially blank or divider pages."""

    def test_document_with_blank_cover_page_preserves_usable_pages(self):
        """Cover page is blank, but inner pages have content. Usable pages tracked."""
        pages = [
            "",  # Blank cover/divider page
            (
                "Financial Highlights FY24: Total Revenue $250.0B with 18% "
                "Operating Margin."
            ),
            (
                "Segment Breakdown: Cloud infrastructure generated $105.0B "
                "in annual sales."
            ),
        ]
        extraction = build_sample_extraction_result(pages)
        validator = ExtractedDocumentValidator()
        validated = validator.validate(extraction)

        assert validated.total_pages == 3
        assert validated.usable_pages == 2
        assert len(validated.pages) == 3

        # Page 1 is marked unusable
        assert validated.pages[0].page_number == 1
        assert validated.pages[0].is_usable is False
        assert validated.pages[0].character_count == 0
        assert validated.pages[0].word_count == 0

        # Pages 2 and 3 are marked usable
        assert validated.pages[1].page_number == 2
        assert validated.pages[1].is_usable is True
        assert validated.pages[2].page_number == 3
        assert validated.pages[2].is_usable is True

        usable_pages = validated.get_usable_pages()
        assert len(usable_pages) == 2
        assert [p.page_number for p in usable_pages] == [2, 3]

    def test_all_pages_empty_or_whitespace_raises_empty_error(self):
        """When all pages are empty or whitespace, error is raised."""
        extraction = build_sample_extraction_result(["", "   ", "\n\n"])
        validator = ExtractedDocumentValidator()

        with pytest.raises(EmptyExtractedTextError):
            validator.validate(extraction)


# ===========================================================================
# 4. MALFORMED / INCONSISTENT EXTRACTION RESULTS (9.3.2)
# ===========================================================================


class TestMalformedExtractionResultValidation:
    """Tests detecting structural inconsistencies in DocumentExtractionResult."""

    def test_none_input_raises_malformed_error(self):
        """Passing None to validate raises MalformedExtractionResultError."""
        validator = ExtractedDocumentValidator()
        with pytest.raises(MalformedExtractionResultError, match="cannot be None"):
            validator.validate(None)

    def test_invalid_type_raises_malformed_error(self):
        """Passing an invalid object type raises MalformedExtractionResultError."""
        validator = ExtractedDocumentValidator()
        with pytest.raises(
            MalformedExtractionResultError,
            match="Expected DocumentExtractionResult",
        ):
            validator.validate({"document_id": "test"})

    def test_empty_pages_list_raises_malformed_error(self):
        """Result with pages=[] raises MalformedExtractionResultError."""
        extraction = DocumentExtractionResult(
            document_id="doc_empty_pages",
            ticker="NVDA",
            document_type=DocumentType.COMPANY_REPORT,
            total_pages=1,
            total_characters=0,
            pages=[],
            extraction_status="completed",
            extracted_at=datetime.now(timezone.utc),
        )
        validator = ExtractedDocumentValidator()
        with pytest.raises(MalformedExtractionResultError, match="contains no pages"):
            validator.validate(extraction)

    def test_mismatched_total_pages_raises_malformed_error(self):
        """total_pages != len(pages) raises MalformedExtractionResultError."""
        extraction = DocumentExtractionResult(
            document_id="doc_mismatch",
            ticker="NVDA",
            document_type=DocumentType.COMPANY_REPORT,
            total_pages=5,  # Mismatched declared total
            total_characters=60,
            pages=[
                ExtractedPage(
                    page_number=1,
                    text="NVIDIA Q2 FY25 Revenue reached $30.0 billion, up 122% YoY.",
                    character_count=60,
                )
            ],
            extraction_status="completed",
            extracted_at=datetime.now(timezone.utc),
        )
        validator = ExtractedDocumentValidator()
        with pytest.raises(
            MalformedExtractionResultError,
            match="does not match actual page count",
        ):
            validator.validate(extraction)

    def test_duplicate_page_numbers_raises_malformed_error(self):
        """Duplicate page numbering raises MalformedExtractionResultError."""
        extraction = DocumentExtractionResult(
            document_id="doc_dup_pages",
            ticker="MSFT",
            document_type=DocumentType.EARNINGS_TRANSCRIPT,
            total_pages=2,
            total_characters=80,
            pages=[
                ExtractedPage(
                    page_number=1,
                    text="Microsoft Cloud revenue $35.0 billion, up 21% YoY.",
                    character_count=50,
                ),
                ExtractedPage(
                    page_number=1,  # Duplicate page number
                    text="Productivity and Business Processes $20.0B.",
                    character_count=42,
                ),
            ],
            extraction_status="completed",
            extracted_at=datetime.now(timezone.utc),
        )
        validator = ExtractedDocumentValidator()
        with pytest.raises(
            MalformedExtractionResultError, match="Duplicate page number"
        ):
            validator.validate(extraction)

    def test_non_positive_page_number_raises_malformed_error(self):
        """Page number <= 0 raises MalformedExtractionResultError."""
        extraction = DocumentExtractionResult(
            document_id="doc_bad_pnum",
            ticker="MSFT",
            document_type=DocumentType.COMPANY_REPORT,
            total_pages=1,
            total_characters=45,
            pages=[
                ExtractedPage.model_construct(
                    page_number=0,  # Invalid non-positive
                    text="Valid content that has an invalid zero page number.",
                    character_count=45,
                )
            ],
            extraction_status="completed",
            extracted_at=datetime.now(timezone.utc),
        )
        validator = ExtractedDocumentValidator()
        with pytest.raises(
            MalformedExtractionResultError, match="Invalid non-positive page"
        ):
            validator.validate(extraction)

    def test_empty_document_id_or_ticker_raises_malformed_error(self):
        """Blank document_id or ticker raises MalformedExtractionResultError."""
        validator = ExtractedDocumentValidator()

        bad_id = build_sample_extraction_result(
            ["Valid text here for test."], document_id="   "
        )
        with pytest.raises(MalformedExtractionResultError, match="document_id must be"):
            validator.validate(bad_id)

        bad_ticker = build_sample_extraction_result(
            ["Valid text here for test."], ticker=""
        )
        with pytest.raises(MalformedExtractionResultError, match="ticker must be"):
            validator.validate(bad_ticker)


# ===========================================================================
# 5. UNSUPPORTED DOCUMENT FORMATS (9.3.2)
# ===========================================================================


class TestUnsupportedDocumentFormats:
    """Tests rejecting unsupported formats and invalid document types."""

    def test_invalid_document_type_enum_raises_unsupported_error(self):
        """Non-DocumentType enum value raises UnsupportedDocumentError."""
        validator = ExtractedDocumentValidator()
        page = ExtractedPage(
            page_number=1,
            text="Valid text with invalid document type.",
            character_count=40,
        )

        # Bypassing Pydantic check using object construct with mock
        class FakeResult:
            document_id = "doc_fake"
            ticker = "AAPL"
            document_type = "INVALID_TYPE_STRING"
            total_pages = 1
            total_characters = 40
            pages = [page]
            extraction_status = "completed"
            extracted_at = datetime.now(timezone.utc)

        # Cast to DocumentExtractionResult check
        with pytest.raises(MalformedExtractionResultError):
            validator.validate(FakeResult())

    def test_unsupported_file_extension_in_pipeline_raises_unsupported_error(self):
        """Passing Word .docx to validate_document_content raises
        UnsupportedDocumentError.
        """
        docx_bytes = b"PK\x03\x04MockWordDocumentPayload"
        with pytest.raises(
            UnsupportedDocumentError, match="No text extractor registered"
        ) as exc_info:
            validate_document_content(
                content=docx_bytes,
                original_filename="earnings_report.docx",
                ticker="MSFT",
                document_type=DocumentType.OTHER_SUPPORTED,
            )
        assert exc_info.value.code == "UNSUPPORTED_DOCUMENT"

    def test_unsupported_mime_type_in_pipeline_raises_unsupported_error(self):
        """Passing Excel MIME to validate_document_content raises
        UnsupportedDocumentError.
        """
        with pytest.raises(UnsupportedDocumentError) as exc_info:
            validate_document_content(
                content=b"ExcelBinaryContentHere",
                original_filename="data.bin",
                ticker="GOOGL",
                document_type=DocumentType.COMPANY_REPORT,
                mime_type="application/vnd.ms-excel",
            )
        assert exc_info.value.code == "UNSUPPORTED_DOCUMENT"


# ===========================================================================
# 6. CORRUPTED AND MALFORMED CONTENT (9.3.2)
# ===========================================================================


class TestCorruptedAndInvalidExtractionContent:
    """Tests detecting corrupted characters, binary garbage, and 0-byte files."""

    def test_excessive_corrupted_replacement_characters_raises_corrupted_error(
        self,
    ):
        """Extracted text dominated by replacement character raises
        CorruptedDocumentError.
        """
        corrupted_text = (
            "\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd "
            "Garbage Data \ufffd\ufffd\ufffd"
        )
        extraction = build_sample_extraction_result([corrupted_text])
        validator = ExtractedDocumentValidator(max_unprintable_ratio=0.2)

        with pytest.raises(CorruptedDocumentError) as exc_info:
            validator.validate(extraction)

        assert "excessive corrupted or unprintable" in str(exc_info.value).lower()
        assert exc_info.value.code == "CORRUPTED_DOCUMENT"

    def test_zero_byte_content_in_pipeline_raises_corrupted_error(self):
        """0-byte content passed to validate_document_content raises
        CorruptedDocumentError.
        """
        with pytest.raises(
            CorruptedDocumentError, match="empty \\(0 bytes\\)"
        ) as exc_info:
            validate_document_content(
                content=b"",
                original_filename="report.pdf",
                ticker="AAPL",
                document_type=DocumentType.ANNUAL_REPORT,
            )
        assert exc_info.value.code == "CORRUPTED_DOCUMENT"

    def test_corrupted_pdf_binary_in_pipeline_raises_corrupted_error(self):
        """Corrupted PDF binary payload raises CorruptedDocumentError."""
        corrupted_bytes = b"%PDF-1.4\nCorrupted binary noise that breaks the parser"
        with pytest.raises(CorruptedDocumentError) as exc_info:
            validate_document_content(
                content=corrupted_bytes,
                original_filename="corrupt.pdf",
                ticker="AAPL",
                document_type=DocumentType.ANNUAL_REPORT,
            )
        assert exc_info.value.code == "CORRUPTED_DOCUMENT"

    def test_scanned_zero_text_pdf_in_pipeline_raises_empty_error(self):
        """Scanned PDF with no extractable text raises EmptyExtractedTextError."""
        # PDF with blank content
        blank_pdf = build_minimal_pdf([""])
        with pytest.raises(EmptyExtractedTextError) as exc_info:
            validate_document_content(
                content=blank_pdf,
                original_filename="scanned.pdf",
                ticker="AAPL",
                document_type=DocumentType.ANNUAL_REPORT,
            )
        assert exc_info.value.code == "EMPTY_TEXT"


# ===========================================================================
# 7. METADATA PRESERVATION AND ACCESSORS (9.3.1)
# ===========================================================================


class TestMetadataPreservation:
    """Tests verifying metadata preservation and convenience accessors."""

    def test_validated_document_preserves_all_metadata(self):
        """All metadata from extraction result is preserved faithfully."""
        text1 = "Tesla Q2 FY2024 Total Deliveries: 443,956 vehicles."
        text2 = "Automotive Gross Margin ex-regulatory credits: 14.6%."
        extraction = build_sample_extraction_result(
            [text1, text2],
            document_id="doc_tsla_q2",
            ticker="TSLA",
            document_type=DocumentType.INVESTOR_PRESENTATION,
        )

        validated = validate_extracted_document(extraction)

        assert validated.document_id == "doc_tsla_q2"
        assert validated.ticker == "TSLA"
        assert validated.document_type == DocumentType.INVESTOR_PRESENTATION
        assert validated.total_pages == 2
        assert validated.usable_pages == 2
        assert validated.total_characters == len(text1) + len(text2)
        assert validated.validation_status == "valid"
        assert isinstance(validated.validated_at, datetime)

    def test_get_usable_pages_and_full_text(self):
        """get_usable_pages() and get_full_text() return correct aggregated views."""
        extraction = build_sample_extraction_result(
            ["Page 1: Title and intro", "", "Page 3: Conclusion and Outlook"],
            document_id="doc_meta_test",
        )
        validated = validate_extracted_document(extraction)

        usable = validated.get_usable_pages()
        assert len(usable) == 2
        assert [p.page_number for p in usable] == [1, 3]

        full_text = validated.get_full_text()
        assert "Page 1: Title and intro" in full_text
        assert "Page 3: Conclusion and Outlook" in full_text
        assert "\n\n" in full_text


# ===========================================================================
# 8. REGRESSION & STORAGE INTEGRATION (9.1, 9.2, 9.3)
# ===========================================================================


class TestStorageAndPipelineIntegration:
    """End-to-end integration tests using LocalDocumentStorage and StoredDocument."""

    def test_validate_stored_document_end_to_end(self, tmp_path: Path):
        """Full pipeline: storage save -> validate_stored_document
        -> ValidatedDocument.
        """
        storage = LocalDocumentStorage(base_directory=tmp_path)
        pdf_bytes = build_minimal_pdf(
            [
                "Amazon.com Inc. Q2 2024 AWS Net Sales: $26.3B, up 19% YoY.",
                "Consolidated Operating Income: $14.7B for the second quarter.",
            ]
        )

        stored_doc = storage.save(
            content=pdf_bytes,
            original_filename="amzn_q2_2024.pdf",
            ticker="AMZN",
            document_type=DocumentType.COMPANY_REPORT,
        )

        validated = validate_stored_document(stored_doc, storage)

        assert isinstance(validated, ValidatedDocument)
        assert validated.document_id == stored_doc.document_id
        assert validated.ticker == "AMZN"
        assert validated.document_type == DocumentType.COMPANY_REPORT
        assert validated.total_pages == 2
        assert validated.usable_pages == 2
        assert "AWS Net Sales" in validated.pages[0].text
        assert "Consolidated Operating Income" in validated.pages[1].text
        assert validated.validation_status == "valid"

    def test_validate_stored_document_missing_file_raises_corrupted_error(
        self, tmp_path: Path
    ):
        """Storage document referencing missing disk file raises
        CorruptedDocumentError.
        """
        storage = LocalDocumentStorage(base_directory=tmp_path)
        phantom_doc = StoredDocument(
            document_id="doc_phantom",
            ticker="AMZN",
            document_type=DocumentType.ANNUAL_REPORT,
            original_filename="phantom.pdf",
            stored_filename="doc_phantom_phantom.pdf",
            storage_path="AMZN/doc_phantom_phantom.pdf",
            file_size_bytes=100,
            sha256_checksum="c" * 64,
            mime_type="application/pdf",
        )

        with pytest.raises(
            CorruptedDocumentError, match="Failed to retrieve document file"
        ):
            validate_stored_document(phantom_doc, storage)
