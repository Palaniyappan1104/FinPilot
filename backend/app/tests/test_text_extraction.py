"""Phase 9.2 Text Extraction unit and integration tests.

Verifies:
- 9.2.1 PDF text extraction (page-by-page preservation, financial number fidelity).
- 9.2.2 Plain text extraction (UTF-8, Latin-1 fallback, Word out-of-scope enforcement).
- 9.2.3 Extraction failure handling:
    * Scanned / image-only PDFs (detected deterministically, ScannedDocumentError).
    * Password-protected / encrypted PDFs (EncryptedDocumentError).
    * Corrupted / malformed PDFs (MalformedDocumentError).
    * Missing storage documents (DocumentNotFoundExtractionError).
    * Unsupported formats (UnsupportedDocumentFormatError).
- Text cleaning and normalization preserving exact numerical values and line breaks.
- Storage integration with LocalDocumentStorage.
"""

import io
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.models.documents import (
    DocumentExtractionResult,
    DocumentType,
    ExtractedPage,
    StoredDocument,
)
from app.services.text_extraction import (
    PdfTextExtractor,
    PlainTextExtractor,
    clean_extracted_text,
    get_document_extractor,
)
from app.services.text_extraction_exceptions import (
    DocumentNotFoundExtractionError,
    EncryptedDocumentError,
    MalformedDocumentError,
    ScannedDocumentError,
    UnsupportedDocumentFormatError,
)
from app.storage.local import LocalDocumentStorage


def build_test_pdf(pages: list[str]) -> bytes:
    """Construct a valid in-memory multi-page PDF byte payload
    with exact byte offsets.
    """
    total_pages = len(pages)
    page_obj_nums = [4 + i * 2 + 1 for i in range(total_pages)]
    content_obj_nums = [4 + i * 2 for i in range(total_pages)]

    # obj 1: Catalog
    obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    # obj 2: Pages root
    kids = " ".join(f"{p} 0 R" for p in page_obj_nums)
    obj2 = f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {total_pages} >>\nendobj\n"
    # obj 3: Base font
    obj3 = "3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    objects = [obj1, obj2, obj3]
    for c_num, p_num, text in zip(content_obj_nums, page_obj_nums, pages):
        stream_data = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET\n"
        c_obj = (
            f"{c_num} 0 obj\n<< /Length {len(stream_data)} >>\n"
            f"stream\n{stream_data}endstream\nendobj\n"
        )
        p_obj = (
            f"{p_num} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {c_num} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n"
        )
        objects.extend([c_obj, p_obj])

    full = "%PDF-1.4\n"
    obj_offsets = [0]
    for o in objects:
        obj_offsets.append(len(full.encode("latin1")))
        full += o

    xref_pos = len(full.encode("latin1"))
    xref = f"xref\n0 {len(obj_offsets)}\n0000000000 65535 f \n"
    for off in obj_offsets[1:]:
        xref += f"{off:010d} 00000 n \n"
    trailer = (
        f"trailer\n<< /Size {len(obj_offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )
    full += xref + trailer
    return full.encode("latin1")


def build_blank_pdf(page_count: int = 2) -> bytes:
    """Generate a PDF containing only blank/image-only pages (no text streams)."""
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def build_encrypted_pdf(password: str = "secret123") -> bytes:
    """Generate an encrypted password-protected PDF."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ===========================================================================
# 1. TEXT CLEANING UNIT TESTS
# ===========================================================================


class TestCleanExtractedText:
    """Unit tests for safe text cleaning and financial number preservation."""

    def test_clean_text_preserves_financial_figures(self):
        """Financial metrics, currency figures, and percentages remain identical."""
        raw = "Revenue  USD 12,345 million\nEBITDA  USD 3,450M\nMargin 28.5%"
        cleaned = clean_extracted_text(raw)
        assert "12,345" in cleaned
        assert "3,450M" in cleaned
        assert "28.5%" in cleaned

    def test_clean_text_normalizes_crlf_and_trailing_whitespace(self):
        """Windows CRLF is converted to LF and trailing spaces are removed."""
        raw = "First line with trailing spaces   \r\nSecond line \rThird line"
        cleaned = clean_extracted_text(raw)
        assert cleaned == "First line with trailing spaces\nSecond line\nThird line"

    def test_clean_text_collapses_excessive_newlines(self):
        """Runs of 3 or more newlines collapse into standard paragraph breaks
        (2 newlines).
        """
        raw = "Header\n\n\n\n\nParagraph text\n\n\nFooter"
        cleaned = clean_extracted_text(raw)
        assert cleaned == "Header\n\nParagraph text\n\nFooter"

    def test_clean_text_handles_non_breaking_spaces_and_control_chars(self):
        """Non-breaking spaces are normalized and unprintable control characters
        are stripped.
        """
        raw = "Operating\xa0Income\x00\x08: $500M"
        cleaned = clean_extracted_text(raw)
        assert cleaned == "Operating Income: $500M"

    def test_clean_text_empty_input(self):
        """Empty or whitespace-only input returns empty string."""
        assert clean_extracted_text("") == ""
        assert clean_extracted_text("   \n\n   ") == ""


# ===========================================================================
# 2. PDF TEXT EXTRACTION UNIT TESTS (9.2.1)
# ===========================================================================


class TestPdfTextExtractor:
    """Unit tests for PdfTextExtractor implementation."""

    def test_extract_single_page_pdf_success(self):
        """Single-page text PDF extracts correct text, page count,
        and character count.
        """
        pdf_bytes = build_test_pdf(["Apple Q3 Revenue: USD 81,800 million."])
        extractor = PdfTextExtractor()

        result = extractor.extract_from_bytes(
            content=pdf_bytes,
            document_id="doc_aapl_q3",
            ticker="AAPL",
            document_type=DocumentType.ANNUAL_REPORT,
        )

        assert isinstance(result, DocumentExtractionResult)
        assert result.document_id == "doc_aapl_q3"
        assert result.ticker == "AAPL"
        assert result.document_type == DocumentType.ANNUAL_REPORT
        assert result.total_pages == 1
        assert len(result.pages) == 1

        page1 = result.pages[0]
        assert isinstance(page1, ExtractedPage)
        assert page1.page_number == 1
        assert "Apple Q3 Revenue: USD 81,800 million" in page1.text
        assert page1.character_count == len(page1.text)
        assert result.total_characters == len(page1.text)
        assert result.extraction_status == "completed"

    def test_extract_multi_page_pdf_preserves_boundaries(self):
        """Multi-page PDF extracts each page separately with sequential
        1-indexed numbering.
        """
        pages_content = [
            "Page 1: Executive Overview and FY2024 Strategic Milestones",
            "Page 2: Consolidated Financial Results - Revenue USD 26,044M",
            "Page 3: Balance Sheet - Cash and Equivalents USD 15,200M",
        ]
        pdf_bytes = build_test_pdf(pages_content)
        extractor = PdfTextExtractor()

        result = extractor.extract_from_bytes(
            content=pdf_bytes,
            document_id="doc_nvda_fy24",
            ticker="NVDA",
            document_type=DocumentType.ANNUAL_REPORT,
        )

        assert result.total_pages == 3
        assert len(result.pages) == 3

        # Verify page 1
        assert result.pages[0].page_number == 1
        assert "Page 1: Executive Overview" in result.pages[0].text
        # Verify page 2
        assert result.pages[1].page_number == 2
        assert "Page 2: Consolidated Financial Results" in result.pages[1].text
        assert "USD 26,044M" in result.pages[1].text
        # Verify page 3
        assert result.pages[2].page_number == 3
        assert "Page 3: Balance Sheet" in result.pages[2].text
        assert "USD 15,200M" in result.pages[2].text

        assert result.total_characters == sum(p.character_count for p in result.pages)

    def test_extract_with_stored_document_and_storage(self, tmp_path: Path):
        """Integration test extracting a StoredDocument retrieved via
        LocalDocumentStorage.
        """
        storage = LocalDocumentStorage(base_directory=tmp_path)
        pdf_bytes = build_test_pdf(["Microsoft Cloud Revenue: USD 35,000 million."])

        stored = storage.save(
            content=pdf_bytes,
            original_filename="msft_earnings.pdf",
            ticker="MSFT",
            document_type=DocumentType.EARNINGS_TRANSCRIPT,
        )

        extractor = PdfTextExtractor()
        result = extractor.extract_text(stored, storage)

        assert result.document_id == stored.document_id
        assert result.ticker == "MSFT"
        assert result.document_type == DocumentType.EARNINGS_TRANSCRIPT
        assert result.total_pages == 1
        assert "Microsoft Cloud Revenue: USD 35,000 million." in result.pages[0].text


# ===========================================================================
# 3. EXTRACTION FAILURE & SCANNED PDF TESTS (9.2.3)
# ===========================================================================


class TestExtractionFailures:
    """Test suite for error detection: scanned PDFs, encryption, corruption,
    and missing files.
    """

    def test_scanned_or_image_only_pdf_raises_scanned_document_error(self):
        """Blank or image-only PDF with zero extractable text raises
        ScannedDocumentError.
        """
        scanned_bytes = build_blank_pdf(page_count=3)
        extractor = PdfTextExtractor()

        with pytest.raises(ScannedDocumentError) as exc_info:
            extractor.extract_from_bytes(
                content=scanned_bytes,
                document_id="doc_scanned_scan",
                ticker="AAPL",
            )

        err_msg = str(exc_info.value).lower()
        assert "scanned or image-only" in err_msg
        assert "ocr" in err_msg and "required" in err_msg
        assert "3 page(s)" in err_msg

    def test_encrypted_pdf_raises_encrypted_document_error(self):
        """Password-protected PDF raises EncryptedDocumentError with clear message."""
        enc_bytes = build_encrypted_pdf(password="secretpass")
        extractor = PdfTextExtractor()

        with pytest.raises(EncryptedDocumentError) as exc_info:
            extractor.extract_from_bytes(
                content=enc_bytes,
                document_id="doc_locked",
                ticker="TSLA",
            )

        err_msg = str(exc_info.value)
        assert "password-protected or encrypted" in err_msg.lower()

    def test_empty_content_raises_malformed_document_error(self):
        """0-byte payload raises MalformedDocumentError."""
        extractor = PdfTextExtractor()
        with pytest.raises(MalformedDocumentError, match="empty \\(0 bytes\\)"):
            extractor.extract_from_bytes(b"", document_id="doc_empty")

    def test_corrupted_bytes_raise_malformed_document_error(self):
        """Corrupted/random non-PDF bytes raise MalformedDocumentError."""
        corrupted_bytes = (
            b"%PDF-1.4\nCorrupted binary garbage that cannot be parsed by pypdf"
        )
        extractor = PdfTextExtractor()

        with pytest.raises(MalformedDocumentError) as exc_info:
            extractor.extract_from_bytes(corrupted_bytes, document_id="doc_corrupt")

        assert "pdf parsing error" in str(exc_info.value).lower()

    def test_missing_document_in_storage_raises_not_found(self, tmp_path: Path):
        """Attempting to extract a StoredDocument missing from disk raises
        DocumentNotFoundExtractionError.
        """
        storage = LocalDocumentStorage(base_directory=tmp_path)
        missing_doc = StoredDocument(
            document_id="doc_ghost",
            ticker="GOOGL",
            document_type=DocumentType.COMPANY_REPORT,
            original_filename="ghost.pdf",
            stored_filename="doc_ghost_ghost.pdf",
            storage_path="GOOGL/doc_ghost_ghost.pdf",
            file_size_bytes=100,
            sha256_checksum="a" * 64,
            mime_type="application/pdf",
        )

        extractor = PdfTextExtractor()
        with pytest.raises(
            DocumentNotFoundExtractionError, match="Document file missing from storage"
        ):
            extractor.extract_text(missing_doc, storage)

    def test_unsupported_format_for_pdf_extractor(self, tmp_path: Path):
        """Non-PDF document passed to PdfTextExtractor raises
        UnsupportedDocumentFormatError.
        """
        storage = LocalDocumentStorage(base_directory=tmp_path)
        doc = StoredDocument(
            document_id="doc_word",
            ticker="NVDA",
            document_type=DocumentType.OTHER_SUPPORTED,
            original_filename="report.docx",
            stored_filename="doc_word_report.docx",
            storage_path="NVDA/doc_word_report.docx",
            file_size_bytes=50,
            sha256_checksum="b" * 64,
            mime_type="application/vnd.openxmlformats",
        )

        extractor = PdfTextExtractor()
        with pytest.raises(
            UnsupportedDocumentFormatError, match="Only PDF documents are supported"
        ):
            extractor.extract_text(doc, storage)


# ===========================================================================
# 4. PLAIN TEXT EXTRACTION UNIT TESTS (9.2.2)
# ===========================================================================


class TestPlainTextExtractor:
    """Unit tests for PlainTextExtractor implementation."""

    def test_extract_utf8_plain_text(self):
        """Valid UTF-8 plain text extracts as page 1."""
        raw_text = "Earnings Transcript:\nQ2 Net Income: $1,200M\nEPS: $2.45"
        extractor = PlainTextExtractor()

        result = extractor.extract_from_bytes(
            content=raw_text.encode("utf-8"),
            document_id="doc_txt_utf8",
            ticker="AAPL",
            document_type=DocumentType.EARNINGS_TRANSCRIPT,
        )

        assert result.total_pages == 1
        assert len(result.pages) == 1
        assert result.pages[0].page_number == 1
        assert "Q2 Net Income: $1,200M" in result.pages[0].text
        assert result.total_characters == len(result.pages[0].text)

    def test_extract_latin1_fallback_plain_text(self):
        """Latin-1 encoded plain text decodes gracefully."""
        raw_text = "Kreditwürdigkeit: Höchstnote für FinPilot £100M"
        latin1_bytes = raw_text.encode("latin-1")
        extractor = PlainTextExtractor()

        result = extractor.extract_from_bytes(
            content=latin1_bytes,
            document_id="doc_txt_latin1",
            ticker="SAP",
        )

        assert result.total_pages == 1
        assert "FinPilot" in result.pages[0].text

    def test_extract_plain_text_empty_error(self):
        """0-byte plain text file raises MalformedDocumentError."""
        extractor = PlainTextExtractor()
        with pytest.raises(MalformedDocumentError, match="empty \\(0 bytes\\)"):
            extractor.extract_from_bytes(b"", document_id="doc_txt_empty")

    def test_extract_plain_text_whitespace_only_error(self):
        """Whitespace-only plain text file raises ScannedDocumentError."""
        extractor = PlainTextExtractor()
        with pytest.raises(ScannedDocumentError, match="no usable text content"):
            extractor.extract_from_bytes(b"   \n\n\t   ", document_id="doc_txt_blank")


# ===========================================================================
# 5. FACTORY & REGISTRY TESTS
# ===========================================================================


class TestDocumentExtractorFactory:
    """Unit tests for get_document_extractor factory."""

    def test_factory_returns_pdf_extractor_for_pdf_mime_and_extension(self):
        """PDF MIME and .pdf filename return PdfTextExtractor."""
        assert isinstance(
            get_document_extractor(mime_type="application/pdf"), PdfTextExtractor
        )
        assert isinstance(
            get_document_extractor(filename="report.pdf"), PdfTextExtractor
        )
        assert isinstance(
            get_document_extractor(filename="ANNUAL_REPORT.PDF"), PdfTextExtractor
        )

    def test_factory_returns_plain_text_extractor(self):
        """Plain text MIME and .txt filename return PlainTextExtractor."""
        assert isinstance(
            get_document_extractor(mime_type="text/plain"), PlainTextExtractor
        )
        assert isinstance(
            get_document_extractor(filename="notes.txt"), PlainTextExtractor
        )

    def test_factory_unsupported_formats_raise_error(self):
        """Unsupported formats like Word or Excel raise
        UnsupportedDocumentFormatError.
        """
        with pytest.raises(
            UnsupportedDocumentFormatError, match="No text extractor registered"
        ):
            get_document_extractor(filename="report.docx")

        with pytest.raises(
            UnsupportedDocumentFormatError, match="No text extractor registered"
        ):
            get_document_extractor(mime_type="application/vnd.ms-excel")
