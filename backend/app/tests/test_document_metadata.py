"""Tests for Document Metadata (Phase 9.5).

Verifies the 15 required metadata scenarios:
1. Complete valid metadata creation
2. Document ID preservation
3. Source document/filename preservation
4. Page number/page range preservation
5. Ticker/company preservation
6. DocumentType preservation using existing enum
7. Upload date preservation
8. Metadata serialization to plain dict for vector stores
9. Shared metadata retention across chunks from same document
10. Distinction of chunks across different documents
11. Rejection of invalid/missing required metadata
12. Invalid page metadata handling
13. Invalid ticker/document type handling
14. Regression compatibility with Phase 9.1-9.4 models
15. End-to-end metadata propagation
    (StoredDocument -> ValidatedDocument -> DocumentChunk -> ChunkMetadata)
"""

from datetime import datetime, timezone

import pytest

from app.models.documents import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    StoredDocument,
    ValidatedDocument,
    ValidatedPage,
)
from app.services.document_chunker import chunk_document
from app.services.document_metadata import (
    InvalidMetadataError,
    MissingSourceProvenanceError,
    attach_metadata_to_chunked_document,
    build_document_chunks_metadata,
    create_chunk_metadata,
)


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """Fixture providing a standard valid DocumentChunk."""
    return DocumentChunk(
        chunk_id="doc_AAPL_10k_c0001",
        document_id="doc_AAPL_10k",
        ticker="AAPL",
        document_type=DocumentType.ANNUAL_REPORT,
        chunk_index=1,
        text="Apple Inc. designs and manufactures high-quality electronics.",
        character_count=62,
        word_count=8,
        page_numbers=[1],
        start_page=1,
        end_page=1,
        section_name="ITEM 1. BUSINESS",
    )


@pytest.fixture
def sample_multi_page_chunk() -> DocumentChunk:
    """Fixture providing a chunk spanning multiple pages."""
    return DocumentChunk(
        chunk_id="doc_MSFT_10k_c0003",
        document_id="doc_MSFT_10k",
        ticker="MSFT",
        document_type=DocumentType.ANNUAL_REPORT,
        chunk_index=3,
        text="Cloud revenue increased 25% year over year across Azure services.",
        character_count=67,
        word_count=10,
        page_numbers=[2, 3],
        start_page=2,
        end_page=3,
        section_name="ITEM 7. MD&A",
    )


class TestDocumentMetadataPhase95:
    """Test suite for Phase 9.5 Document Metadata."""

    # 1. Complete valid metadata creation
    def test_complete_valid_metadata_creation(self, sample_chunk: DocumentChunk):
        upload_time = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        meta = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="apple_q4_2025.pdf",
            upload_date=upload_time,
        )

        assert isinstance(meta, ChunkMetadata)
        assert meta.chunk_id == "doc_AAPL_10k_c0001"
        assert meta.document_id == "doc_AAPL_10k"
        assert meta.source_document == "apple_q4_2025.pdf"
        assert meta.ticker == "AAPL"
        assert meta.document_type == DocumentType.ANNUAL_REPORT
        assert meta.page_numbers == [1]
        assert meta.start_page == 1
        assert meta.end_page == 1
        assert meta.upload_date == upload_time
        assert meta.chunk_index == 1
        assert meta.section_name == "ITEM 1. BUSINESS"
        assert meta.character_count == 62
        assert meta.word_count == 8

    # 2. Document ID preservation
    def test_document_id_preservation(self, sample_chunk: DocumentChunk):
        meta = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="report.pdf",
            document_id="doc_OVERRIDE_ID",
        )
        assert meta.document_id == "doc_OVERRIDE_ID"

        # Default preserves chunk document_id
        meta_default = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="report.pdf",
        )
        assert meta_default.document_id == sample_chunk.document_id

    # 3. Source document/filename preservation
    def test_source_document_filename_preservation(self, sample_chunk: DocumentChunk):
        filename = "FY2025_Annual_Report_Final_v2.pdf"
        meta = create_chunk_metadata(
            chunk=sample_chunk,
            source_document=filename,
        )
        assert meta.source_document == filename

    # 4. Page number/page range preservation
    def test_page_number_and_range_preservation(
        self, sample_multi_page_chunk: DocumentChunk
    ):
        meta = create_chunk_metadata(
            chunk=sample_multi_page_chunk,
            source_document="msft_10k.pdf",
        )
        assert meta.page_numbers == [2, 3]
        assert meta.start_page == 2
        assert meta.end_page == 3

    # 5. Ticker/company preservation
    def test_ticker_preservation_and_normalization(self, sample_chunk: DocumentChunk):
        meta = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="report.pdf",
            ticker="tsla",
        )
        assert meta.ticker == "TSLA"

    # 6. DocumentType preservation using existing enum
    def test_document_type_preservation(self, sample_chunk: DocumentChunk):
        meta = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="report.pdf",
            document_type=DocumentType.EARNINGS_TRANSCRIPT,
        )
        assert meta.document_type == DocumentType.EARNINGS_TRANSCRIPT
        assert isinstance(meta.document_type, DocumentType)

        # String enum parsing
        meta_str = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="report.pdf",
            document_type="investor_presentation",
        )
        assert meta_str.document_type == DocumentType.INVESTOR_PRESENTATION

    # 7. Upload date preservation
    def test_upload_date_preservation(self, sample_chunk: DocumentChunk):
        custom_date = datetime(2025, 11, 20, 8, 30, 0, tzinfo=timezone.utc)
        meta = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="report.pdf",
            upload_date=custom_date,
        )
        assert meta.upload_date == custom_date
        assert meta.upload_date.tzinfo is not None

        # Naive datetime gets converted to UTC
        naive_date = datetime(2025, 11, 20, 8, 30, 0)
        meta_naive = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="report.pdf",
            upload_date=naive_date,
        )
        assert meta_naive.upload_date.tzinfo == timezone.utc

    # 8. Metadata serialization to plain dict for vector stores
    def test_metadata_serialization_vector_store(
        self, sample_multi_page_chunk: DocumentChunk
    ):
        upload_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        meta = create_chunk_metadata(
            chunk=sample_multi_page_chunk,
            source_document="msft_10k.pdf",
            upload_date=upload_time,
        )

        vector_meta = meta.to_vector_store_metadata()
        assert isinstance(vector_meta, dict)

        # All values must be ChromaDB-safe primitives
        for key, val in vector_meta.items():
            assert isinstance(
                val, (str, int, float, bool)
            ), f"Value for {key} is {type(val)}, not a primitive!"

        assert vector_meta["chunk_id"] == "doc_MSFT_10k_c0003"
        assert vector_meta["document_id"] == "doc_MSFT_10k"
        assert vector_meta["source_document"] == "msft_10k.pdf"
        assert vector_meta["ticker"] == "MSFT"
        assert vector_meta["document_type"] == "annual_report"
        assert vector_meta["page_numbers"] == "2,3"
        assert vector_meta["start_page"] == 2
        assert vector_meta["end_page"] == 3
        assert vector_meta["upload_date"] == "2026-01-01T00:00:00+00:00"
        assert vector_meta["chunk_index"] == 3
        assert vector_meta["section_name"] == "ITEM 7. MD&A"

        # Also verify rich to_dict()
        rich_dict = meta.to_dict()
        assert rich_dict["page_numbers"] == [2, 3]
        assert isinstance(rich_dict["page_numbers"], list)

    # 9. Shared metadata retention across chunks from same document
    def test_shared_metadata_across_chunks(self, sample_chunk: DocumentChunk):
        chunk_2 = DocumentChunk(
            chunk_id="doc_AAPL_10k_c0002",
            document_id="doc_AAPL_10k",
            ticker="AAPL",
            document_type=DocumentType.ANNUAL_REPORT,
            chunk_index=2,
            text="Operating income increased across services and wearables segments.",
            character_count=67,
            word_count=8,
            page_numbers=[2],
            start_page=2,
            end_page=2,
        )

        chunks_list = [sample_chunk, chunk_2]
        upload_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
        meta_list = build_document_chunks_metadata(
            chunks=chunks_list,
            source_document="apple_10k.pdf",
            upload_date=upload_time,
        )

        assert len(meta_list) == 2
        # Shared document-level provenance
        assert meta_list[0].document_id == meta_list[1].document_id == "doc_AAPL_10k"
        assert (
            meta_list[0].source_document
            == meta_list[1].source_document
            == "apple_10k.pdf"
        )
        assert meta_list[0].ticker == meta_list[1].ticker == "AAPL"
        assert (
            meta_list[0].document_type
            == meta_list[1].document_type
            == DocumentType.ANNUAL_REPORT
        )
        assert meta_list[0].upload_date == meta_list[1].upload_date == upload_time

        # Distinct chunk-level fields
        assert meta_list[0].chunk_id != meta_list[1].chunk_id
        assert meta_list[0].chunk_index == 1
        assert meta_list[1].chunk_index == 2
        assert meta_list[0].page_numbers == [1]
        assert meta_list[1].page_numbers == [2]

    # 10. Distinction of chunks across different documents
    def test_distinction_across_different_documents(self, sample_chunk: DocumentChunk):
        chunk_goog = DocumentChunk(
            chunk_id="doc_GOOG_10k_c0001",
            document_id="doc_GOOG_10k",
            ticker="GOOG",
            document_type=DocumentType.ANNUAL_REPORT,
            chunk_index=1,
            text="Alphabet revenues grew driven by Search and YouTube ads.",
            character_count=56,
            word_count=8,
            page_numbers=[1],
            start_page=1,
            end_page=1,
        )

        meta_aapl = create_chunk_metadata(
            chunk=sample_chunk,
            source_document="apple_10k.pdf",
        )
        meta_goog = create_chunk_metadata(
            chunk=chunk_goog,
            source_document="goog_10k.pdf",
        )

        assert meta_aapl.document_id != meta_goog.document_id
        assert meta_aapl.ticker != meta_goog.ticker
        assert meta_aapl.source_document != meta_goog.source_document
        assert meta_aapl.chunk_id != meta_goog.chunk_id

    # 11. Rejection of invalid/missing required metadata
    def test_rejection_of_missing_required_metadata(self, sample_chunk: DocumentChunk):
        # Empty source_document
        with pytest.raises(MissingSourceProvenanceError) as exc_info:
            create_chunk_metadata(chunk=sample_chunk, source_document="")
        assert "source_document" in str(exc_info.value)

        # Whitespace source_document
        with pytest.raises(MissingSourceProvenanceError):
            create_chunk_metadata(chunk=sample_chunk, source_document="   ")

        # Empty document_id override
        with pytest.raises(MissingSourceProvenanceError):
            create_chunk_metadata(
                chunk=sample_chunk,
                source_document="valid.pdf",
                document_id="  ",
            )

        # Missing chunk
        with pytest.raises(InvalidMetadataError):
            create_chunk_metadata(chunk=None, source_document="valid.pdf")  # type: ignore

    # 12. Invalid page metadata handling
    def test_invalid_page_metadata_handling(self):
        # Empty page_numbers
        bad_chunk = DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            ticker="AAPL",
            document_type=DocumentType.ANNUAL_REPORT,
            chunk_index=1,
            text="Some financial text for chunk validation testing.",
            character_count=49,
            word_count=7,
            page_numbers=[1],
            start_page=1,
            end_page=1,
        )

        # Non-positive page number
        object.__setattr__(bad_chunk, "page_numbers", [0])
        with pytest.raises(InvalidMetadataError) as exc_info:
            create_chunk_metadata(chunk=bad_chunk, source_document="doc.pdf")
        assert "Page numbers must be positive" in str(exc_info.value)

    # 13. Invalid ticker/document type handling
    def test_invalid_ticker_and_doc_type_handling(self, sample_chunk: DocumentChunk):
        # Empty ticker override
        with pytest.raises(MissingSourceProvenanceError):
            create_chunk_metadata(
                chunk=sample_chunk,
                source_document="doc.pdf",
                ticker="  ",
            )

        # Unsupported document type string
        with pytest.raises(InvalidMetadataError) as exc_info:
            create_chunk_metadata(
                chunk=sample_chunk,
                source_document="doc.pdf",
                document_type="non_existent_type",
            )
        assert "Unsupported document type" in str(exc_info.value)

    # 14. Regression compatibility with Phase 9.1-9.4 models
    def test_regression_compatibility(self, sample_chunk: DocumentChunk):
        # DocumentChunk can exist without metadata (Phase 9.4 compatibility)
        assert sample_chunk.metadata is None
        assert sample_chunk.chunk_id == "doc_AAPL_10k_c0001"

        # ValidatedDocument can exist without original_filename / uploaded_at
        val_doc = ValidatedDocument(
            document_id="doc_legacy",
            ticker="AAPL",
            document_type=DocumentType.ANNUAL_REPORT,
            total_pages=1,
            usable_pages=1,
            total_characters=100,
            total_words=20,
            pages=[
                ValidatedPage(
                    page_number=1,
                    text="Quarterly net income was fifty billion dollars.",
                    character_count=47,
                    word_count=7,
                    is_usable=True,
                )
            ],
        )
        assert val_doc.original_filename is None
        assert val_doc.uploaded_at is None

        # Chunking works identically
        chunked = chunk_document(val_doc)
        assert chunked.total_chunks >= 1
        assert chunked.chunks[0].metadata is None

        # Enriched attachment works seamlessly on ChunkedDocument
        enriched = attach_metadata_to_chunked_document(
            chunked,
            source_document="apple_legacy.pdf",
        )
        assert enriched.chunks[0].metadata is not None
        assert enriched.chunks[0].metadata.source_document == "apple_legacy.pdf"

    # 15. End-to-end metadata propagation
    def test_end_to_end_metadata_propagation(self):
        """StoredDocument -> ValidatedDocument -> DocumentChunk -> ChunkMetadata."""
        # Step A: StoredDocument (Phase 9.1)
        upload_time = datetime(2026, 4, 10, 14, 30, 0, tzinfo=timezone.utc)
        stored_doc = StoredDocument(
            document_id="doc_NVDA_q1_2026",
            ticker="NVDA",
            document_type=DocumentType.INVESTOR_PRESENTATION,
            original_filename="nvda_investor_day_2026.pdf",
            stored_filename="uuid_nvda.pdf",
            storage_path="documents/NVDA/uuid_nvda.pdf",
            file_size_bytes=1024,
            sha256_checksum="abc123hash",
            uploaded_at=upload_time,
        )

        # Step B: ValidatedDocument with metadata (Phase 9.3)
        valid_page = ValidatedPage(
            page_number=1,
            text="Data Center revenue expanded 400% driven by accelerated computing.",
            character_count=67,
            word_count=9,
            is_usable=True,
        )
        validated_doc = ValidatedDocument(
            document_id=stored_doc.document_id,
            ticker=stored_doc.ticker,
            document_type=stored_doc.document_type,
            total_pages=1,
            usable_pages=1,
            total_characters=67,
            total_words=9,
            pages=[valid_page],
            original_filename=stored_doc.original_filename,
            uploaded_at=stored_doc.uploaded_at,
        )

        assert validated_doc.original_filename == "nvda_investor_day_2026.pdf"
        assert validated_doc.uploaded_at == upload_time

        # Step C: DocumentChunking (Phase 9.4)
        chunked = chunk_document(validated_doc)
        assert chunked.total_chunks == 1

        # Step D: Attach Phase 9.5 Metadata using ValidatedDocument provenance
        enriched_chunked = attach_metadata_to_chunked_document(
            chunked,
            validated_doc=validated_doc,
        )

        chunk = enriched_chunked.chunks[0]
        assert chunk.metadata is not None
        assert chunk.metadata.document_id == "doc_NVDA_q1_2026"
        assert chunk.metadata.source_document == "nvda_investor_day_2026.pdf"
        assert chunk.metadata.ticker == "NVDA"
        assert chunk.metadata.document_type == DocumentType.INVESTOR_PRESENTATION
        assert chunk.metadata.upload_date == upload_time
        assert chunk.metadata.page_numbers == [1]
        assert chunk.metadata.start_page == 1
        assert chunk.metadata.end_page == 1

        # Step E: Confirm vector store payload readiness
        vector_meta = chunk.metadata.to_vector_store_metadata()
        assert vector_meta["ticker"] == "NVDA"
        assert vector_meta["source_document"] == "nvda_investor_day_2026.pdf"
        assert vector_meta["upload_date"] == "2026-04-10T14:30:00+00:00"
        assert vector_meta["document_type"] == "investor_presentation"
