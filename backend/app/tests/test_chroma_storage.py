"""Unit and integration tests for Phase 9.7: ChromaDB Storage.

Tests cover all 21 required scenarios using isolated in-memory and temporary
ChromaDB clients. No real external API calls (Gemini/OpenAI) are made.

Scenarios tested:
1.  ChromaDB initialization (ephemeral & persistent).
2.  Persistent/local storage configuration.
3.  Deterministic collection naming (finpilot_{ticker}_{document_id}).
4.  Collection separation by company/document.
5.  Valid single-chunk insertion.
6.  Multiple-chunk insertion.
7.  Embedding vector storage and retrieval.
8.  Chunk document text storage and retrieval.
9.  Metadata storage and retrieval.
10. Full citation provenance preservation (to_vector_store_metadata).
11. Upsert existing chunk (updates vector, text, and metadata).
12. Repeated ingestion does not create uncontrolled duplicates (count remains 1).
13. Duplicate/conflicting chunk IDs within the same batch rejected.
14. Empty batch handling (graceful no-op, 0 records affected).
15. Invalid embedding vector (empty or non-finite) rejected.
16. Inconsistent vector dimensions across batch rejected.
17. Invalid non-primitive metadata rejected.
18. Invalid collection name rejected.
19. Storage/database error handling.
20. Regression compatibility with Phase 9.1–9.6 models.
21. End-to-end: Chunk -> Embedder -> EmbeddingResult -> ChromaVectorStore.
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

import chromadb
import pytest

from app.models.documents import (
    ChunkedDocument,
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
)
from app.models.embeddings import (
    ChunkEmbeddingRequest,
    ChunkEmbeddingResult,
    EmbeddingModelInfo,
)
from app.models.vector_store import VectorRecord, VectorStoreInsertionResult
from app.providers.embedding import EmbeddingProvider
from app.services.document_embedder import DocumentEmbedder
from app.services.document_vector_service import (
    chunk_embedding_to_vector_record,
    store_document_embeddings,
)
from app.storage.chroma_vector_store import ChromaVectorStore, get_vector_store
from app.storage.vector_base import (
    build_company_collection_name,
    build_document_collection_name,
    validate_collection_name,
)
from app.storage.vector_exceptions import (
    CollectionNotFoundError,
    DuplicateRecordIdError,
    InvalidCollectionNameError,
    VectorDimensionMismatchError,
    VectorStoreError,
    VectorStoreInitializationError,
)

# ===========================================================================
# CONSTANTS & FIXTURES
# ===========================================================================

_TEST_TICKER = "AAPL"
_TEST_DOC_ID = "doc-annual-2024"
_TEST_COLLECTION = "finpilot_aapl_doc-annual-2024"
_TEST_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_TEST_DIMS = 8


@pytest.fixture
def ephemeral_chroma_client():
    """Isolated in-memory ChromaDB client for testing, reset before each test."""
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.Client(ChromaSettings(is_persistent=False, allow_reset=True))
    client.reset()
    yield client
    client.reset()


@pytest.fixture
def vector_store(ephemeral_chroma_client) -> ChromaVectorStore:
    """ChromaVectorStore backed by an ephemeral in-memory client."""
    return ChromaVectorStore(client=ephemeral_chroma_client)


@pytest.fixture
def sample_vector_record() -> VectorRecord:
    """Valid VectorRecord for single-record tests."""
    return VectorRecord(
        id="chunk-001",
        embedding=list(_TEST_VECTOR),
        document="Apple Inc. reported record revenue in Q4 2024.",
        metadata={
            "chunk_id": "chunk-001",
            "document_id": _TEST_DOC_ID,
            "ticker": _TEST_TICKER,
            "document_type": "annual_report",
            "page_numbers": "1,2",
            "start_page": 1,
            "end_page": 2,
            "upload_date": "2024-01-01T00:00:00+00:00",
            "chunk_index": 1,
            "character_count": 47,
            "word_count": 8,
        },
    )


@pytest.fixture
def sample_vector_records(sample_vector_record: VectorRecord) -> List[VectorRecord]:
    """Three valid VectorRecords for batch testing."""
    records = [sample_vector_record]
    for i in range(2, 4):
        records.append(
            VectorRecord(
                id=f"chunk-{i:03d}",
                embedding=[float(i * 0.1)] * _TEST_DIMS,
                document=f"Financial statement chunk {i} details and disclosures.",
                metadata={
                    "chunk_id": f"chunk-{i:03d}",
                    "document_id": _TEST_DOC_ID,
                    "ticker": _TEST_TICKER,
                    "document_type": "annual_report",
                    "page_numbers": str(i),
                    "start_page": i,
                    "end_page": i,
                    "upload_date": "2024-01-01T00:00:00+00:00",
                    "chunk_index": i,
                    "character_count": 55,
                    "word_count": 7,
                },
            )
        )
    return records


class MockTestEmbeddingProvider(EmbeddingProvider):
    """Mock EmbeddingProvider for end-to-end tests."""

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-embed-001"

    @property
    def dimensions(self) -> int:
        return _TEST_DIMS

    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider="mock",
            model_name="mock-embed-001",
            dimensions=_TEST_DIMS,
        )

    def embed_single(self, request: ChunkEmbeddingRequest) -> ChunkEmbeddingResult:
        return ChunkEmbeddingResult(
            chunk_id=request.chunk_id,
            document_id=request.document_id,
            ticker=request.ticker,
            document_type=request.document_type,
            page_numbers=request.page_numbers,
            start_page=request.start_page,
            end_page=request.end_page,
            chunk_index=request.chunk_index,
            section_name=request.section_name,
            metadata=request.metadata,
            text=request.text,
            embedding=list(_TEST_VECTOR),
            embedding_model="mock-embed-001",
            embedding_dimensions=_TEST_DIMS,
        )

    def embed_batch(
        self, requests: List[ChunkEmbeddingRequest]
    ) -> List[ChunkEmbeddingResult]:
        return [self.embed_single(r) for r in requests]


# ===========================================================================
# TESTS
# ===========================================================================


class TestPhase97ChromaStorage:
    """Test suite covering Phase 9.7 ChromaDB Storage requirements."""

    # ------------------------------------------------------------------
    # 1. ChromaDB initialization
    # ------------------------------------------------------------------

    def test_chroma_initialization_with_injected_client(
        self, ephemeral_chroma_client: chromadb.ClientAPI
    ) -> None:
        """ChromaVectorStore initializes successfully with an injected client."""
        store = ChromaVectorStore(client=ephemeral_chroma_client)
        assert store.store_name == "chroma"
        assert store.persist_directory is None

    def test_chroma_initialization_failure_raises_typed_error(self) -> None:
        """VectorStoreInitializationError is raised when path cannot be created."""
        fake_settings = MagicMock()
        # Non-creatable path on Windows / system
        fake_settings.CHROMA_PERSIST_DIRECTORY = "Z:\\non_existent_drive\\chroma"
        with pytest.raises(VectorStoreInitializationError):
            ChromaVectorStore(settings=fake_settings)

    # ------------------------------------------------------------------
    # 2. Persistent/local storage configuration
    # ------------------------------------------------------------------

    def test_persistent_storage_configuration(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """ChromaVectorStore creates persistent storage at configured path."""
        target_dir = str(tmp_path / "chroma_test_persist")
        store = ChromaVectorStore(persist_directory=target_dir)
        assert store.persist_directory == str(tmp_path / "chroma_test_persist")
        assert (tmp_path / "chroma_test_persist").exists()

    # ------------------------------------------------------------------
    # 3. Deterministic collection naming
    # ------------------------------------------------------------------

    def test_deterministic_collection_naming(self) -> None:
        """build_document_collection_name creates predictable, sanitized names."""
        name1 = build_document_collection_name("AAPL", "doc-123")
        name2 = build_document_collection_name("aapl", "doc-123")
        assert name1 == "finpilot_aapl_doc-123"
        assert name1 == name2

    def test_deterministic_collection_naming_sanitization(self) -> None:
        """Special characters in ticker or document_id are sanitized safely."""
        name = build_document_collection_name("BRK.A", "doc/annual#2024")
        assert ".." not in name
        assert "/" not in name
        assert "#" not in name
        assert name.startswith("finpilot_")

    def test_build_company_collection_name(self) -> None:
        """build_company_collection_name creates company-keyed names."""
        name = build_company_collection_name("MSFT")
        assert name == "finpilot_company_msft"

    # ------------------------------------------------------------------
    # 4. Collection separation by company/document
    # ------------------------------------------------------------------

    def test_collection_separation_by_company_and_document(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """Different documents or companies are stored in distinct collections."""
        col_apple = build_document_collection_name("AAPL", "doc-001")
        col_msft = build_document_collection_name("MSFT", "doc-001")

        vector_store.upsert_records(col_apple, [sample_vector_record])
        assert vector_store.has_collection(col_apple)
        assert not vector_store.has_collection(col_msft)

        # Count in Apple collection is 1, MSFT doesn't exist
        assert vector_store.count_records(col_apple) == 1

    # ------------------------------------------------------------------
    # 5. Valid single-chunk insertion
    # ------------------------------------------------------------------

    def test_valid_single_chunk_insertion(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """A single valid VectorRecord is successfully inserted."""
        result = vector_store.upsert_records(_TEST_COLLECTION, [sample_vector_record])
        assert isinstance(result, VectorStoreInsertionResult)
        assert result.total_records == 1
        assert result.record_ids == [sample_vector_record.id]
        assert vector_store.count_records(_TEST_COLLECTION) == 1

    # ------------------------------------------------------------------
    # 6. Multiple-chunk insertion
    # ------------------------------------------------------------------

    def test_multiple_chunk_insertion(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_records: List[VectorRecord],
    ) -> None:
        """A batch of multiple VectorRecords is inserted."""
        result = vector_store.upsert_records(_TEST_COLLECTION, sample_vector_records)
        assert result.total_records == len(sample_vector_records)
        assert vector_store.count_records(_TEST_COLLECTION) == len(
            sample_vector_records
        )

    # ------------------------------------------------------------------
    # 7. Embedding vector storage and retrieval
    # ------------------------------------------------------------------

    def test_embedding_storage_and_retrieval(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """Embedding vectors are stored and retrieved with correct numerical values."""
        vector_store.upsert_records(_TEST_COLLECTION, [sample_vector_record])
        fetched = vector_store.get_record(_TEST_COLLECTION, sample_vector_record.id)
        assert fetched is not None
        assert len(fetched.embedding) == len(sample_vector_record.embedding)
        for orig, stored in zip(sample_vector_record.embedding, fetched.embedding):
            assert abs(orig - stored) < 1e-5

    # ------------------------------------------------------------------
    # 8. Chunk document text storage and retrieval
    # ------------------------------------------------------------------

    def test_chunk_text_storage_and_retrieval(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """Document text is stored and retrieved faithfully."""
        vector_store.upsert_records(_TEST_COLLECTION, [sample_vector_record])
        fetched = vector_store.get_record(_TEST_COLLECTION, sample_vector_record.id)
        assert fetched is not None
        assert fetched.document == sample_vector_record.document

    # ------------------------------------------------------------------
    # 9. Metadata storage and retrieval
    # ------------------------------------------------------------------

    def test_metadata_storage_and_retrieval(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """Metadata key-values are stored and retrieved faithfully."""
        vector_store.upsert_records(_TEST_COLLECTION, [sample_vector_record])
        fetched = vector_store.get_record(_TEST_COLLECTION, sample_vector_record.id)
        assert fetched is not None
        assert fetched.metadata["ticker"] == _TEST_TICKER
        assert fetched.metadata["document_id"] == _TEST_DOC_ID
        assert fetched.metadata["chunk_index"] == 1

    # ------------------------------------------------------------------
    # 10. Full citation provenance preservation
    # ------------------------------------------------------------------

    def test_full_provenance_preservation(
        self,
        vector_store: ChromaVectorStore,
    ) -> None:
        """ChunkMetadata.to_vector_store_metadata() preserves all citation fields."""
        meta = ChunkMetadata(
            chunk_id="chunk-prov-001",
            document_id="doc-apple-2024",
            source_document="apple_10k_2024.pdf",
            ticker="AAPL",
            document_type=DocumentType.ANNUAL_REPORT,
            page_numbers=[3, 4],
            start_page=3,
            end_page=4,
            upload_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            chunk_index=5,
            section_name="Item 7. MD&A",
            character_count=120,
            word_count=20,
        )
        v_meta = meta.to_vector_store_metadata()
        rec = VectorRecord(
            id="chunk-prov-001",
            embedding=list(_TEST_VECTOR),
            document="Management's discussion and analysis of financial condition.",
            metadata=v_meta,
        )
        col = build_document_collection_name("AAPL", "doc-apple-2024")
        vector_store.upsert_records(col, [rec])
        stored = vector_store.get_record(col, "chunk-prov-001")
        assert stored is not None
        assert stored.metadata["document_id"] == "doc-apple-2024"
        assert stored.metadata["source_document"] == "apple_10k_2024.pdf"
        assert stored.metadata["ticker"] == "AAPL"
        assert stored.metadata["document_type"] == "annual_report"
        assert stored.metadata["page_numbers"] == "3,4"
        assert stored.metadata["start_page"] == 3
        assert stored.metadata["end_page"] == 4
        assert stored.metadata["chunk_index"] == 5
        assert stored.metadata["section_name"] == "Item 7. MD&A"

    # ------------------------------------------------------------------
    # 11. Upsert existing chunk (updates vector, text, and metadata)
    # ------------------------------------------------------------------

    def test_upsert_existing_chunk_updates_record(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """Upserting an existing chunk ID updates its content and metadata."""
        vector_store.upsert_records(_TEST_COLLECTION, [sample_vector_record])

        # Updated record with the same ID
        updated_vector = [0.9] * _TEST_DIMS
        updated_text = "Updated Apple revenue disclosures."
        updated_record = VectorRecord(
            id=sample_vector_record.id,
            embedding=updated_vector,
            document=updated_text,
            metadata={"ticker": "AAPL", "version": 2},
        )
        vector_store.upsert_records(_TEST_COLLECTION, [updated_record])

        assert vector_store.count_records(_TEST_COLLECTION) == 1
        fetched = vector_store.get_record(_TEST_COLLECTION, sample_vector_record.id)
        assert fetched is not None
        assert fetched.document == updated_text
        assert fetched.metadata["version"] == 2
        for val in fetched.embedding:
            assert abs(val - 0.9) < 1e-5

    # ------------------------------------------------------------------
    # 12. Repeated ingestion does not create uncontrolled duplicates
    # ------------------------------------------------------------------

    def test_repeated_ingestion_idempotence(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_records: List[VectorRecord],
    ) -> None:
        """Ingesting the same batch multiple times does not increase record count."""
        vector_store.upsert_records(_TEST_COLLECTION, sample_vector_records)
        assert vector_store.count_records(_TEST_COLLECTION) == len(
            sample_vector_records
        )

        # Ingest again
        vector_store.upsert_records(_TEST_COLLECTION, sample_vector_records)
        assert vector_store.count_records(_TEST_COLLECTION) == len(
            sample_vector_records
        )

    # ------------------------------------------------------------------
    # 13. Duplicate/conflicting chunk IDs within the same batch
    # ------------------------------------------------------------------

    def test_duplicate_ids_within_batch_rejected(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """Submitting duplicate IDs in a single batch raises DuplicateRecordIdError."""
        dup_record = VectorRecord(
            id=sample_vector_record.id,
            embedding=list(_TEST_VECTOR),
            document="Conflicting duplicate text.",
            metadata={"ticker": "AAPL"},
        )
        with pytest.raises(DuplicateRecordIdError) as exc_info:
            vector_store.upsert_records(
                _TEST_COLLECTION, [sample_vector_record, dup_record]
            )
        assert sample_vector_record.id in exc_info.value.duplicate_ids

    # ------------------------------------------------------------------
    # 14. Empty batch handling
    # ------------------------------------------------------------------

    def test_empty_batch_handling(
        self,
        vector_store: ChromaVectorStore,
    ) -> None:
        """Empty batch returns a 0-record result without error."""
        result = vector_store.upsert_records(_TEST_COLLECTION, [])
        assert isinstance(result, VectorStoreInsertionResult)
        assert result.total_records == 0
        assert result.record_ids == []

    # ------------------------------------------------------------------
    # 15. Invalid embedding vector (empty or non-finite)
    # ------------------------------------------------------------------

    def test_empty_embedding_vector_rejected(self) -> None:
        """VectorRecord rejects empty embedding vector."""
        with pytest.raises(Exception):
            VectorRecord(
                id="bad-chunk",
                embedding=[],
                document="Valid text",
                metadata={},
            )

    def test_non_finite_embedding_vector_rejected(self) -> None:
        """VectorRecord rejects non-finite float values in embedding."""
        with pytest.raises(Exception):
            VectorRecord(
                id="bad-chunk",
                embedding=[float("nan"), 0.1],
                document="Valid text",
                metadata={},
            )

    # ------------------------------------------------------------------
    # 16. Inconsistent vector dimensions across batch
    # ------------------------------------------------------------------

    def test_inconsistent_dimensions_across_batch_rejected(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """VectorDimensionMismatchError raised on inconsistent vector lengths."""
        mismatch_record = VectorRecord(
            id="chunk-mismatch-dim",
            embedding=[0.1, 0.2, 0.3, 0.4],  # 4 dims != 8 dims
            document="Valid document text.",
            metadata={},
        )
        with pytest.raises(VectorDimensionMismatchError):
            vector_store.upsert_records(
                _TEST_COLLECTION, [sample_vector_record, mismatch_record]
            )

    # ------------------------------------------------------------------
    # 17. Invalid metadata rejected
    # ------------------------------------------------------------------

    def test_non_primitive_metadata_rejected(self) -> None:
        """VectorRecord rejects complex nested objects in metadata."""
        with pytest.raises(Exception):
            VectorRecord(
                id="bad-meta",
                embedding=list(_TEST_VECTOR),
                document="Valid document text.",
                metadata={"nested": {"key": "val"}},  # type: ignore[dict-item]
            )

    # ------------------------------------------------------------------
    # 18. Invalid collection name rejected
    # ------------------------------------------------------------------

    def test_invalid_collection_name_rejected(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """Invalid collection name raises InvalidCollectionNameError."""
        with pytest.raises(InvalidCollectionNameError):
            vector_store.upsert_records("bad..name", [sample_vector_record])

        with pytest.raises(InvalidCollectionNameError):
            vector_store.upsert_records("a", [sample_vector_record])  # too short (< 3)

        with pytest.raises(InvalidCollectionNameError):
            validate_collection_name("")

    # ------------------------------------------------------------------
    # 19. Storage/database error handling
    # ------------------------------------------------------------------

    def test_collection_not_found_error(
        self,
        vector_store: ChromaVectorStore,
    ) -> None:
        """Accessing non-existent collection raises CollectionNotFoundError."""
        with pytest.raises(CollectionNotFoundError):
            vector_store.count_records("finpilot_nonexistent_collection")

    def test_delete_collection_behavior(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_record: VectorRecord,
    ) -> None:
        """delete_collection removes collection and returns boolean."""
        vector_store.upsert_records(_TEST_COLLECTION, [sample_vector_record])
        assert vector_store.has_collection(_TEST_COLLECTION)

        assert vector_store.delete_collection(_TEST_COLLECTION) is True
        assert not vector_store.has_collection(_TEST_COLLECTION)
        assert vector_store.delete_collection(_TEST_COLLECTION) is False

    def test_delete_records_behavior(
        self,
        vector_store: ChromaVectorStore,
        sample_vector_records: List[VectorRecord],
    ) -> None:
        """delete_records removes specified IDs from the collection."""
        vector_store.upsert_records(_TEST_COLLECTION, sample_vector_records)
        assert vector_store.count_records(_TEST_COLLECTION) == 3

        deleted = vector_store.delete_records(
            _TEST_COLLECTION, [sample_vector_records[0].id]
        )
        assert deleted == 1
        assert vector_store.count_records(_TEST_COLLECTION) == 2
        assert (
            vector_store.get_record(_TEST_COLLECTION, sample_vector_records[0].id)
            is None
        )

    # ------------------------------------------------------------------
    # 20. Regression compatibility with Phase 9.1–9.6
    # ------------------------------------------------------------------

    def test_chunk_embedding_to_vector_record_conversion(self) -> None:
        """chunk_embedding_to_vector_record preserves all provenance fields."""
        meta = ChunkMetadata(
            chunk_id="chunk-conv-1",
            document_id=_TEST_DOC_ID,
            source_document="apple_2024.pdf",
            ticker=_TEST_TICKER,
            document_type=DocumentType.ANNUAL_REPORT,
            page_numbers=[1],
            start_page=1,
            end_page=1,
            upload_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            chunk_index=1,
            character_count=40,
            word_count=6,
        )
        emb_res = ChunkEmbeddingResult(
            chunk_id="chunk-conv-1",
            document_id=_TEST_DOC_ID,
            ticker=_TEST_TICKER,
            document_type=DocumentType.ANNUAL_REPORT,
            page_numbers=[1],
            start_page=1,
            end_page=1,
            chunk_index=1,
            metadata=meta,
            text="Operating profit increased by 12 percent.",
            embedding=list(_TEST_VECTOR),
            embedding_model="mock-embed",
            embedding_dimensions=_TEST_DIMS,
        )
        rec = chunk_embedding_to_vector_record(emb_res)
        assert rec.id == "chunk-conv-1"
        assert rec.document == "Operating profit increased by 12 percent."
        assert rec.metadata["ticker"] == _TEST_TICKER
        assert rec.metadata["document_id"] == _TEST_DOC_ID
        assert rec.metadata["source_document"] == "apple_2024.pdf"

    # ------------------------------------------------------------------
    # 21. End-to-end: DocumentChunk -> Embedder -> ChromaVectorStore
    # ------------------------------------------------------------------

    def test_end_to_end_chunk_to_chroma_storage(
        self,
        vector_store: ChromaVectorStore,
    ) -> None:
        """Full flow: DocumentChunk -> DocumentEmbedder -> store_document_embeddings."""
        # 1. Source DocumentChunk with ChunkMetadata (Phase 9.4 / 9.5)
        meta = ChunkMetadata(
            chunk_id="chunk-e2e-001",
            document_id=_TEST_DOC_ID,
            source_document="apple_10k.pdf",
            ticker=_TEST_TICKER,
            document_type=DocumentType.ANNUAL_REPORT,
            page_numbers=[1],
            start_page=1,
            end_page=1,
            upload_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            chunk_index=1,
            section_name="Executive Summary",
            character_count=52,
            word_count=8,
        )
        chunk = DocumentChunk(
            chunk_id="chunk-e2e-001",
            document_id=_TEST_DOC_ID,
            ticker=_TEST_TICKER,
            document_type=DocumentType.ANNUAL_REPORT,
            chunk_index=1,
            text="Net sales for fiscal year 2024 reached new highs.",
            character_count=52,
            word_count=8,
            page_numbers=[1],
            start_page=1,
            end_page=1,
            section_name="Executive Summary",
            metadata=meta,
        )
        chunked_doc = ChunkedDocument(
            document_id=_TEST_DOC_ID,
            ticker=_TEST_TICKER,
            document_type=DocumentType.ANNUAL_REPORT,
            total_chunks=1,
            total_characters=52,
            chunk_size=512,
            chunk_overlap=64,
            chunks=[chunk],
        )

        # 2. Embedding generation (Phase 9.6)
        provider = MockTestEmbeddingProvider()
        embedder = DocumentEmbedder(provider=provider)
        batch = embedder.embed_document(chunked_doc)
        assert batch.total_chunks == 1

        # 3. Storage into ChromaDB (Phase 9.7)
        result = store_document_embeddings(
            batch=batch,
            vector_store=vector_store,
            chunks=chunked_doc.chunks,
        )
        assert result.total_records == 1
        expected_col = build_document_collection_name(_TEST_TICKER, _TEST_DOC_ID)
        assert result.collection_name == expected_col

        # 4. Verify stored content
        stored = vector_store.get_record(expected_col, "chunk-e2e-001")
        assert stored is not None
        assert stored.id == "chunk-e2e-001"
        assert stored.document == "Net sales for fiscal year 2024 reached new highs."
        assert stored.metadata["ticker"] == _TEST_TICKER
        assert stored.metadata["document_id"] == _TEST_DOC_ID
        assert stored.metadata["source_document"] == "apple_10k.pdf"
        assert stored.metadata["section_name"] == "Executive Summary"
        assert len(stored.embedding) == _TEST_DIMS

    # ------------------------------------------------------------------
    # Factory test
    # ------------------------------------------------------------------

    def test_get_vector_store_factory(
        self, ephemeral_chroma_client: chromadb.ClientAPI
    ) -> None:
        """get_vector_store returns ChromaVectorStore for default config."""
        fake_settings = MagicMock()
        fake_settings.VECTOR_STORE_PROVIDER = "chroma"
        fake_settings.CHROMA_PERSIST_DIRECTORY = "./chroma_data"
        store = get_vector_store(settings=fake_settings, client=ephemeral_chroma_client)
        assert isinstance(store, ChromaVectorStore)

    def test_get_vector_store_unsupported_provider(self) -> None:
        """get_vector_store raises VectorStoreError for unknown provider."""
        fake_settings = MagicMock()
        fake_settings.VECTOR_STORE_PROVIDER = "unsupported_store"
        with pytest.raises(VectorStoreError):
            get_vector_store(settings=fake_settings)
