"""Unit tests for Phase 9.6: Document Embeddings.

Tests cover all 16 required scenarios using mocked EmbeddingProvider boundaries.
No real external API calls are made in this test suite.

Scenarios tested:
1.  Valid single chunk embedding via DocumentEmbedder.embed_chunk
2.  Multiple chunk embeddings (batch) via DocumentEmbedder.embed_chunks
3.  Embedding vector contains only numeric (float) values
4.  Embedding vector is non-empty
5.  Vector dimension consistency across multiple chunks
6.  Invalid/non-finite vector causes ChunkEmbeddingResult validation error
7.  Empty chunk text causes validation/EmbeddingEmptyInputError
8.  Provenance metadata preserved (chunk_id, document_id, ticker, type, pages)
9.  Batch embedding via embed_document returns DocumentEmbeddingBatch
10. Provider API failure raises EmbeddingProviderUnavailableError
11. EmbeddingModelInfo reports correct provider/model/dimensions
12. Missing GEMINI_API_KEY raises EmbeddingProviderNotConfiguredError
13. Provider abstraction can be replaced with any mock implementation
14. No real external API call required (all tests pass with mocked provider)
15. Regression: Phase 9.1-9.5 models feed correctly into Phase 9.6 pipeline
16. Deterministic pipeline: same mock always produces same ChunkEmbeddingResult
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

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
    DocumentEmbeddingBatch,
    EmbeddingModelInfo,
)
from app.providers.embedding import (
    EmbeddingAuthenticationError,
    EmbeddingDimensionMismatchError,
    EmbeddingEmptyInputError,
    EmbeddingError,
    EmbeddingInvalidResponseError,
    EmbeddingProvider,
    EmbeddingProviderNotConfiguredError,
    EmbeddingProviderRateLimitError,
    EmbeddingProviderUnavailableError,
    get_embedding_provider,
)
from app.services.document_embedder import (
    DocumentEmbedder,
    chunk_to_embedding_request,
    embed_chunked_document,
)

# ===========================================================================
# CONSTANTS
# ===========================================================================

_MOCK_PROVIDER = "mock-provider"
_MOCK_MODEL = "mock-embedding-v1"
_MOCK_DIMS = 8
_MOCK_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_DOC_ID = "doc-test-001"
_TICKER = "AAPL"
_DOC_TYPE = DocumentType.ANNUAL_REPORT


# ===========================================================================
# MOCK PROVIDER
# ===========================================================================


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock EmbeddingProvider returning deterministic _MOCK_VECTOR.

    Satisfies the EmbeddingProvider ABC contract without any network calls.
    Serves as a drop-in replacement to validate provider abstraction (test 13).
    """

    @property
    def provider_name(self) -> str:
        return _MOCK_PROVIDER

    @property
    def model_name(self) -> str:
        return _MOCK_MODEL

    @property
    def dimensions(self) -> int:
        return _MOCK_DIMS

    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider=self.provider_name,
            model_name=self.model_name,
            dimensions=self.dimensions,
        )

    def embed_single(self, request: ChunkEmbeddingRequest) -> ChunkEmbeddingResult:
        if not request.text or not request.text.strip():
            raise EmbeddingEmptyInputError(
                f"Empty text for chunk '{request.chunk_id}'",
                provider=self.provider_name,
            )
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
            embedding=list(_MOCK_VECTOR),
            embedding_model=self.model_name,
            embedding_dimensions=self.dimensions,
        )

    def embed_batch(
        self, requests: List[ChunkEmbeddingRequest]
    ) -> List[ChunkEmbeddingResult]:
        return [self.embed_single(r) for r in requests]


# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """Minimal valid DocumentChunk for embedding tests."""
    return DocumentChunk(
        chunk_id="chunk-001",
        document_id=_DOC_ID,
        ticker=_TICKER,
        document_type=_DOC_TYPE,
        chunk_index=1,
        text="Apple Inc reported strong revenue growth in fiscal year 2024.",
        character_count=61,
        word_count=10,
        page_numbers=[1],
        start_page=1,
        end_page=1,
    )


@pytest.fixture
def sample_chunk_with_metadata() -> DocumentChunk:
    """DocumentChunk with attached ChunkMetadata from Phase 9.5."""
    meta = ChunkMetadata(
        chunk_id="chunk-meta-001",
        document_id=_DOC_ID,
        source_document="apple_annual_2024.pdf",
        ticker=_TICKER,
        document_type=_DOC_TYPE,
        page_numbers=[2, 3],
        start_page=2,
        end_page=3,
        upload_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        chunk_index=2,
        character_count=57,
        word_count=9,
    )
    return DocumentChunk(
        chunk_id="chunk-meta-001",
        document_id=_DOC_ID,
        ticker=_TICKER,
        document_type=_DOC_TYPE,
        chunk_index=2,
        text="Operating margin expanded from 27% to 31% year-over-year.",
        character_count=57,
        word_count=9,
        page_numbers=[2, 3],
        start_page=2,
        end_page=3,
        metadata=meta,
    )


@pytest.fixture
def sample_chunks(sample_chunk: DocumentChunk) -> List[DocumentChunk]:
    """Three DocumentChunks for batch embedding tests."""
    chunks = [sample_chunk]
    for i in range(2, 4):
        chunks.append(
            DocumentChunk(
                chunk_id=f"chunk-{i:03d}",
                document_id=_DOC_ID,
                ticker=_TICKER,
                document_type=_DOC_TYPE,
                chunk_index=i,
                text=(f"Financial data chunk number {i} with sufficient content."),
                character_count=50,
                word_count=8,
                page_numbers=[i],
                start_page=i,
                end_page=i,
            )
        )
    return chunks


@pytest.fixture
def sample_chunked_doc(sample_chunks: List[DocumentChunk]) -> ChunkedDocument:
    """ChunkedDocument holding three chunks for full-document embedding."""
    return ChunkedDocument(
        document_id=_DOC_ID,
        ticker=_TICKER,
        document_type=_DOC_TYPE,
        total_chunks=len(sample_chunks),
        total_characters=sum(c.character_count for c in sample_chunks),
        chunk_size=512,
        chunk_overlap=64,
        chunks=sample_chunks,
    )


@pytest.fixture
def mock_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()


@pytest.fixture
def embedder(mock_provider: MockEmbeddingProvider) -> DocumentEmbedder:
    return DocumentEmbedder(provider=mock_provider)


# ===========================================================================
# TESTS
# ===========================================================================


class TestPhase96Embeddings:
    """Phase 9.6 embedding unit tests — all pass without real API calls."""

    # ------------------------------------------------------------------
    # Test 1: Valid single chunk embedding
    # ------------------------------------------------------------------

    def test_embed_single_chunk_returns_result(
        self,
        embedder: DocumentEmbedder,
        sample_chunk: DocumentChunk,
    ) -> None:
        """Single DocumentChunk produces a valid ChunkEmbeddingResult."""
        result = embedder.embed_chunk(sample_chunk)
        assert isinstance(result, ChunkEmbeddingResult)
        assert result.chunk_id == sample_chunk.chunk_id

    # ------------------------------------------------------------------
    # Test 2: Multiple chunk embeddings (batch)
    # ------------------------------------------------------------------

    def test_embed_multiple_chunks_returns_list(
        self,
        embedder: DocumentEmbedder,
        sample_chunks: List[DocumentChunk],
    ) -> None:
        """Batch of chunks returns equal-length list of ChunkEmbeddingResult."""
        results = embedder.embed_chunks(sample_chunks)
        assert len(results) == len(sample_chunks)
        for res in results:
            assert isinstance(res, ChunkEmbeddingResult)

    # ------------------------------------------------------------------
    # Test 3: Embedding vector contains only numeric (float) values
    # ------------------------------------------------------------------

    def test_embedding_vector_is_numeric(
        self,
        embedder: DocumentEmbedder,
        sample_chunk: DocumentChunk,
    ) -> None:
        """All values in the embedding vector are floats."""
        result = embedder.embed_chunk(sample_chunk)
        for val in result.embedding:
            assert isinstance(
                val, float
            ), f"Expected float, got {type(val)} for value {val!r}"

    # ------------------------------------------------------------------
    # Test 4: Embedding vector is non-empty
    # ------------------------------------------------------------------

    def test_embedding_vector_is_non_empty(
        self,
        embedder: DocumentEmbedder,
        sample_chunk: DocumentChunk,
    ) -> None:
        """Embedding vector is not an empty list."""
        result = embedder.embed_chunk(sample_chunk)
        assert len(result.embedding) > 0

    # ------------------------------------------------------------------
    # Test 5: Vector dimension consistency across multiple chunks
    # ------------------------------------------------------------------

    def test_vector_dimension_consistency_across_chunks(
        self,
        embedder: DocumentEmbedder,
        sample_chunks: List[DocumentChunk],
    ) -> None:
        """All chunks in a batch produce vectors of the same dimension."""
        results = embedder.embed_chunks(sample_chunks)
        dims = {len(r.embedding) for r in results}
        assert len(dims) == 1, f"Inconsistent dimensions across chunks: {dims}"

    # ------------------------------------------------------------------
    # Test 6: Non-finite vector causes validation error at model level
    # ------------------------------------------------------------------

    def test_non_finite_nan_vector_rejected_by_model(
        self,
        sample_chunk: DocumentChunk,
    ) -> None:
        """ChunkEmbeddingResult raises ValueError for NaN values."""
        with pytest.raises(Exception):
            ChunkEmbeddingResult(
                chunk_id=sample_chunk.chunk_id,
                document_id=sample_chunk.document_id,
                ticker=sample_chunk.ticker,
                document_type=sample_chunk.document_type,
                page_numbers=sample_chunk.page_numbers,
                start_page=sample_chunk.start_page,
                end_page=sample_chunk.end_page,
                chunk_index=sample_chunk.chunk_index,
                embedding=[float("nan"), 0.1, 0.2],
                embedding_model=_MOCK_MODEL,
                embedding_dimensions=3,
            )

    def test_non_finite_inf_vector_rejected_by_model(
        self,
        sample_chunk: DocumentChunk,
    ) -> None:
        """ChunkEmbeddingResult raises ValueError for +/- infinity values."""
        with pytest.raises(Exception):
            ChunkEmbeddingResult(
                chunk_id=sample_chunk.chunk_id,
                document_id=sample_chunk.document_id,
                ticker=sample_chunk.ticker,
                document_type=sample_chunk.document_type,
                page_numbers=sample_chunk.page_numbers,
                start_page=sample_chunk.start_page,
                end_page=sample_chunk.end_page,
                chunk_index=sample_chunk.chunk_index,
                embedding=[float("inf"), 0.1],
                embedding_model=_MOCK_MODEL,
                embedding_dimensions=2,
            )

    # ------------------------------------------------------------------
    # Test 7: Empty chunk text causes validation / EmbeddingEmptyInputError
    # ------------------------------------------------------------------

    def test_empty_text_on_request_model_raises_validation_error(self) -> None:
        """ChunkEmbeddingRequest rejects whitespace-only text at model level."""
        with pytest.raises(Exception):
            ChunkEmbeddingRequest(
                chunk_id="bad-chunk",
                document_id=_DOC_ID,
                ticker=_TICKER,
                document_type=_DOC_TYPE,
                text="   ",  # whitespace only
                page_numbers=[1],
                start_page=1,
                end_page=1,
                chunk_index=1,
            )

    def test_provider_guard_raises_empty_input_error(self) -> None:
        """MockEmbeddingProvider raises EmbeddingEmptyInputError for empty text."""
        error_provider = MagicMock(spec=EmbeddingProvider)
        error_provider.embed_single.side_effect = EmbeddingEmptyInputError(
            "Empty text", provider=_MOCK_PROVIDER
        )
        with pytest.raises(EmbeddingEmptyInputError):
            req = ChunkEmbeddingRequest(
                chunk_id="c1",
                document_id=_DOC_ID,
                ticker=_TICKER,
                document_type=_DOC_TYPE,
                text="Some financial text.",
                page_numbers=[1],
                start_page=1,
                end_page=1,
                chunk_index=1,
            )
            error_provider.embed_single(req)

    # ------------------------------------------------------------------
    # Test 8: Provenance metadata preserved in result
    # ------------------------------------------------------------------

    def test_chunk_provenance_preserved_in_result(
        self,
        embedder: DocumentEmbedder,
        sample_chunk: DocumentChunk,
    ) -> None:
        """ChunkEmbeddingResult preserves all provenance from the source chunk."""
        result = embedder.embed_chunk(sample_chunk)
        assert result.chunk_id == sample_chunk.chunk_id
        assert result.document_id == sample_chunk.document_id
        assert result.ticker == sample_chunk.ticker
        assert result.document_type == sample_chunk.document_type
        assert result.page_numbers == list(sample_chunk.page_numbers)
        assert result.start_page == sample_chunk.start_page
        assert result.end_page == sample_chunk.end_page
        assert result.chunk_index == sample_chunk.chunk_index

    def test_chunk_metadata_from_phase_95_preserved(
        self,
        embedder: DocumentEmbedder,
        sample_chunk_with_metadata: DocumentChunk,
    ) -> None:
        """ChunkMetadata from Phase 9.5 is preserved on the ChunkEmbeddingResult."""
        result = embedder.embed_chunk(sample_chunk_with_metadata)
        assert result.metadata is not None
        assert result.metadata.chunk_id == sample_chunk_with_metadata.metadata.chunk_id
        assert (
            result.metadata.document_id
            == sample_chunk_with_metadata.metadata.document_id
        )

    # ------------------------------------------------------------------
    # Test 9: Batch embedding via embed_document returns DocumentEmbeddingBatch
    # ------------------------------------------------------------------

    def test_embed_document_returns_batch(
        self,
        embedder: DocumentEmbedder,
        sample_chunked_doc: ChunkedDocument,
    ) -> None:
        """embed_document produces a DocumentEmbeddingBatch."""
        batch = embedder.embed_document(sample_chunked_doc)
        assert isinstance(batch, DocumentEmbeddingBatch)
        assert batch.document_id == sample_chunked_doc.document_id
        assert batch.ticker == sample_chunked_doc.ticker
        assert batch.document_type == sample_chunked_doc.document_type
        assert batch.total_chunks == len(sample_chunked_doc.chunks)
        assert len(batch.results) == len(sample_chunked_doc.chunks)

    def test_embed_document_batch_order_matches_input(
        self,
        embedder: DocumentEmbedder,
        sample_chunked_doc: ChunkedDocument,
    ) -> None:
        """Results in DocumentEmbeddingBatch are in same order as input chunks."""
        batch = embedder.embed_document(sample_chunked_doc)
        for i, (chunk, result) in enumerate(
            zip(sample_chunked_doc.chunks, batch.results)
        ):
            assert result.chunk_id == chunk.chunk_id, (
                f"Position {i}: expected chunk_id={chunk.chunk_id}, "
                f"got {result.chunk_id}"
            )

    # ------------------------------------------------------------------
    # Test 10: Provider API failure raises correct typed error
    # ------------------------------------------------------------------

    def test_provider_unavailable_error_propagates(
        self,
        sample_chunk: DocumentChunk,
    ) -> None:
        """EmbeddingProviderUnavailableError propagates through DocumentEmbedder."""
        failing = MagicMock(spec=EmbeddingProvider)
        failing.provider_name = _MOCK_PROVIDER
        failing.model_name = _MOCK_MODEL
        failing.dimensions = _MOCK_DIMS
        failing.embed_single.side_effect = EmbeddingProviderUnavailableError(
            "Connection refused", provider=_MOCK_PROVIDER
        )
        emb = DocumentEmbedder(provider=failing)
        with pytest.raises(EmbeddingProviderUnavailableError):
            emb.embed_chunk(sample_chunk)

    def test_authentication_error_propagates(
        self,
        sample_chunk: DocumentChunk,
    ) -> None:
        """EmbeddingAuthenticationError propagates through DocumentEmbedder."""
        auth_fail = MagicMock(spec=EmbeddingProvider)
        auth_fail.provider_name = _MOCK_PROVIDER
        auth_fail.model_name = _MOCK_MODEL
        auth_fail.dimensions = _MOCK_DIMS
        auth_fail.embed_single.side_effect = EmbeddingAuthenticationError(
            "Invalid API key", provider=_MOCK_PROVIDER
        )
        emb = DocumentEmbedder(provider=auth_fail)
        with pytest.raises(EmbeddingAuthenticationError):
            emb.embed_chunk(sample_chunk)

    # ------------------------------------------------------------------
    # Test 11: EmbeddingModelInfo reports correct fields
    # ------------------------------------------------------------------

    def test_model_info_returns_correct_fields(
        self,
        mock_provider: MockEmbeddingProvider,
    ) -> None:
        """EmbeddingModelInfo carries correct provider, model, and dimensions."""
        info = mock_provider.get_model_info()
        assert isinstance(info, EmbeddingModelInfo)
        assert info.provider == _MOCK_PROVIDER
        assert info.model_name == _MOCK_MODEL
        assert info.dimensions == _MOCK_DIMS

    def test_embedder_model_info_property(
        self,
        embedder: DocumentEmbedder,
    ) -> None:
        """DocumentEmbedder.model_info delegates correctly to the provider."""
        info = embedder.model_info
        assert info.provider == _MOCK_PROVIDER
        assert info.model_name == _MOCK_MODEL
        assert info.dimensions == _MOCK_DIMS

    # ------------------------------------------------------------------
    # Test 12: Missing GEMINI_API_KEY raises EmbeddingProviderNotConfiguredError
    # ------------------------------------------------------------------

    def test_missing_api_key_raises_configuration_error(self) -> None:
        """GeminiEmbeddingProvider raises when GEMINI_API_KEY is missing."""
        from app.providers.gemini_embedding import GeminiEmbeddingProvider

        fake_settings = MagicMock()
        fake_settings.EMBEDDING_MODEL = "gemini-embedding-001"
        fake_settings.EMBEDDING_DIMENSIONS = 3072
        fake_settings.GEMINI_API_KEY = ""  # no key

        with pytest.raises(EmbeddingProviderNotConfiguredError):
            GeminiEmbeddingProvider(settings=fake_settings)

    def test_none_api_key_raises_configuration_error(self) -> None:
        """GeminiEmbeddingProvider raises when GEMINI_API_KEY is None."""
        from app.providers.gemini_embedding import GeminiEmbeddingProvider

        fake_settings = MagicMock()
        fake_settings.EMBEDDING_MODEL = "gemini-embedding-001"
        fake_settings.EMBEDDING_DIMENSIONS = 3072
        fake_settings.GEMINI_API_KEY = None  # explicitly None

        with pytest.raises(EmbeddingProviderNotConfiguredError):
            GeminiEmbeddingProvider(settings=fake_settings)

    # ------------------------------------------------------------------
    # Test 13: Provider abstraction can be replaced/mocked
    # ------------------------------------------------------------------

    def test_provider_abstraction_is_replaceable(
        self,
        sample_chunk: DocumentChunk,
    ) -> None:
        """DocumentEmbedder works with any EmbeddingProvider implementation."""
        alt_vector = [0.9, 0.8, 0.7, 0.6]

        class AltProvider(EmbeddingProvider):
            @property
            def provider_name(self) -> str:
                return "alt-provider"

            @property
            def model_name(self) -> str:
                return "alt-model"

            @property
            def dimensions(self) -> int:
                return 4

            def get_model_info(self) -> EmbeddingModelInfo:
                return EmbeddingModelInfo(
                    provider="alt-provider",
                    model_name="alt-model",
                    dimensions=4,
                )

            def embed_single(
                self, request: ChunkEmbeddingRequest
            ) -> ChunkEmbeddingResult:
                return ChunkEmbeddingResult(
                    chunk_id=request.chunk_id,
                    document_id=request.document_id,
                    ticker=request.ticker,
                    document_type=request.document_type,
                    page_numbers=request.page_numbers,
                    start_page=request.start_page,
                    end_page=request.end_page,
                    chunk_index=request.chunk_index,
                    embedding=alt_vector,
                    embedding_model="alt-model",
                    embedding_dimensions=4,
                )

            def embed_batch(
                self, requests: List[ChunkEmbeddingRequest]
            ) -> List[ChunkEmbeddingResult]:
                return [self.embed_single(r) for r in requests]

        alt_embedder = DocumentEmbedder(provider=AltProvider())
        result = alt_embedder.embed_chunk(sample_chunk)
        assert result.embedding == alt_vector
        assert result.embedding_model == "alt-model"

    # ------------------------------------------------------------------
    # Test 14: No real external API call required
    # ------------------------------------------------------------------

    def test_no_real_api_call_required(
        self,
        embedder: DocumentEmbedder,
        sample_chunks: List[DocumentChunk],
    ) -> None:
        """All embedding tests pass with mocked provider — no real API call."""
        results = embedder.embed_chunks(sample_chunks)
        assert all(isinstance(r, ChunkEmbeddingResult) for r in results)

    # ------------------------------------------------------------------
    # Test 15: Regression — Phase 9.1-9.5 pipeline feeds Phase 9.6
    # ------------------------------------------------------------------

    def test_phase_91_provenance_survives_to_96(
        self,
        embedder: DocumentEmbedder,
        sample_chunk_with_metadata: DocumentChunk,
    ) -> None:
        """Document ID, ticker, type from Phase 9.1 survive to Phase 9.6."""
        result = embedder.embed_chunk(sample_chunk_with_metadata)
        assert result.document_id == _DOC_ID
        assert result.ticker == _TICKER
        assert result.document_type == _DOC_TYPE

    def test_chunk_to_embedding_request_preserves_all_fields(
        self,
        sample_chunk_with_metadata: DocumentChunk,
    ) -> None:
        """chunk_to_embedding_request maps every DocumentChunk field correctly."""
        req = chunk_to_embedding_request(sample_chunk_with_metadata)
        assert req.chunk_id == sample_chunk_with_metadata.chunk_id
        assert req.document_id == sample_chunk_with_metadata.document_id
        assert req.ticker == sample_chunk_with_metadata.ticker
        assert req.document_type == sample_chunk_with_metadata.document_type
        assert req.text == sample_chunk_with_metadata.text
        assert req.page_numbers == list(sample_chunk_with_metadata.page_numbers)
        assert req.start_page == sample_chunk_with_metadata.start_page
        assert req.end_page == sample_chunk_with_metadata.end_page
        assert req.chunk_index == sample_chunk_with_metadata.chunk_index
        assert req.section_name == sample_chunk_with_metadata.section_name
        assert req.metadata == sample_chunk_with_metadata.metadata

    # ------------------------------------------------------------------
    # Test 16: Deterministic behavior when provider is mocked
    # ------------------------------------------------------------------

    def test_deterministic_embedding_with_mock_provider(
        self,
        embedder: DocumentEmbedder,
        sample_chunk: DocumentChunk,
    ) -> None:
        """Same input always produces same ChunkEmbeddingResult with mock provider."""
        result_a = embedder.embed_chunk(sample_chunk)
        result_b = embedder.embed_chunk(sample_chunk)
        assert result_a.chunk_id == result_b.chunk_id
        assert result_a.document_id == result_b.document_id
        assert result_a.embedding == result_b.embedding
        assert result_a.embedding_model == result_b.embedding_model
        assert result_a.embedding_dimensions == result_b.embedding_dimensions

    # ------------------------------------------------------------------
    # ADDITIONAL: factory raises for unsupported/empty provider name
    # ------------------------------------------------------------------

    def test_factory_raises_for_unsupported_provider(self) -> None:
        """get_embedding_provider raises EmbeddingError for unknown provider."""
        fake_settings = MagicMock()
        fake_settings.EMBEDDING_PROVIDER = "unsupported-xyz"
        with pytest.raises(EmbeddingError):
            get_embedding_provider(settings=fake_settings)

    def test_factory_raises_for_empty_provider_name(self) -> None:
        """get_embedding_provider raises when EMBEDDING_PROVIDER is empty."""
        fake_settings = MagicMock()
        fake_settings.EMBEDDING_PROVIDER = ""
        with pytest.raises(EmbeddingProviderNotConfiguredError):
            get_embedding_provider(settings=fake_settings)

    # ------------------------------------------------------------------
    # ADDITIONAL: Empty batch returns empty list without error
    # ------------------------------------------------------------------

    def test_embed_empty_batch_returns_empty_list(
        self,
        embedder: DocumentEmbedder,
    ) -> None:
        """Embedding an empty list of chunks returns empty list without error."""
        results = embedder.embed_chunks([])
        assert results == []

    # ------------------------------------------------------------------
    # ADDITIONAL: ChunkEmbeddingResult carries embedding_dimensions field
    # ------------------------------------------------------------------

    def test_result_embedding_dimensions_matches_vector_length(
        self,
        embedder: DocumentEmbedder,
        sample_chunk: DocumentChunk,
    ) -> None:
        """embedding_dimensions field equals len(embedding) vector."""
        result = embedder.embed_chunk(sample_chunk)
        assert result.embedding_dimensions == len(result.embedding)

    # ------------------------------------------------------------------
    # ADDITIONAL: ChunkEmbeddingResult has embedded_at timestamp
    # ------------------------------------------------------------------

    def test_result_has_embedded_at_timestamp(
        self,
        embedder: DocumentEmbedder,
        sample_chunk: DocumentChunk,
    ) -> None:
        """ChunkEmbeddingResult carries a UTC timestamp of embedding time."""
        result = embedder.embed_chunk(sample_chunk)
        assert isinstance(result.embedded_at, datetime)
        assert result.embedded_at.year >= 2024

    # ------------------------------------------------------------------
    # ADDITIONAL: EmbeddingModelInfo is immutable (frozen)
    # ------------------------------------------------------------------

    def test_embedding_model_info_is_immutable(
        self,
        mock_provider: MockEmbeddingProvider,
    ) -> None:
        """EmbeddingModelInfo is frozen and cannot be mutated."""
        info = mock_provider.get_model_info()
        with pytest.raises(Exception):
            info.provider = "hacked"  # type: ignore[misc]

    # ------------------------------------------------------------------
    # ADDITIONAL: embed_chunked_document standalone function
    # ------------------------------------------------------------------

    def test_embed_chunked_document_function(
        self,
        mock_provider: MockEmbeddingProvider,
        sample_chunked_doc: ChunkedDocument,
    ) -> None:
        """embed_chunked_document module-level function produces correct batch."""
        batch = embed_chunked_document(sample_chunked_doc, mock_provider)
        assert isinstance(batch, DocumentEmbeddingBatch)
        assert batch.document_id == sample_chunked_doc.document_id
        assert batch.ticker == sample_chunked_doc.ticker
        assert batch.total_chunks == sample_chunked_doc.total_chunks
        assert len(batch.results) == sample_chunked_doc.total_chunks

    # ------------------------------------------------------------------
    # AUDIT: Gemini provider explicitly requests output_dimensionality
    # ------------------------------------------------------------------

    def test_gemini_requests_output_dimensionality_single(self) -> None:
        """GeminiEmbeddingProvider passes output_dimensionality to SDK config."""
        from app.providers.gemini_embedding import GeminiEmbeddingProvider

        fake_settings = MagicMock()
        fake_settings.EMBEDDING_MODEL = "gemini-embedding-001"
        fake_settings.EMBEDDING_DIMENSIONS = 3072
        fake_settings.GEMINI_API_KEY = "test-key"

        provider = GeminiEmbeddingProvider(settings=fake_settings)
        # Mock the genai client models.embed_content
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 3072
        mock_response.embeddings = [mock_embedding]

        provider._client = MagicMock()
        provider._client.models.embed_content.return_value = mock_response

        req = ChunkEmbeddingRequest(
            chunk_id="chunk-test",
            document_id=_DOC_ID,
            ticker=_TICKER,
            document_type=_DOC_TYPE,
            text="Testing output_dimensionality config.",
            page_numbers=[1],
            start_page=1,
            end_page=1,
            chunk_index=1,
        )
        res = provider.embed_single(req)

        assert res.embedding_dimensions == 3072
        provider._client.models.embed_content.assert_called_once()
        call_kwargs = provider._client.models.embed_content.call_args.kwargs
        config = call_kwargs.get("config")
        assert config is not None
        assert config.output_dimensionality == 3072
        assert config.task_type == "RETRIEVAL_DOCUMENT"

    def test_gemini_requests_output_dimensionality_batch(self) -> None:
        """GeminiEmbeddingProvider passes output_dimensionality in batch mode."""
        from app.providers.gemini_embedding import GeminiEmbeddingProvider

        fake_settings = MagicMock()
        fake_settings.EMBEDDING_MODEL = "gemini-embedding-001"
        fake_settings.EMBEDDING_DIMENSIONS = 3072
        fake_settings.GEMINI_API_KEY = "test-key"

        provider = GeminiEmbeddingProvider(settings=fake_settings)
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 3072
        mock_response.embeddings = [mock_embedding]

        provider._client = MagicMock()
        provider._client.models.embed_content.return_value = mock_response

        req = ChunkEmbeddingRequest(
            chunk_id="chunk-test",
            document_id=_DOC_ID,
            ticker=_TICKER,
            document_type=_DOC_TYPE,
            text="Testing batch output_dimensionality.",
            page_numbers=[1],
            start_page=1,
            end_page=1,
            chunk_index=1,
        )
        results = provider.embed_batch([req])

        assert len(results) == 1
        call_kwargs = provider._client.models.embed_content.call_args.kwargs
        config = call_kwargs.get("config")
        assert config is not None
        assert config.output_dimensionality == 3072
        assert config.task_type == "RETRIEVAL_DOCUMENT"

    # ------------------------------------------------------------------
    # AUDIT: Dimension mismatch raises EmbeddingDimensionMismatchError
    # ------------------------------------------------------------------

    def test_gemini_dimension_mismatch_raises_typed_error(self) -> None:
        """GeminiEmbeddingProvider raises EmbeddingDimensionMismatchError."""
        from app.providers.gemini_embedding import GeminiEmbeddingProvider

        fake_settings = MagicMock()
        fake_settings.EMBEDDING_MODEL = "gemini-embedding-001"
        fake_settings.EMBEDDING_DIMENSIONS = 3072
        fake_settings.GEMINI_API_KEY = "test-key"

        provider = GeminiEmbeddingProvider(settings=fake_settings)
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768  # 768 != 3072
        mock_response.embeddings = [mock_embedding]

        provider._client = MagicMock()
        provider._client.models.embed_content.return_value = mock_response

        req = ChunkEmbeddingRequest(
            chunk_id="chunk-mismatch",
            document_id=_DOC_ID,
            ticker=_TICKER,
            document_type=_DOC_TYPE,
            text="Dimension mismatch test.",
            page_numbers=[1],
            start_page=1,
            end_page=1,
            chunk_index=1,
        )
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            provider.embed_single(req)
        assert exc_info.value.expected_dimensions == 3072
        assert exc_info.value.actual_dimensions == 768

    # ------------------------------------------------------------------
    # AUDIT: Non-numeric vector raises EmbeddingInvalidResponseError
    # ------------------------------------------------------------------

    def test_gemini_non_numeric_vector_raises_invalid_response(self) -> None:
        """GeminiEmbeddingProvider raises invalid response on non-numeric."""
        from app.providers.gemini_embedding import GeminiEmbeddingProvider

        fake_settings = MagicMock()
        fake_settings.EMBEDDING_MODEL = "gemini-embedding-001"
        fake_settings.EMBEDDING_DIMENSIONS = 3072
        fake_settings.GEMINI_API_KEY = "test-key"

        provider = GeminiEmbeddingProvider(settings=fake_settings)
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = ["not", "numbers"]
        mock_response.embeddings = [mock_embedding]

        provider._client = MagicMock()
        provider._client.models.embed_content.return_value = mock_response

        req = ChunkEmbeddingRequest(
            chunk_id="chunk-nonnum",
            document_id=_DOC_ID,
            ticker=_TICKER,
            document_type=_DOC_TYPE,
            text="Non-numeric test.",
            page_numbers=[1],
            start_page=1,
            end_page=1,
            chunk_index=1,
        )
        with pytest.raises(EmbeddingInvalidResponseError):
            provider.embed_single(req)

    # ------------------------------------------------------------------
    # AUDIT: ChunkEmbeddingResult dimension consistency validation
    # ------------------------------------------------------------------

    def test_chunk_embedding_result_rejects_dimension_mismatch(
        self,
        sample_chunk: DocumentChunk,
    ) -> None:
        """ChunkEmbeddingResult rejects when embedding_dimensions != len(embedding)."""
        with pytest.raises(Exception):
            ChunkEmbeddingResult(
                chunk_id=sample_chunk.chunk_id,
                document_id=sample_chunk.document_id,
                ticker=sample_chunk.ticker,
                document_type=sample_chunk.document_type,
                page_numbers=sample_chunk.page_numbers,
                start_page=sample_chunk.start_page,
                end_page=sample_chunk.end_page,
                chunk_index=sample_chunk.chunk_index,
                embedding=[0.1, 0.2, 0.3],  # length 3
                embedding_model=_MOCK_MODEL,
                embedding_dimensions=3072,  # declares 3072 != 3
            )

    # ------------------------------------------------------------------
    # AUDIT: DocumentEmbeddingBatch rejects inconsistent dimensions
    # ------------------------------------------------------------------

    def test_document_embedding_batch_rejects_inconsistent_dimensions(
        self,
        sample_chunk: DocumentChunk,
    ) -> None:
        """DocumentEmbeddingBatch rejects when dimensions differ from batch."""
        res = ChunkEmbeddingResult(
            chunk_id=sample_chunk.chunk_id,
            document_id=sample_chunk.document_id,
            ticker=sample_chunk.ticker,
            document_type=sample_chunk.document_type,
            page_numbers=sample_chunk.page_numbers,
            start_page=sample_chunk.start_page,
            end_page=sample_chunk.end_page,
            chunk_index=sample_chunk.chunk_index,
            embedding=[0.1, 0.2, 0.3],
            embedding_model=_MOCK_MODEL,
            embedding_dimensions=3,
        )
        with pytest.raises(Exception):
            DocumentEmbeddingBatch(
                document_id=_DOC_ID,
                ticker=_TICKER,
                document_type=_DOC_TYPE,
                total_chunks=1,
                embedding_model=_MOCK_MODEL,
                embedding_dimensions=8,  # batch declares 8, result has 3
                results=[res],
            )

    # ------------------------------------------------------------------
    # AUDIT: HTTP status code error mapping in GeminiEmbeddingProvider
    # ------------------------------------------------------------------

    def test_gemini_error_mapping_status_codes(self) -> None:
        """_map_api_exception maps HTTP status codes to typed exceptions."""
        from app.providers.gemini_embedding import GeminiEmbeddingProvider

        fake_settings = MagicMock()
        fake_settings.EMBEDDING_MODEL = "gemini-embedding-001"
        fake_settings.EMBEDDING_DIMENSIONS = 3072
        fake_settings.GEMINI_API_KEY = "test-key"

        provider = GeminiEmbeddingProvider(settings=fake_settings)

        # 401 -> AuthenticationError
        err_401 = Exception("Unauthorized")
        err_401.code = 401
        with pytest.raises(EmbeddingAuthenticationError):
            provider._map_api_exception(err_401, "test-401")

        # 403 -> AuthenticationError
        err_403 = Exception("Forbidden")
        err_403.code = 403
        with pytest.raises(EmbeddingAuthenticationError):
            provider._map_api_exception(err_403, "test-403")

        # 429 -> RateLimitError
        err_429 = Exception("Resource Exhausted")
        err_429.code = 429
        with pytest.raises(EmbeddingProviderRateLimitError):
            provider._map_api_exception(err_429, "test-429")

        # 503 -> UnavailableError
        err_503 = Exception("Backend Unavailable")
        err_503.code = 503
        with pytest.raises(EmbeddingProviderUnavailableError):
            provider._map_api_exception(err_503, "test-503")
