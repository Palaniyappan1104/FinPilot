"""Unit and integration tests for Phase 9.9: Similarity Search.

Tests cover:
1. VectorSearchResult schema validation, distance semantics, and provenance accessors.
2. SimilaritySearchResults schema validation and immutability.
3. SimilaritySearchService with mock VectorStore.
4. Interoperability with Phase 9.8 QueryEmbeddingResult.
5. Convenience functions (search_similarity, search_document, search_company).
6. Input validation (empty query, NaN/Inf, non-numeric, invalid type).
7. Invalid collection name handling (InvalidCollectionNameError).
8. Missing collection handling (CollectionNotFoundError).
9. n_results validation (must be positive integer).
10. Query dimension mismatch detection (VectorDimensionMismatchError).
11. Provenance preservation from stored metadata.
12. Distance metric semantics (lower distance = closer match).
13. Metadata filtering using where clauses.
14. VectorStore error propagation.
15. End-to-end integration with in-memory EphemeralClient ChromaVectorStore.
16. Safe logging verification (no raw vectors or text in logs).
"""

import logging
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import chromadb
import pytest
from pydantic import ValidationError

from app.models.embeddings import QueryEmbeddingResult
from app.models.vector_store import (
    SimilaritySearchResults,
    VectorRecord,
    VectorSearchResult,
)
from app.services.similarity_search import (
    SimilaritySearchService,
    search_company_similarity,
    search_document_similarity,
    search_similarity,
)
from app.storage.chroma_vector_store import ChromaVectorStore
from app.storage.vector_base import VectorStore
from app.storage.vector_exceptions import (
    CollectionNotFoundError,
    InvalidCollectionNameError,
    VectorDimensionMismatchError,
    VectorStoreError,
    VectorValidationError,
)

# ---------------------------------------------------------------------------
# Test constants & Mock VectorStore
# ---------------------------------------------------------------------------

_TEST_COLLECTION = "finpilot_aapl_doc-2024"
_TEST_DIMS = 4
_SAMPLE_QUERY_VEC = [0.1, 0.2, 0.3, 0.4]


class MockVectorStore(VectorStore):
    """Isolated mock VectorStore for unit testing SimilaritySearchService."""

    def __init__(self, existing_collections: Optional[List[str]] = None) -> None:
        self._collections = set(existing_collections or [_TEST_COLLECTION])
        self.last_query_collection: Optional[str] = None
        self.last_query_embedding: Optional[List[float]] = None
        self.last_n_results: Optional[int] = None
        self.last_where: Optional[Dict[str, Any]] = None
        self.mock_results: List[VectorSearchResult] = []

    @property
    def store_name(self) -> str:
        return "mock_vector_store"

    def get_or_create_collection(self, collection_name: str) -> None:
        self._collections.add(collection_name)

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def delete_collection(self, collection_name: str) -> bool:
        if collection_name in self._collections:
            self._collections.remove(collection_name)
            return True
        return False

    def list_collections(self) -> List[str]:
        return sorted(self._collections)

    def upsert_records(self, collection_name: str, records: list) -> Any:
        raise NotImplementedError

    def count_records(self, collection_name: str) -> int:
        if collection_name not in self._collections:
            raise CollectionNotFoundError(collection_name)
        return len(self.mock_results)

    def get_record(self, collection_name: str, record_id: str) -> Any:
        raise NotImplementedError

    def get_records(self, collection_name: str, record_ids: list) -> list:
        raise NotImplementedError

    def delete_records(self, collection_name: str, record_ids: list) -> int:
        raise NotImplementedError

    def query_similarity(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> List[VectorSearchResult]:
        if collection_name not in self._collections:
            raise CollectionNotFoundError(collection_name)
        self.last_query_collection = collection_name
        self.last_query_embedding = query_embedding
        self.last_n_results = n_results
        self.last_where = where
        return self.mock_results


@pytest.fixture
def ephemeral_chroma_client():
    """Isolated in-memory ChromaDB client for testing, reset before each test."""
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.Client(ChromaSettings(is_persistent=False, allow_reset=True))
    client.reset()
    yield client
    client.reset()


# ===========================================================================
# 1. VectorSearchResult & SimilaritySearchResults Model Tests
# ===========================================================================


class TestVectorSearchResultModel:
    """Test VectorSearchResult schema, validation, and provenance accessors."""

    def test_valid_instantiation(self) -> None:
        result = VectorSearchResult(
            id="chunk-001",
            distance=0.15,
            document="Apple reported Q3 revenue of $85.8 billion.",
            metadata={
                "document_id": "doc-aapl-2024",
                "ticker": "AAPL",
                "document_type": "10-q",
                "page_numbers": "1,2",
                "start_page": 1,
                "end_page": 2,
                "section_name": "Management Discussion",
            },
            distance_metric="cosine_distance",
        )
        assert result.id == "chunk-001"
        assert result.distance == 0.15
        assert result.document_id == "doc-aapl-2024"
        assert result.ticker == "AAPL"
        assert result.document_type == "10-q"
        assert result.page_numbers == [1, 2]
        assert result.start_page == 1
        assert result.end_page == 2
        assert result.section_name == "Management Discussion"
        assert result.distance_metric == "cosine_distance"

    def test_page_numbers_fallback_to_start_and_end_page(self) -> None:
        result = VectorSearchResult(
            id="chunk-002",
            distance=0.20,
            document="Some text",
            metadata={"start_page": 3, "end_page": 5},
        )
        assert result.page_numbers == [3, 4, 5]

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VectorSearchResult(id="", distance=0.1, document="Text")

    def test_nan_distance_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VectorSearchResult(id="c1", distance=float("nan"), document="Text")

    def test_inf_distance_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VectorSearchResult(id="c1", distance=float("inf"), document="Text")

    def test_model_immutability(self) -> None:
        result = VectorSearchResult(id="chunk-001", distance=0.1, document="Text")
        with pytest.raises(ValidationError):
            result.distance = 0.5  # type: ignore[misc]


class TestSimilaritySearchResultsModel:
    """Test SimilaritySearchResults container model."""

    def test_valid_container(self) -> None:
        match = VectorSearchResult(id="c1", distance=0.05, document="Revenue grew")
        container = SimilaritySearchResults(
            collection_name="finpilot_msft_doc-2024",
            results=[match],
            total_results=1,
            distance_metric="cosine_distance",
        )
        assert container.collection_name == "finpilot_msft_doc-2024"
        assert container.total_results == 1
        assert len(container.results) == 1
        assert container.searched_at is not None

    def test_container_immutability(self) -> None:
        container = SimilaritySearchResults(
            collection_name="finpilot_msft_doc-2024",
            results=[],
            total_results=0,
        )
        with pytest.raises(ValidationError):
            container.total_results = 5  # type: ignore[misc]


# ===========================================================================
# 2. SimilaritySearchService Unit Tests (with MockVectorStore)
# ===========================================================================


class TestSimilaritySearchServiceUnit:
    """Unit tests for SimilaritySearchService logic using MockVectorStore."""

    def test_search_with_raw_float_list(self) -> None:
        store = MockVectorStore()
        store.mock_results = [
            VectorSearchResult(id="c1", distance=0.08, document="Quarterly revenue")
        ]
        service = SimilaritySearchService(vector_store=store)

        res = service.search(
            collection_name=_TEST_COLLECTION,
            query=_SAMPLE_QUERY_VEC,
            n_results=5,
        )

        assert isinstance(res, SimilaritySearchResults)
        assert res.total_results == 1
        assert res.collection_name == _TEST_COLLECTION
        assert store.last_query_collection == _TEST_COLLECTION
        assert store.last_query_embedding == _SAMPLE_QUERY_VEC
        assert store.last_n_results == 5

    def test_search_with_query_embedding_result_object(self) -> None:
        store = MockVectorStore()
        store.mock_results = [
            VectorSearchResult(id="c2", distance=0.12, document="Cash flow positive")
        ]
        service = SimilaritySearchService(vector_store=store)

        query_obj = QueryEmbeddingResult(
            query="Operating cash flow",
            query_id="qid-10",
            ticker="AAPL",
            embedding=_SAMPLE_QUERY_VEC,
            embedding_model="gemini-embedding-001",
            embedding_dimensions=_TEST_DIMS,
        )

        res = service.search(
            collection_name=_TEST_COLLECTION,
            query=query_obj,
            n_results=3,
        )

        assert res.total_results == 1
        assert res.results[0].id == "c2"
        assert store.last_query_embedding == _SAMPLE_QUERY_VEC

    def test_search_with_metadata_filter(self) -> None:
        store = MockVectorStore()
        service = SimilaritySearchService(vector_store=store)
        filter_dict = {"document_type": "10-k"}

        service.search(
            collection_name=_TEST_COLLECTION,
            query=_SAMPLE_QUERY_VEC,
            where=filter_dict,
        )

        assert store.last_where == filter_dict

    def test_search_document_helper_builds_correct_collection(self) -> None:
        store = MockVectorStore(existing_collections=["finpilot_nvda_doc-annual-2024"])
        service = SimilaritySearchService(vector_store=store)

        service.search_document(
            ticker="NVDA",
            document_id="doc-annual-2024",
            query=_SAMPLE_QUERY_VEC,
            n_results=4,
        )

        assert store.last_query_collection == "finpilot_nvda_doc-annual-2024"
        assert store.last_n_results == 4

    def test_search_company_helper_builds_correct_collection(self) -> None:
        store = MockVectorStore(existing_collections=["finpilot_company_goog"])
        service = SimilaritySearchService(vector_store=store)

        service.search_company(
            ticker="GOOG",
            query=_SAMPLE_QUERY_VEC,
            n_results=8,
        )

        assert store.last_query_collection == "finpilot_company_goog"
        assert store.last_n_results == 8

    def test_convenience_functions(self) -> None:
        store = MockVectorStore(
            existing_collections=[
                "finpilot_tsla_doc-q3",
                "finpilot_company_tsla",
                _TEST_COLLECTION,
            ]
        )

        res1 = search_similarity(
            query=_SAMPLE_QUERY_VEC,
            collection_name=_TEST_COLLECTION,
            vector_store=store,
        )
        assert res1.collection_name == _TEST_COLLECTION

        res2 = search_document_similarity(
            ticker="TSLA",
            document_id="doc-q3",
            query=_SAMPLE_QUERY_VEC,
            vector_store=store,
        )
        assert res2.collection_name == "finpilot_tsla_doc-q3"

        res3 = search_company_similarity(
            ticker="TSLA",
            query=_SAMPLE_QUERY_VEC,
            vector_store=store,
        )
        assert res3.collection_name == "finpilot_company_tsla"


# ===========================================================================
# 3. Input Validation & Error Handling
# ===========================================================================


class TestSimilaritySearchValidation:
    """Test validation and error handling for search inputs."""

    def test_empty_query_vector_rejected(self) -> None:
        service = SimilaritySearchService(vector_store=MockVectorStore())
        with pytest.raises(VectorValidationError) as exc_info:
            service.search(collection_name=_TEST_COLLECTION, query=[])
        assert "cannot be empty" in str(exc_info.value)

    def test_nan_in_query_vector_rejected(self) -> None:
        service = SimilaritySearchService(vector_store=MockVectorStore())
        with pytest.raises(VectorValidationError) as exc_info:
            service.search(
                collection_name=_TEST_COLLECTION,
                query=[0.1, float("nan"), 0.3, 0.4],
            )
        assert "non-finite" in str(exc_info.value)

    def test_inf_in_query_vector_rejected(self) -> None:
        service = SimilaritySearchService(vector_store=MockVectorStore())
        with pytest.raises(VectorValidationError) as exc_info:
            service.search(
                collection_name=_TEST_COLLECTION,
                query=[0.1, float("inf"), 0.3, 0.4],
            )
        assert "non-finite" in str(exc_info.value)

    def test_non_numeric_query_vector_rejected(self) -> None:
        service = SimilaritySearchService(vector_store=MockVectorStore())
        with pytest.raises(VectorValidationError) as exc_info:
            service.search(
                collection_name=_TEST_COLLECTION,
                query=["bad", "data"],  # type: ignore[arg-type]
            )
        assert "non-numeric" in str(exc_info.value)

    def test_invalid_query_type_rejected(self) -> None:
        service = SimilaritySearchService(vector_store=MockVectorStore())
        with pytest.raises(VectorValidationError) as exc_info:
            service.search(
                collection_name=_TEST_COLLECTION,
                query="raw string query",  # type: ignore[arg-type]
            )
        assert "must be a list of floats" in str(exc_info.value)

    def test_invalid_collection_name_rejected(self) -> None:
        service = SimilaritySearchService(vector_store=MockVectorStore())
        with pytest.raises(InvalidCollectionNameError):
            service.search(collection_name="INVALID!!NAME", query=_SAMPLE_QUERY_VEC)

    def test_non_existent_collection_raises_not_found(self) -> None:
        service = SimilaritySearchService(vector_store=MockVectorStore())
        with pytest.raises(CollectionNotFoundError) as exc_info:
            service.search(
                collection_name="finpilot_unknown_doc-1",
                query=_SAMPLE_QUERY_VEC,
            )
        assert "finpilot_unknown_doc-1" in str(exc_info.value)

    def test_invalid_n_results_zero_or_negative_rejected(self) -> None:
        service = SimilaritySearchService(vector_store=MockVectorStore())
        with pytest.raises(VectorValidationError):
            service.search(
                collection_name=_TEST_COLLECTION,
                query=_SAMPLE_QUERY_VEC,
                n_results=0,
            )
        with pytest.raises(VectorValidationError):
            service.search(
                collection_name=_TEST_COLLECTION,
                query=_SAMPLE_QUERY_VEC,
                n_results=-5,
            )

    def test_expected_dimensions_mismatch_raises_typed_error(self) -> None:
        service = SimilaritySearchService(
            vector_store=MockVectorStore(), expected_dimensions=4
        )
        with pytest.raises(VectorDimensionMismatchError) as exc_info:
            service.search(
                collection_name=_TEST_COLLECTION,
                query=[0.1, 0.2],  # 2 dims instead of 4
            )
        assert exc_info.value.expected_dim == 4
        assert exc_info.value.actual_dim == 2

    def test_underlying_vector_store_error_propagates(self) -> None:
        failing_store = MagicMock(spec=VectorStore)
        failing_store.has_collection.return_value = True
        failing_store.query_similarity.side_effect = VectorStoreError(
            "ChromaDB connection severed", collection=_TEST_COLLECTION
        )
        service = SimilaritySearchService(vector_store=failing_store)

        with pytest.raises(VectorStoreError) as exc_info:
            service.search(collection_name=_TEST_COLLECTION, query=_SAMPLE_QUERY_VEC)
        assert "connection severed" in str(exc_info.value)


# ===========================================================================
# 4. ChromaVectorStore Integration Tests (In-Memory EphemeralClient)
# ===========================================================================


class TestChromaVectorStoreSimilaritySearch:
    """Integration tests executing real similarity search against ChromaVectorStore."""

    def test_end_to_end_search_finds_nearest_neighbor(
        self, ephemeral_chroma_client: chromadb.ClientAPI
    ) -> None:
        store = ChromaVectorStore(client=ephemeral_chroma_client)
        col_name = "finpilot_test_search_e2e"
        store.get_or_create_collection(col_name)

        # Upsert 3 records with distinct embeddings
        # c1 is very close to [1.0, 0.0, 0.0, 0.0]
        # c2 is perpendicular [0.0, 1.0, 0.0, 0.0]
        # c3 is opposite [-1.0, 0.0, 0.0, 0.0]
        records = [
            VectorRecord(
                id="chunk-close",
                embedding=[0.99, 0.01, 0.0, 0.0],
                document="Apple gross margin expanded to 46.2%.",
                metadata={"document_id": "doc-1", "ticker": "AAPL", "page": 5},
            ),
            VectorRecord(
                id="chunk-orthogonal",
                embedding=[0.01, 0.99, 0.0, 0.0],
                document="R&D expenses increased by 8%.",
                metadata={"document_id": "doc-1", "ticker": "AAPL", "page": 12},
            ),
            VectorRecord(
                id="chunk-far",
                embedding=[-0.99, 0.0, 0.0, 0.0],
                document="Unrelated environmental statement.",
                metadata={"document_id": "doc-1", "ticker": "AAPL", "page": 40},
            ),
        ]
        store.upsert_records(col_name, records)

        # Query near chunk-close
        query_vec = [1.0, 0.0, 0.0, 0.0]
        results = store.query_similarity(
            collection_name=col_name, query_embedding=query_vec, n_results=3
        )

        assert len(results) == 3
        # First match must be chunk-close
        assert results[0].id == "chunk-close"
        assert "gross margin" in results[0].document
        assert results[0].document_id == "doc-1"
        assert results[0].ticker == "AAPL"

        # Distances must be monotonically non-decreasing (nearest first)
        assert results[0].distance <= results[1].distance <= results[2].distance
        assert results[0].distance < 0.1

    def test_search_with_where_filter_restricts_results(
        self, ephemeral_chroma_client: chromadb.ClientAPI
    ) -> None:
        store = ChromaVectorStore(client=ephemeral_chroma_client)
        col_name = "finpilot_test_filter_e2e"
        store.get_or_create_collection(col_name)

        records = [
            VectorRecord(
                id="chunk-10k",
                embedding=[0.5, 0.5, 0.0, 0.0],
                document="Annual Report 10-K text",
                metadata={"document_type": "10-k", "year": 2024},
            ),
            VectorRecord(
                id="chunk-10q",
                embedding=[0.51, 0.49, 0.0, 0.0],
                document="Quarterly Report 10-Q text",
                metadata={"document_type": "10-q", "year": 2024},
            ),
        ]
        store.upsert_records(col_name, records)

        # Query with filter for document_type == '10-k'
        results = store.query_similarity(
            collection_name=col_name,
            query_embedding=[0.5, 0.5, 0.0, 0.0],
            where={"document_type": "10-k"},
        )

        assert len(results) == 1
        assert results[0].id == "chunk-10k"
        assert results[0].document_type == "10-k"

    def test_empty_collection_returns_empty_results(
        self, ephemeral_chroma_client: chromadb.ClientAPI
    ) -> None:
        store = ChromaVectorStore(client=ephemeral_chroma_client)
        col_name = "finpilot_test_empty_col"
        store.get_or_create_collection(col_name)

        results = store.query_similarity(
            collection_name=col_name,
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            n_results=5,
        )
        assert results == []

    def test_search_on_uncreated_collection_raises_not_found(
        self, ephemeral_chroma_client: chromadb.ClientAPI
    ) -> None:
        store = ChromaVectorStore(client=ephemeral_chroma_client)
        with pytest.raises(CollectionNotFoundError):
            store.query_similarity(
                collection_name="finpilot_non_existent_col",
                query_embedding=[0.1, 0.2],
            )

    def test_malformed_chroma_response_raises_vector_store_error(
        self, ephemeral_chroma_client: chromadb.ClientAPI
    ) -> None:
        store = ChromaVectorStore(client=ephemeral_chroma_client)
        col_name = "finpilot_test_malformed"
        store.get_or_create_collection(col_name)
        # Put 1 record so col.count() > 0
        store.upsert_records(
            col_name,
            [
                VectorRecord(
                    id="c1",
                    embedding=[0.1, 0.2],
                    document="Text",
                    metadata={"dummy": "value"},
                )
            ],
        )

        # Mock collection.query to return mismatched arrays
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "ids": [["c1"]],
            "distances": [[]],  # mismatched: empty distances list
            "documents": [["Text"]],
            "metadatas": [[{}]],
        }
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(store._client, "get_collection", lambda name: mock_col)
            with pytest.raises(VectorStoreError) as exc_info:
                store.query_similarity(
                    collection_name=col_name, query_embedding=[0.1, 0.2]
                )
            assert "mismatched array lengths" in str(exc_info.value)


# ===========================================================================
# 5. Safe Logging Verification
# ===========================================================================


class TestSimilaritySearchSafeLogging:
    """Ensure query vectors and confidential text are not leaked in log messages."""

    def test_vectors_not_emitted_in_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = MockVectorStore()
        service = SimilaritySearchService(vector_store=store)

        with caplog.at_level(logging.INFO):
            service.search(
                collection_name=_TEST_COLLECTION,
                query=_SAMPLE_QUERY_VEC,
            )

        for record in caplog.records:
            # Vector floats should never be in the log output string
            assert str(_SAMPLE_QUERY_VEC) not in record.message
            # Dimension metadata should be present
            assert f"dims={len(_SAMPLE_QUERY_VEC)}" in record.message
