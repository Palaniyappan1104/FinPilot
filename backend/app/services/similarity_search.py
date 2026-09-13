"""Similarity Search Service for FinPilot (Phase 9.9).

Provides:
- SimilaritySearchService: Provider-agnostic service for vector proximity queries.
- search_similarity: Standalone functional helper for collection similarity search.
- search_document_similarity: Helper scoped to a specific document collection.
- search_company_similarity: Helper scoped to a company-level collection.

Executes semantic search against stored document chunk embeddings in a
VectorStore, returning structured VectorSearchResult matches with explicit
distance semantics (lower distance = closer match) and preserved citation
provenance.
"""

import math
from typing import Dict, List, Optional, Union

from app.core.logging import get_logger
from app.models.embeddings import QueryEmbeddingResult
from app.models.vector_store import SimilaritySearchResults, VectorSearchResult
from app.storage.chroma_vector_store import get_vector_store
from app.storage.vector_base import (
    VectorStore,
    build_company_collection_name,
    build_document_collection_name,
    validate_collection_name,
)
from app.storage.vector_exceptions import (
    CollectionNotFoundError,
    VectorDimensionMismatchError,
    VectorValidationError,
)

logger = get_logger("app.services.similarity_search")


class SimilaritySearchService:
    """Provider-agnostic similarity search service for FinPilot.

    Executes vector proximity search against collections in a VectorStore,
    returning structured VectorSearchResult matches with explicit distance
    semantics, full chunk text, and preserved document provenance.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        expected_dimensions: Optional[int] = None,
    ) -> None:
        """Initialize with a concrete VectorStore and optional dimension constraint.

        Args:
            vector_store: VectorStore implementation (defaults to configured store).
            expected_dimensions: Optional expected dimensionality of query vectors.
        """
        self._vector_store = vector_store or get_vector_store()
        self._expected_dimensions = expected_dimensions
        logger.debug(
            "SimilaritySearchService initialized with store=%s, expected_dims=%s",
            self._vector_store.store_name,
            expected_dimensions,
        )

    @property
    def vector_store(self) -> VectorStore:
        """Return the backing VectorStore instance."""
        return self._vector_store

    def search(
        self,
        collection_name: str,
        query: Union[List[float], QueryEmbeddingResult],
        n_results: int = 10,
        where: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        expected_dimensions: Optional[int] = None,
    ) -> SimilaritySearchResults:
        """Execute vector similarity search against a named collection.

        Args:
            collection_name: Target collection name.
            query: Query vector as a list of floats or a QueryEmbeddingResult.
            n_results: Maximum number of nearest matches to return (must be > 0).
            where: Optional metadata filter dictionary.
            expected_dimensions: Optional override for expected vector dimension.

        Returns:
            SimilaritySearchResults: Container of ranked matches and search metadata.

        Raises:
            InvalidCollectionNameError: If collection name violates format rules.
            CollectionNotFoundError: If collection does not exist.
            VectorValidationError: If query vector or n_results is invalid.
            VectorDimensionMismatchError: If query vector length mismatches expected.
            VectorStoreError: On database or query failure.
        """
        # 1. Validate collection name
        validated_name = validate_collection_name(collection_name)

        # 2. Check collection existence
        if not self._vector_store.has_collection(validated_name):
            raise CollectionNotFoundError(validated_name)

        # 3. Validate n_results
        if not isinstance(n_results, int) or n_results <= 0:
            raise VectorValidationError(
                f"n_results must be a positive integer, got {n_results}.",
                collection=validated_name,
            )

        # 4. Extract and validate vector
        vec = self._extract_and_validate_vector(
            query=query,
            collection_name=validated_name,
            expected_dimensions=expected_dimensions or self._expected_dimensions,
        )

        # 5. Safe operational logging (log dimensions & count, never raw vector)
        logger.info(
            "Executing similarity search in '%s' (dims=%d, n_results=%d, filter=%s)",
            validated_name,
            len(vec),
            n_results,
            bool(where),
        )

        # 6. Execute search via VectorStore abstraction
        results: List[VectorSearchResult] = self._vector_store.query_similarity(
            collection_name=validated_name,
            query_embedding=vec,
            n_results=n_results,
            where=where,
        )

        # 7. Determine reported distance metric
        metric_name = results[0].distance_metric if results else "cosine_distance"

        return SimilaritySearchResults(
            collection_name=validated_name,
            results=results,
            total_results=len(results),
            distance_metric=metric_name,
        )

    def search_document(
        self,
        ticker: str,
        document_id: str,
        query: Union[List[float], QueryEmbeddingResult],
        n_results: int = 10,
        where: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        expected_dimensions: Optional[int] = None,
    ) -> SimilaritySearchResults:
        """Search a deterministic document-level collection (finpilot_{ticker}_{doc}).

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL').
            document_id: Source document identifier (e.g. 'doc-annual-2024').
            query: Query vector or QueryEmbeddingResult.
            n_results: Maximum number of nearest matches to return.
            where: Optional metadata filter dictionary.
            expected_dimensions: Optional expected vector dimension.

        Returns:
            SimilaritySearchResults: Matches from the document's collection.
        """
        collection_name = build_document_collection_name(
            ticker=ticker, document_id=document_id
        )
        return self.search(
            collection_name=collection_name,
            query=query,
            n_results=n_results,
            where=where,
            expected_dimensions=expected_dimensions,
        )

    def search_company(
        self,
        ticker: str,
        query: Union[List[float], QueryEmbeddingResult],
        n_results: int = 10,
        where: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        expected_dimensions: Optional[int] = None,
    ) -> SimilaritySearchResults:
        """Search a deterministic company-level collection (finpilot_company_{ticker}).

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL').
            query: Query vector or QueryEmbeddingResult.
            n_results: Maximum number of nearest matches to return.
            where: Optional metadata filter dictionary.
            expected_dimensions: Optional expected vector dimension.

        Returns:
            SimilaritySearchResults: Matches from the company's collection.
        """
        collection_name = build_company_collection_name(ticker=ticker)
        return self.search(
            collection_name=collection_name,
            query=query,
            n_results=n_results,
            where=where,
            expected_dimensions=expected_dimensions,
        )

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    def _extract_and_validate_vector(
        self,
        query: Union[List[float], QueryEmbeddingResult],
        collection_name: str,
        expected_dimensions: Optional[int] = None,
    ) -> List[float]:
        """Extract float list from input and validate integrity & dimensions."""
        if isinstance(query, QueryEmbeddingResult):
            raw_vec = query.embedding
        elif isinstance(query, (list, tuple)):
            raw_vec = list(query)
        else:
            raise VectorValidationError(
                "Query must be a list of floats or a QueryEmbeddingResult, "
                f"got {type(query).__name__}.",
                collection=collection_name,
            )

        if not raw_vec:
            raise VectorValidationError(
                "Query vector cannot be empty.",
                collection=collection_name,
            )

        try:
            vec = [float(x) for x in raw_vec]
        except (TypeError, ValueError) as exc:
            raise VectorValidationError(
                f"Query vector contains non-numeric values: {exc}",
                collection=collection_name,
            ) from exc

        non_finite = [x for x in vec if not math.isfinite(x)]
        if non_finite:
            raise VectorValidationError(
                f"Query vector contains {len(non_finite)} non-finite value(s).",
                collection=collection_name,
            )

        if expected_dimensions is not None and len(vec) != expected_dimensions:
            raise VectorDimensionMismatchError(
                f"Query vector dimension mismatch: got {len(vec)}, "
                f"expected {expected_dimensions}.",
                collection=collection_name,
                expected_dim=expected_dimensions,
                actual_dim=len(vec),
            )

        return vec


# ===========================================================================
# CONVENIENCE FUNCTIONS
# ===========================================================================


def search_similarity(
    query: Union[List[float], QueryEmbeddingResult],
    collection_name: str,
    vector_store: Optional[VectorStore] = None,
    n_results: int = 10,
    where: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    expected_dimensions: Optional[int] = None,
) -> SimilaritySearchResults:
    """Execute vector similarity search against a collection (Phase 9.9).

    Convenience wrapper around SimilaritySearchService.

    Args:
        query: Query vector or QueryEmbeddingResult.
        collection_name: Target collection name.
        vector_store: Optional VectorStore instance.
        n_results: Maximum number of matches to return.
        where: Optional metadata filter dictionary.
        expected_dimensions: Optional expected vector dimension.

    Returns:
        SimilaritySearchResults: Container of ranked matches and search metadata.
    """
    service = SimilaritySearchService(
        vector_store=vector_store,
        expected_dimensions=expected_dimensions,
    )
    return service.search(
        collection_name=collection_name,
        query=query,
        n_results=n_results,
        where=where,
    )


def search_document_similarity(
    ticker: str,
    document_id: str,
    query: Union[List[float], QueryEmbeddingResult],
    vector_store: Optional[VectorStore] = None,
    n_results: int = 10,
    where: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    expected_dimensions: Optional[int] = None,
) -> SimilaritySearchResults:
    """Search a document-level collection (finpilot_{ticker}_{doc}) by similarity.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        document_id: Unique document identifier (e.g. 'doc-annual-2024').
        query: Query vector or QueryEmbeddingResult.
        vector_store: Optional VectorStore instance.
        n_results: Maximum number of matches to return.
        where: Optional metadata filter dictionary.
        expected_dimensions: Optional expected vector dimension.

    Returns:
        SimilaritySearchResults: Ranked matches from the document's chunks.
    """
    service = SimilaritySearchService(
        vector_store=vector_store,
        expected_dimensions=expected_dimensions,
    )
    return service.search_document(
        ticker=ticker,
        document_id=document_id,
        query=query,
        n_results=n_results,
        where=where,
    )


def search_company_similarity(
    ticker: str,
    query: Union[List[float], QueryEmbeddingResult],
    vector_store: Optional[VectorStore] = None,
    n_results: int = 10,
    where: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    expected_dimensions: Optional[int] = None,
) -> SimilaritySearchResults:
    """Search a company-level collection (finpilot_company_{ticker}) by similarity.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        query: Query vector or QueryEmbeddingResult.
        vector_store: Optional VectorStore instance.
        n_results: Maximum number of matches to return.
        where: Optional metadata filter dictionary.
        expected_dimensions: Optional expected vector dimension.

    Returns:
        SimilaritySearchResults: Ranked matches across the company's collections.
    """
    service = SimilaritySearchService(
        vector_store=vector_store,
        expected_dimensions=expected_dimensions,
    )
    return service.search_company(
        ticker=ticker,
        query=query,
        n_results=n_results,
        where=where,
    )
