"""Query Embedding Service for FinPilot (Phase 9.8).

Provides:
- QueryEmbedder: Service class wrapping an EmbeddingProvider for user queries.
- embed_query: Functional helper to embed a search/research query.

Converts incoming user research questions into the same dense vector space
used by document chunks (Phase 9.6/9.7), enabling downstream similarity
search (Phase 9.9) while keeping vector store and retrieval logic strictly
decoupled.
"""

import math
from typing import Optional, Union

from app.core.logging import get_logger
from app.models.embeddings import (
    EmbeddingModelInfo,
    QueryEmbeddingRequest,
    QueryEmbeddingResult,
)
from app.providers.embedding import (
    EmbeddingDimensionMismatchError,
    EmbeddingEmptyInputError,
    EmbeddingInvalidResponseError,
    EmbeddingProvider,
    get_embedding_provider,
)

logger = get_logger("app.services.query_embedder")


# ===========================================================================
# SERVICE CLASS: QueryEmbedder
# ===========================================================================


class QueryEmbedder:
    """Service class for generating dense embeddings from user research queries.

    Keeps the embedding provider isolated, ensures dimensionality matches the
    configured document embedding space, and enforces safe logging without
    exposing sensitive user query text.
    """

    def __init__(self, provider: Optional[EmbeddingProvider] = None) -> None:
        """Initialize QueryEmbedder with a concrete EmbeddingProvider.

        Args:
            provider: Optional concrete EmbeddingProvider (Gemini, mock, etc.).
                     If omitted, resolves the application-configured provider.
        """
        self._provider = provider or get_embedding_provider()
        logger.debug(
            "QueryEmbedder initialized with provider=%s model=%s dims=%d",
            self._provider.provider_name,
            self._provider.model_name,
            self._provider.dimensions,
        )

    @property
    def provider(self) -> EmbeddingProvider:
        """Return the underlying EmbeddingProvider instance."""
        return self._provider

    @property
    def model_info(self) -> EmbeddingModelInfo:
        """Return metadata about the backing provider and model configuration."""
        return self._provider.get_model_info()

    def embed_query(
        self,
        query: Union[str, QueryEmbeddingRequest],
        query_id: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> QueryEmbeddingResult:
        """Generate a dense embedding vector for a research question.

        Args:
            query: Non-empty query string or QueryEmbeddingRequest.
            query_id: Optional tracking identifier for the query.
            ticker: Optional ticker symbol associated with the query context.

        Returns:
            QueryEmbeddingResult: Validated embedding vector with query metadata.

        Raises:
            EmbeddingEmptyInputError: If query is blank or invalid.
            EmbeddingDimensionMismatchError: If dimension does not match provider.
            EmbeddingInvalidResponseError: If vector is empty or non-finite.
            EmbeddingError: On provider-level or API failure.
        """
        # 1. Normalize and validate input
        if isinstance(query, QueryEmbeddingRequest):
            request = query
        elif isinstance(query, str):
            clean_query = query.strip()
            if not clean_query:
                raise EmbeddingEmptyInputError(
                    "Query text cannot be empty or whitespace-only.",
                    provider=self._provider.provider_name,
                )
            request = QueryEmbeddingRequest(
                query=clean_query,
                query_id=query_id,
                ticker=ticker,
            )
        else:
            raise EmbeddingEmptyInputError(
                "Query must be a string or QueryEmbeddingRequest, "
                f"got {type(query).__name__}.",
                provider=self._provider.provider_name,
            )

        # 2. Safe operational logging (log length/provenance, NEVER query text)
        logger.info(
            "Embedding query (chars=%d, query_id=%s, ticker=%s) via %s/%s",
            len(request.query),
            request.query_id or "none",
            request.ticker or "none",
            self._provider.provider_name,
            self._provider.model_name,
        )

        # 3. Invoke provider
        vector = self._provider.embed_query(request.query)

        # 4. Service-layer defensive validation
        if not vector:
            raise EmbeddingInvalidResponseError(
                "Embedding provider returned an empty vector for query.",
                provider=self._provider.provider_name,
            )

        non_finite = [x for x in vector if not math.isfinite(x)]
        if non_finite:
            raise EmbeddingInvalidResponseError(
                f"Embedding vector contains {len(non_finite)} non-finite value(s).",
                provider=self._provider.provider_name,
            )

        expected_dims = self._provider.dimensions
        if expected_dims and len(vector) != expected_dims:
            raise EmbeddingDimensionMismatchError(
                f"Query embedding dimension mismatch: got {len(vector)}, "
                f"expected {expected_dims}.",
                provider=self._provider.provider_name,
                expected_dimensions=expected_dims,
                actual_dimensions=len(vector),
            )

        # 5. Assemble structured result
        return QueryEmbeddingResult(
            query=request.query,
            query_id=request.query_id,
            ticker=request.ticker,
            embedding=vector,
            embedding_model=self._provider.model_name,
            embedding_dimensions=len(vector),
        )


# ===========================================================================
# CONVENIENCE FUNCTION: embed_query
# ===========================================================================


def embed_query(
    query: Union[str, QueryEmbeddingRequest],
    provider: Optional[EmbeddingProvider] = None,
    query_id: Optional[str] = None,
    ticker: Optional[str] = None,
) -> QueryEmbeddingResult:
    """Generate a dense embedding vector for a search/research query.

    Convenience wrapper around QueryEmbedder.

    Args:
        query: Non-empty query string or QueryEmbeddingRequest.
        provider: Optional concrete EmbeddingProvider. Defaults to configured provider.
        query_id: Optional tracking identifier for the query.
        ticker: Optional ticker symbol associated with the query context.

    Returns:
        QueryEmbeddingResult: Validated embedding vector coupled with query metadata.
    """
    embedder = QueryEmbedder(provider=provider)
    return embedder.embed_query(query, query_id=query_id, ticker=ticker)
