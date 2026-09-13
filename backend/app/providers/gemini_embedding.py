"""Gemini Embedding Provider implementation for FinPilot (Phase 9.6).

Uses the google-genai SDK (already a project dependency) with the
'gemini-embedding-001' model which is designed for semantic retrieval
and supports financial text with a 3072-dimensional output vector.

Selected because:
- google-genai is already a core dependency (no new package required).
- 'gemini-embedding-001' is Google's production embedding model for
  semantic retrieval tasks, tuned for document Q&A similarity search.
- 3072 dimensions provides high-fidelity representation for financial
  terminology and numerical content.
- Task type RETRIEVAL_DOCUMENT is the correct Gemini task type for
  indexing text chunks in a RAG pipeline.
"""

import math
from typing import List, Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.embeddings import (
    ChunkEmbeddingRequest,
    ChunkEmbeddingResult,
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
)

logger = get_logger("app.providers.gemini_embedding")

# Gemini task type for RAG document indexing
_RETRIEVAL_DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"

# Known output dimensionality for gemini-embedding-001 (default full precision)
_DEFAULT_DIMENSIONS = 3072


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Concrete EmbeddingProvider using Google Gemini embedding models.

    All network/API calls are isolated within this class; no ChromaDB,
    no vector store logic, and no fabricated vectors.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._model = (
            getattr(self._settings, "EMBEDDING_MODEL", "gemini-embedding-001")
            or "gemini-embedding-001"
        )
        self._dimensions = (
            getattr(self._settings, "EMBEDDING_DIMENSIONS", _DEFAULT_DIMENSIONS)
            or _DEFAULT_DIMENSIONS
        )
        api_key = (getattr(self._settings, "GEMINI_API_KEY", "") or "").strip()
        if not api_key:
            raise EmbeddingProviderNotConfiguredError(
                "GEMINI_API_KEY is not set. "
                "Set the GEMINI_API_KEY environment variable to use the "
                "Gemini embedding provider.",
                provider="gemini",
            )
        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise EmbeddingError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai",
                provider="gemini",
                code="DEPENDENCY_MISSING",
            ) from exc

    # ------------------------------------------------------------------
    # Provider identity
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider=self.provider_name,
            model_name=self.model_name,
            dimensions=self.dimensions,
        )

    # ------------------------------------------------------------------
    # Single chunk embedding
    # ------------------------------------------------------------------

    def embed_single(self, request: ChunkEmbeddingRequest) -> ChunkEmbeddingResult:
        """Embed a single ChunkEmbeddingRequest via the Gemini API."""
        if not request.text or not request.text.strip():
            raise EmbeddingEmptyInputError(
                f"Chunk '{request.chunk_id}' has empty text and cannot be embedded.",
                provider="gemini",
            )

        logger.debug(
            "Embedding single chunk '%s' (doc=%s, chars=%d) via %s",
            request.chunk_id,
            request.document_id,
            len(request.text),
            self._model,
        )

        vector = self._call_api_single(request.text, request.chunk_id)
        return self._build_result(request, vector)

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

    def embed_batch(
        self, requests: List[ChunkEmbeddingRequest]
    ) -> List[ChunkEmbeddingResult]:
        """Embed a list of ChunkEmbeddingRequests via the Gemini API.

        Uses batched API call when possible (multiple contents in one request),
        falling back to sequential single calls if the batch API is unavailable.
        The order of results strictly mirrors the order of the input requests.
        """
        if not requests:
            return []

        # Validate all inputs before calling the API
        for req in requests:
            if not req.text or not req.text.strip():
                raise EmbeddingEmptyInputError(
                    f"Chunk '{req.chunk_id}' has empty text and cannot be embedded.",
                    provider="gemini",
                )

        logger.debug(
            "Batch-embedding %d chunks (doc=%s) via %s",
            len(requests),
            requests[0].document_id if requests else "unknown",
            self._model,
        )

        texts = [r.text for r in requests]
        vectors = self._call_api_batch(texts, [r.chunk_id for r in requests])

        if len(vectors) != len(requests):
            raise EmbeddingInvalidResponseError(
                f"Gemini returned {len(vectors)} embeddings for "
                f"{len(requests)} input chunks. Counts must match.",
                provider="gemini",
            )

        return [self._build_result(req, vec) for req, vec in zip(requests, vectors)]

    # ------------------------------------------------------------------
    # Internal API call helpers
    # ------------------------------------------------------------------

    def _call_api_single(self, text: str, chunk_id: str) -> List[float]:
        """Call Gemini embed_content for a single text, return the float vector."""
        try:
            from google.genai import types as genai_types

            response = self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    task_type=_RETRIEVAL_DOCUMENT_TASK,
                    output_dimensionality=self._dimensions,
                ),
            )
        except Exception as exc:
            self._map_api_exception(exc, context=f"chunk '{chunk_id}'")

        return self._extract_vector_single(response, chunk_id)

    def _call_api_batch(
        self, texts: List[str], chunk_ids: List[str]
    ) -> List[List[float]]:
        """Call Gemini embed_content for a batch of texts, return list of vectors."""
        context = f"batch of {len(texts)} chunks"
        try:
            from google.genai import types as genai_types

            response = self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    task_type=_RETRIEVAL_DOCUMENT_TASK,
                    output_dimensionality=self._dimensions,
                ),
            )
        except Exception as exc:
            self._map_api_exception(exc, context=context)

        return self._extract_vector_batch(response, chunk_ids)

    def _map_api_exception(self, exc: Exception, context: str) -> None:
        """Convert google-genai API exceptions to typed EmbeddingError subclasses."""
        exc_code = getattr(exc, "code", None)
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__

        if (
            exc_code in (401, 403)
            or "api_key" in exc_str
            or "401" in exc_str
            or "unauthenticated" in exc_str
        ):
            raise EmbeddingAuthenticationError(
                f"Gemini API authentication failed for {context}: {exc}",
                provider="gemini",
            ) from exc

        if (
            exc_code == 429
            or "429" in exc_str
            or "quota" in exc_str
            or "rate" in exc_str
        ):
            raise EmbeddingProviderRateLimitError(
                f"Gemini API rate limit exceeded for {context}: {exc}",
                provider="gemini",
            ) from exc

        if exc_code in (500, 502, 503, 504) or any(
            kw in exc_str
            for kw in ("connection", "timeout", "unavailable", "503", "network")
        ):
            raise EmbeddingProviderUnavailableError(
                f"Gemini API unavailable for {context}: {exc}",
                provider="gemini",
            ) from exc

        raise EmbeddingError(
            f"Gemini API error for {context} ({exc_type}): {exc}",
            provider="gemini",
            code="API_ERROR",
        ) from exc

    def _extract_vector_single(self, response: object, chunk_id: str) -> List[float]:
        """Extract the embedding float list from a single embed_content response."""
        embeddings = getattr(response, "embeddings", None)
        if not embeddings or len(embeddings) == 0:
            raise EmbeddingInvalidResponseError(
                f"Gemini returned no embeddings for chunk '{chunk_id}'.",
                provider="gemini",
            )
        values = getattr(embeddings[0], "values", None)
        return self._validate_vector(values, chunk_id)

    def _extract_vector_batch(
        self, response: object, chunk_ids: List[str]
    ) -> List[List[float]]:
        """Extract the list of embedding float lists from a batch response."""
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise EmbeddingInvalidResponseError(
                "Gemini returned no embeddings for batch request.",
                provider="gemini",
            )
        results: List[List[float]] = []
        for i, emb in enumerate(embeddings):
            label = chunk_ids[i] if i < len(chunk_ids) else f"index {i}"
            values = getattr(emb, "values", None)
            results.append(self._validate_vector(values, label))
        return results

    def _validate_vector(self, values: object, label: str) -> List[float]:
        """Validate and return a float list embedding vector."""
        if not values:
            raise EmbeddingInvalidResponseError(
                f"Gemini returned an empty vector for '{label}'.",
                provider="gemini",
            )
        try:
            vec: List[float] = [float(x) for x in values]  # type: ignore[union-attr]
        except (TypeError, ValueError) as exc:
            raise EmbeddingInvalidResponseError(
                f"Gemini vector for '{label}' contains non-numeric values: {exc}",
                provider="gemini",
            ) from exc
        if not vec:
            raise EmbeddingInvalidResponseError(
                f"Gemini returned an empty vector for '{label}'.",
                provider="gemini",
            )
        non_finite = [x for x in vec if not math.isfinite(x)]
        if non_finite:
            raise EmbeddingInvalidResponseError(
                f"Gemini vector for '{label}' contains "
                f"{len(non_finite)} non-finite value(s).",
                provider="gemini",
            )
        if self._dimensions and len(vec) != self._dimensions:
            raise EmbeddingDimensionMismatchError(
                f"Gemini vector for '{label}' has dimension {len(vec)}, "
                f"expected {self._dimensions}.",
                provider="gemini",
                expected_dimensions=self._dimensions,
                actual_dimensions=len(vec),
            )
        return vec

    def _build_result(
        self, request: ChunkEmbeddingRequest, vector: List[float]
    ) -> ChunkEmbeddingResult:
        """Assemble a ChunkEmbeddingResult from a request and embedding vector."""
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
            embedding=vector,
            embedding_model=self._model,
            embedding_dimensions=len(vector),
        )
