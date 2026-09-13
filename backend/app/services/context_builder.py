"""Context Construction Service for FinPilot (Phase 9.11).

Provides:
- ContextBuilder: Service for assembling Top-K retrieved chunks into structured,
  provenance-preserving context for the Research Analyst.
- build_retrieval_context: Standalone functional helper for context construction.
- Typed exceptions for context validation and budget handling.

Strictly preserves:
- Exact unedited, unparaphrased chunk text
- Phase 9.10 candidate ordering
- Complete source and citation metadata (chunk ID, document ID, ticker, pages, etc.)
"""

import math
from typing import Optional, Set

from app.core.logging import get_logger
from app.models.context import ContextChunk, ContextConfig, RetrievalContext
from app.models.retrieval import TopKRetrievalResult
from app.models.vector_store import VectorSearchResult

logger = get_logger("app.services.context_builder")

# Supported distance metrics from ChromaDB / Phase 9.9 / Phase 9.10
SUPPORTED_DISTANCE_METRICS: Set[str] = {
    "cosine_distance",
    "l2_distance",
    "inner_product",
}


# ===========================================================================
# TYPED EXCEPTIONS (Phase 9.11)
# ===========================================================================


class ContextConstructionError(Exception):
    """Base exception for all Context Construction errors."""

    def __init__(self, message: str, code: str = "CONTEXT_CONSTRUCTION_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class MalformedRetrievalResultError(ContextConstructionError):
    """Raised when input retrieval result is invalid or not TopKRetrievalResult."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="MALFORMED_RETRIEVAL_RESULT")


class InvalidChunkDataError(ContextConstructionError):
    """Raised when a candidate chunk has invalid, empty, or non-finite fields."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="INVALID_CHUNK_DATA")


class MissingProvenanceError(ContextConstructionError):
    """Raised when mandatory citation provenance (e.g. document_id) is missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="MISSING_PROVENANCE")


class InconsistentContextError(ContextConstructionError):
    """Raised when collection name or distance metric is missing or inconsistent."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="INCONSISTENT_CONTEXT")


class ContextBudgetExceededError(ContextConstructionError):
    """Raised when context exceeds max_characters and raise is enabled."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="CONTEXT_BUDGET_EXCEEDED")


# ===========================================================================
# SERVICE CLASS: ContextBuilder
# ===========================================================================


class ContextBuilder:
    """Service for constructing structured, grounded context from Top-K retrieval.

    Converts TopKRetrievalResult into a RetrievalContext without altering or
    truncating chunk text, while preserving complete provenance for citation.
    """

    def __init__(self, default_config: Optional[ContextConfig] = None) -> None:
        """Initialize ContextBuilder with an optional default ContextConfig.

        Args:
            default_config: Optional default configuration for context limits.
        """
        self._default_config = default_config or ContextConfig()

    @property
    def default_config(self) -> ContextConfig:
        """Return the default configuration."""
        return self._default_config

    def build(
        self,
        retrieval_result: TopKRetrievalResult,
        query: Optional[str] = None,
        config: Optional[ContextConfig] = None,
    ) -> RetrievalContext:
        """Construct a RetrievalContext from a TopKRetrievalResult.

        Args:
            retrieval_result: TopKRetrievalResult instance from Phase 9.10.
            query: Optional research query or reference question.
            config: Optional ContextConfig override.

        Returns:
            RetrievalContext: Structured grounded context.

        Raises:
            MalformedRetrievalResultError: If input is not a TopKRetrievalResult.
            InconsistentContextError: If collection_name or metric is invalid.
            InvalidChunkDataError: If chunk ID/text is empty or distance non-finite.
            MissingProvenanceError: If provenance is missing when required.
            ContextBudgetExceededError: If context exceeds budget and raise enabled.
        """
        # 1. Validate top-level retrieval result
        if not isinstance(retrieval_result, TopKRetrievalResult):
            raise MalformedRetrievalResultError(
                "retrieval_result must be an instance of TopKRetrievalResult, "
                f"got {type(retrieval_result).__name__}."
            )

        # 2. Validate collection name
        col_name = retrieval_result.collection_name
        if not col_name or not col_name.strip():
            raise InconsistentContextError(
                "retrieval_result.collection_name must be a non-empty string."
            )
        col_name = col_name.strip()

        # 3. Validate distance metric
        metric = (retrieval_result.distance_metric or "").strip().lower()
        if metric not in SUPPORTED_DISTANCE_METRICS:
            raise InconsistentContextError(
                f"Unsupported distance metric: '{retrieval_result.distance_metric}'. "
                f"Supported: {sorted(SUPPORTED_DISTANCE_METRICS)}."
            )

        # 4. Resolve active configuration
        active_config = config or self._default_config

        # 5. Extract and validate chunks, preserving EXACT Phase 9.10 ordering
        context_chunks = []
        for idx, item in enumerate(retrieval_result.results):
            if not isinstance(item, VectorSearchResult):
                raise MalformedRetrievalResultError(
                    f"Result item at index {idx} is not a VectorSearchResult: "
                    f"got {type(item).__name__}."
                )

            # Validate ID
            if not item.id or not str(item.id).strip():
                raise InvalidChunkDataError(
                    f"Chunk at index {idx} has an empty or invalid id."
                )

            # Validate document text (must be non-empty string)
            if (
                not item.document
                or not isinstance(item.document, str)
                or not item.document.strip()
            ):
                raise InvalidChunkDataError(
                    f"Chunk '{item.id}' at index {idx} has empty or invalid text."
                )

            # Validate distance
            if (
                not isinstance(item.distance, (int, float))
                or isinstance(item.distance, bool)
                or not math.isfinite(item.distance)
            ):
                raise InvalidChunkDataError(
                    f"Chunk '{item.id}' has non-finite distance: " f"{item.distance!r}."
                )

            # Extract provenance
            doc_id = item.document_id
            if active_config.require_provenance and (
                not doc_id or not str(doc_id).strip()
            ):
                raise MissingProvenanceError(
                    f"Chunk '{item.id}' is missing required document_id provenance."
                )

            source_val = (
                item.metadata.get("source_document")
                or item.metadata.get("source")
                or item.metadata.get("original_filename")
            )
            source_str = str(source_val) if source_val is not None else None

            chunk_idx_val = item.metadata.get("chunk_index")
            chunk_idx = (
                int(chunk_idx_val) if isinstance(chunk_idx_val, (int, float)) else None
            )

            # Construct ContextChunk — exact text preserved without modification
            chunk = ContextChunk(
                chunk_id=item.id,
                document_id=doc_id,
                source=source_str,
                ticker=item.ticker,
                document_type=item.document_type,
                pages=list(item.page_numbers),
                section=item.section_name,
                distance=item.distance,
                text=item.document,
                chunk_index=chunk_idx,
                metadata=dict(item.metadata),
            )
            context_chunks.append(chunk)

        # 6. Calculate total size metrics
        total_characters = sum(c.character_count for c in context_chunks)
        total_words = sum(c.word_count for c in context_chunks)
        has_evidence = len(context_chunks) > 0

        # 7. Check context budget
        exceeds_budget = False
        if active_config.max_characters is not None:
            if total_characters > active_config.max_characters:
                if active_config.raise_on_budget_exceeded:
                    raise ContextBudgetExceededError(
                        f"Context character count ({total_characters}) exceeds "
                        f"configured maximum budget ({active_config.max_characters})."
                    )
                exceeds_budget = True

        # 8. Operational logging (safe: summary counts and metadata only, no chunk text)
        logger.info(
            "Context constructed for '%s': chunks=%d, chars=%d, words=%d, "
            "has_evidence=%s, budget_exceeded=%s, metric=%s",
            col_name,
            len(context_chunks),
            total_characters,
            total_words,
            has_evidence,
            exceeds_budget,
            metric,
        )

        # 9. Return structured RetrievalContext
        return RetrievalContext(
            query=query,
            collection_name=col_name,
            distance_metric=metric,
            total_retrieved=len(context_chunks),
            total_characters=total_characters,
            total_words=total_words,
            chunks=context_chunks,
            has_evidence=has_evidence,
            max_distance_threshold=retrieval_result.max_distance_threshold,
            exceeds_budget=exceeds_budget,
            retrieval_metadata={
                "k": retrieval_result.k,
                "total_candidates": retrieval_result.total_candidates,
                "retrieved_at": retrieval_result.retrieved_at.isoformat(),
            },
        )


# ===========================================================================
# CONVENIENCE FUNCTION
# ===========================================================================


def build_retrieval_context(
    retrieval_result: TopKRetrievalResult,
    query: Optional[str] = None,
    config: Optional[ContextConfig] = None,
    max_characters: Optional[int] = None,
    raise_on_budget_exceeded: bool = False,
    require_provenance: bool = False,
) -> RetrievalContext:
    """Construct a RetrievalContext from a TopKRetrievalResult.

    Convenience wrapper around ContextBuilder.

    Args:
        retrieval_result: TopKRetrievalResult from Phase 9.10.
        query: Optional original query string or reference question.
        config: Optional ContextConfig instance.
        max_characters: Optional maximum character limit override.
        raise_on_budget_exceeded: Whether to raise error when limit exceeded.
        require_provenance: Whether to enforce document_id on all chunks.

    Returns:
        RetrievalContext: Structured grounded context.
    """
    if config is not None:
        effective_config = config
    elif max_characters is not None or raise_on_budget_exceeded or require_provenance:
        effective_config = ContextConfig(
            max_characters=max_characters,
            raise_on_budget_exceeded=raise_on_budget_exceeded,
            require_provenance=require_provenance,
        )
    else:
        effective_config = None

    builder = ContextBuilder(default_config=effective_config)
    return builder.build(retrieval_result=retrieval_result, query=query)
