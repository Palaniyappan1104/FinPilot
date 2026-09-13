"""Top-K Retrieval Service for FinPilot (Phase 9.10).

Provides:
- TopKRetriever: Service for selecting and ranking the Top-K most relevant chunks.
- retrieve_top_k: Standalone functional helper for Top-K retrieval.
- Typed exceptions for retrieval validation and error handling.

Takes similarity-search results from Phase 9.9 and deterministically selects the
appropriate number of highest-quality / most relevant results based on
metric-aware distance ordering and optional relevance thresholding.
"""

import math
from typing import List, Optional, Set, Union

from app.core.logging import get_logger
from app.models.retrieval import TopKConfig, TopKRetrievalResult
from app.models.vector_store import SimilaritySearchResults, VectorSearchResult

logger = get_logger("app.services.top_k_retriever")

# Supported distance metrics from ChromaDB / Phase 9.9
# In all three metrics, lower distance values indicate closer semantic proximity.
SUPPORTED_DISTANCE_METRICS: Set[str] = {
    "cosine_distance",
    "l2_distance",
    "inner_product",
}


# ===========================================================================
# TYPED EXCEPTIONS (Phase 9.10)
# ===========================================================================


class TopKRetrievalError(Exception):
    """Base exception for all Top-K retrieval errors."""

    def __init__(self, message: str, code: str = "TOP_K_RETRIEVAL_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class InvalidKError(TopKRetrievalError):
    """Raised when k is non-positive, non-integer, or invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="INVALID_K")


class InvalidRelevanceThresholdError(TopKRetrievalError):
    """Raised when relevance/distance threshold is invalid (negative or non-finite)."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="INVALID_RELEVANCE_THRESHOLD")


class UnsupportedDistanceMetricError(TopKRetrievalError):
    """Raised when an unknown distance metric cannot be safely ordered."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="UNSUPPORTED_DISTANCE_METRIC")


class MalformedSearchResultError(TopKRetrievalError):
    """Raised when input search result objects are malformed or non-finite."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="MALFORMED_SEARCH_RESULT")


# ===========================================================================
# SERVICE CLASS: TopKRetriever
# ===========================================================================


class TopKRetriever:
    """Service class for deterministic Top-K candidate selection (Phase 9.10).

    Takes candidate chunks from similarity search, orders them by relevance
    (ascending distance for distance-based metrics), applies optional relevance
    thresholds, and selects the top K matches while strictly preserving provenance.
    """

    def __init__(self, default_config: Optional[TopKConfig] = None) -> None:
        """Initialize TopKRetriever with an optional default TopKConfig.

        Args:
            default_config: Optional default configuration for retrieval.
        """
        self._default_config = default_config or TopKConfig()
        logger.debug(
            "TopKRetriever initialized with default k=%d, max_distance=%s",
            self._default_config.k,
            self._default_config.max_distance,
        )

    @property
    def default_config(self) -> TopKConfig:
        """Return the default retrieval configuration."""
        return self._default_config

    def retrieve(
        self,
        search_results: Union[SimilaritySearchResults, List[VectorSearchResult]],
        config: Optional[TopKConfig] = None,
        k: Optional[int] = None,
        max_distance: Optional[float] = None,
        collection_name: Optional[str] = None,
        distance_metric: Optional[str] = None,
    ) -> TopKRetrievalResult:
        """Select the Top-K most relevant records from similarity-search candidates.

        Args:
            search_results: SimilaritySearchResults or a list of VectorSearchResult.
            config: Optional TopKConfig instance.
            k: Optional override for k (requested result count, must be >= 1).
            max_distance: Optional override for distance threshold.
            collection_name: Optional collection name override if passing a raw list.
            distance_metric: Optional distance metric override if passing a raw list.

        Returns:
            TopKRetrievalResult: Ranked and filtered Top-K matches.

        Raises:
            InvalidKError: If k <= 0 or not an integer.
            InvalidRelevanceThresholdError: If max_distance < 0 or non-finite.
            UnsupportedDistanceMetricError: If distance metric is unrecognized.
            MalformedSearchResultError: If candidate records are invalid.
        """
        # 1. Resolve configuration parameters
        active_config = config or self._default_config
        eff_k = k if k is not None else active_config.k
        eff_max_distance = (
            max_distance if max_distance is not None else active_config.max_distance
        )

        # 2. Validate K
        if isinstance(eff_k, bool) or not isinstance(eff_k, int) or eff_k <= 0:
            raise InvalidKError(f"k must be an integer greater than 0, got {eff_k!r}.")

        # 3. Validate threshold if provided
        if eff_max_distance is not None:
            if (
                isinstance(eff_max_distance, bool)
                or not isinstance(eff_max_distance, (int, float))
                or not math.isfinite(eff_max_distance)
                or eff_max_distance < 0.0
            ):
                raise InvalidRelevanceThresholdError(
                    "max_distance must be a non-negative finite float, "
                    f"got {eff_max_distance!r}."
                )

        # 4. Extract candidates, collection name, and metric
        if isinstance(search_results, SimilaritySearchResults):
            candidates = search_results.results
            resolved_col = collection_name or search_results.collection_name
            resolved_metric = distance_metric or search_results.distance_metric
        elif isinstance(search_results, list):
            candidates = search_results
            resolved_col = collection_name or "unknown_collection"
            resolved_metric = distance_metric or "cosine_distance"
        else:
            raise MalformedSearchResultError(
                "search_results must be a SimilaritySearchResults or list of "
                f"VectorSearchResult, got {type(search_results).__name__}."
            )

        # 5. Validate distance metric
        clean_metric = (resolved_metric or "").strip().lower()
        if clean_metric not in SUPPORTED_DISTANCE_METRICS:
            raise UnsupportedDistanceMetricError(
                f"Unsupported distance metric: '{resolved_metric}'. "
                f"Supported metrics: {sorted(SUPPORTED_DISTANCE_METRICS)}."
            )

        # 6. Validate candidates
        for idx, item in enumerate(candidates):
            if not isinstance(item, VectorSearchResult):
                raise MalformedSearchResultError(
                    f"Candidate match at index {idx} is not a VectorSearchResult: "
                    f"got {type(item).__name__}."
                )
            if not math.isfinite(item.distance):
                raise MalformedSearchResultError(
                    f"Candidate match '{item.id}' has non-finite distance: "
                    f"{item.distance}."
                )

        total_candidates = len(candidates)

        # 7. Apply relevance thresholding if configured (Phase 9.10.2)
        if eff_max_distance is not None:
            eligible = [
                item for item in candidates if item.distance <= eff_max_distance
            ]
        else:
            eligible = list(candidates)

        # 8. Deterministic relevance ranking
        # In distance metrics (cosine_distance, l2_distance, inner_product),
        # lower distance indicates higher relevance. Ties are broken deterministically
        # by the original candidate sequence order without fabricating relevance.
        indexed_eligible = list(enumerate(eligible))
        indexed_eligible.sort(key=lambda pair: (pair[1].distance, pair[0]))
        sorted_candidates = [pair[1] for pair in indexed_eligible]

        # 9. Select Top-K
        top_k_results = sorted_candidates[:eff_k]

        # 10. Safe operational logging (log counts and parameters, never document text)
        logger.info(
            "Top-K retrieval completed for '%s': candidates=%d, eligible=%d, "
            "retrieved=%d, k=%d, threshold=%s, metric=%s",
            resolved_col,
            total_candidates,
            len(eligible),
            len(top_k_results),
            eff_k,
            eff_max_distance,
            clean_metric,
        )

        # 11. Assemble structured output
        return TopKRetrievalResult(
            collection_name=resolved_col,
            k=eff_k,
            results=top_k_results,
            total_candidates=total_candidates,
            total_retrieved=len(top_k_results),
            distance_metric=clean_metric,
            max_distance_threshold=eff_max_distance,
        )


# ===========================================================================
# CONVENIENCE FUNCTION
# ===========================================================================


def retrieve_top_k(
    search_results: Union[SimilaritySearchResults, List[VectorSearchResult]],
    k: int = 5,
    max_distance: Optional[float] = None,
    collection_name: Optional[str] = None,
    distance_metric: Optional[str] = None,
) -> TopKRetrievalResult:
    """Select the Top-K most relevant records from similarity-search candidates.

    Convenience wrapper around TopKRetriever.

    Args:
        search_results: SimilaritySearchResults or a list of VectorSearchResult items.
        k: Number of top results to retrieve (must be >= 1).
        max_distance: Optional maximum distance threshold (weak matches filtered out).
        collection_name: Optional collection name override if passing a raw list.
        distance_metric: Optional distance metric override if passing a raw list.

    Returns:
        TopKRetrievalResult: Ranked and filtered Top-K matches.
    """
    retriever = TopKRetriever()
    return retriever.retrieve(
        search_results=search_results,
        k=k,
        max_distance=max_distance,
        collection_name=collection_name,
        distance_metric=distance_metric,
    )
