"""Unit tests for Phase 9.10: Top-K Retrieval.

Covers:
- TopKConfig schema validation and edge cases
- TopKRetrievalResult schema, properties, and immutability
- TopKRetriever deterministic ordering (ascending distance)
- Metric-aware distance handling (cosine_distance, l2_distance, inner_product)
- Deterministic tie-breaking on identical distances
- Relevance thresholding (max_distance filter)
- Candidate counts and K bounds (k < total, k == total, k > total, empty candidates)
- Complete provenance preservation
- Error conditions and typed exceptions
- Functional helper retrieve_top_k
- Safe operational logging
"""

import logging

import pytest
from pydantic import ValidationError

from app.models.retrieval import TopKConfig, TopKRetrievalResult
from app.models.vector_store import SimilaritySearchResults, VectorSearchResult
from app.services.top_k_retriever import (
    InvalidKError,
    InvalidRelevanceThresholdError,
    MalformedSearchResultError,
    TopKRetrievalError,
    TopKRetriever,
    UnsupportedDistanceMetricError,
    retrieve_top_k,
)

# ===========================================================================
# FIXTURES & HELPERS
# ===========================================================================


def make_vector_result(
    chunk_id: str,
    distance: float,
    doc_id: str = "doc_10k_2024",
    ticker: str = "AAPL",
    chunk_index: int = 0,
    text: str = "Sample financial text chunk.",
) -> VectorSearchResult:
    """Helper to construct a valid VectorSearchResult for testing."""
    return VectorSearchResult(
        id=chunk_id,
        distance=distance,
        document=text,
        metadata={
            "document_id": doc_id,
            "ticker": ticker,
            "chunk_index": chunk_index,
            "source": "10-K",
            "year": 2024,
        },
    )


def make_similarity_results(
    results: list,
    collection_name: str = "company_AAPL",
    metric: str = "cosine_distance",
) -> SimilaritySearchResults:
    """Helper to construct a valid SimilaritySearchResults container."""
    return SimilaritySearchResults(
        collection_name=collection_name,
        results=results,
        total_results=len(results),
        distance_metric=metric,
    )


# ===========================================================================
# 1. MODEL TESTS: TopKConfig & TopKRetrievalResult
# ===========================================================================


def test_top_k_config_defaults():
    """Verify default TopKConfig values."""
    cfg = TopKConfig()
    assert cfg.k == 5
    assert cfg.max_distance is None


def test_top_k_config_custom_valid():
    """Verify custom TopKConfig creation with valid values."""
    cfg = TopKConfig(k=10, max_distance=0.45)
    assert cfg.k == 10
    assert cfg.max_distance == 0.45


def test_top_k_config_immutability():
    """Verify TopKConfig is frozen and immutable."""
    cfg = TopKConfig(k=5)
    with pytest.raises(ValidationError):
        cfg.k = 10  # type: ignore


@pytest.mark.parametrize(
    "invalid_k",
    [0, -1, -100, 2.5, "5", True, False, None],
)
def test_top_k_config_invalid_k(invalid_k):
    """Verify validation errors for non-positive or non-integer k."""
    with pytest.raises(ValidationError):
        TopKConfig(k=invalid_k)


@pytest.mark.parametrize(
    "invalid_dist",
    [-0.1, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "0.5"],
)
def test_top_k_config_invalid_max_distance(invalid_dist):
    """Verify validation errors for negative or non-finite max_distance."""
    with pytest.raises(ValidationError):
        TopKConfig(max_distance=invalid_dist)


def test_top_k_retrieval_result_properties():
    """Verify helper properties of TopKRetrievalResult."""
    res1 = make_vector_result("c1", 0.15, doc_id="doc_A", ticker="AAPL")
    res2 = make_vector_result("c2", 0.25, doc_id="doc_B", ticker="MSFT")
    res3 = make_vector_result("c3", 0.35, doc_id="doc_A", ticker="AAPL")

    result = TopKRetrievalResult(
        collection_name="company_AAPL",
        k=5,
        results=[res1, res2, res3],
        total_candidates=10,
        total_retrieved=3,
        distance_metric="cosine_distance",
        max_distance_threshold=0.5,
    )

    assert not result.is_empty
    assert result.document_ids == ["doc_A", "doc_B"]
    assert result.tickers == ["AAPL", "MSFT"]
    assert result.total_candidates == 10
    assert result.total_retrieved == 3
    assert result.distance_metric == "cosine_distance"
    assert result.max_distance_threshold == 0.5


def test_top_k_retrieval_result_empty_properties():
    """Verify properties when TopKRetrievalResult contains no matches."""
    empty_result = TopKRetrievalResult(
        collection_name="company_AAPL",
        k=5,
        results=[],
        total_candidates=0,
        total_retrieved=0,
    )
    assert empty_result.is_empty
    assert empty_result.document_ids == []
    assert empty_result.tickers == []


# ===========================================================================
# 2. RETRIEVER: Deterministic Ranking & K-Selection
# ===========================================================================


def test_retriever_ranks_by_ascending_distance():
    """Verify candidates are sorted in strictly ascending distance order."""
    # Candidates in random distance order
    candidates = [
        make_vector_result("c_mid", 0.40),
        make_vector_result("c_far", 0.85),
        make_vector_result("c_close", 0.12),
        make_vector_result("c_near", 0.25),
    ]
    sim_results = make_similarity_results(candidates)

    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=4)

    assert retrieval.total_candidates == 4
    assert retrieval.total_retrieved == 4
    assert [r.id for r in retrieval.results] == ["c_close", "c_near", "c_mid", "c_far"]
    assert [r.distance for r in retrieval.results] == [0.12, 0.25, 0.40, 0.85]


def test_retriever_selects_top_k_fewer_than_total():
    """Verify selecting k items when total candidates > k."""
    candidates = [make_vector_result(f"c_{i}", float(i) * 0.1) for i in range(10)]
    sim_results = make_similarity_results(candidates)

    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=3)

    assert retrieval.k == 3
    assert retrieval.total_candidates == 10
    assert retrieval.total_retrieved == 3
    assert len(retrieval.results) == 3
    assert [r.id for r in retrieval.results] == ["c_0", "c_1", "c_2"]


def test_retriever_selects_k_equal_to_total():
    """Verify selecting k items when total candidates == k."""
    candidates = [
        make_vector_result("c_1", 0.1),
        make_vector_result("c_2", 0.2),
    ]
    sim_results = make_similarity_results(candidates)

    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=2)

    assert retrieval.total_retrieved == 2
    assert [r.id for r in retrieval.results] == ["c_1", "c_2"]


def test_retriever_k_greater_than_total():
    """Verify retriever returns all available candidates when k > total candidates."""
    candidates = [
        make_vector_result("c_1", 0.15),
        make_vector_result("c_2", 0.25),
    ]
    sim_results = make_similarity_results(candidates)

    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=10)

    assert retrieval.k == 10
    assert retrieval.total_candidates == 2
    assert retrieval.total_retrieved == 2
    assert len(retrieval.results) == 2


def test_retriever_k_one():
    """Verify retriever behavior when k=1 (single closest match)."""
    candidates = [
        make_vector_result("c_mid", 0.45),
        make_vector_result("c_best", 0.05),
        make_vector_result("c_worst", 0.95),
    ]
    retriever = TopKRetriever()
    retrieval = retriever.retrieve(make_similarity_results(candidates), k=1)

    assert retrieval.total_retrieved == 1
    assert retrieval.results[0].id == "c_best"


def test_retriever_deterministic_tie_breaking():
    """Verify tie-breaking on identical distances preserves original sequence order."""
    candidates = [
        make_vector_result("c_tie_1", 0.30),
        make_vector_result("c_closer", 0.10),
        make_vector_result("c_tie_2", 0.30),
        make_vector_result("c_tie_3", 0.30),
        make_vector_result("c_far", 0.50),
    ]
    retriever = TopKRetriever()
    retrieval = retriever.retrieve(make_similarity_results(candidates), k=5)

    assert [r.id for r in retrieval.results] == [
        "c_closer",
        "c_tie_1",
        "c_tie_2",
        "c_tie_3",
        "c_far",
    ]


# ===========================================================================
# 3. RELEVANCE THRESHOLDING (Phase 9.10.2)
# ===========================================================================


def test_relevance_threshold_filters_distant_matches():
    """Verify candidate matches with distance > max_distance are excluded."""
    candidates = [
        make_vector_result("c_strong_1", 0.10),
        make_vector_result("c_strong_2", 0.25),
        make_vector_result("c_boundary", 0.50),
        make_vector_result("c_weak_1", 0.51),
        make_vector_result("c_weak_2", 0.85),
    ]
    sim_results = make_similarity_results(candidates)

    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=5, max_distance=0.50)

    assert retrieval.total_candidates == 5
    assert retrieval.total_retrieved == 3
    assert [r.id for r in retrieval.results] == [
        "c_strong_1",
        "c_strong_2",
        "c_boundary",
    ]
    assert retrieval.max_distance_threshold == 0.50


def test_relevance_threshold_all_filtered():
    """Verify when no candidates meet threshold, empty result list is returned."""
    candidates = [
        make_vector_result("c_1", 0.60),
        make_vector_result("c_2", 0.75),
    ]
    sim_results = make_similarity_results(candidates)

    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=5, max_distance=0.50)

    assert retrieval.total_candidates == 2
    assert retrieval.total_retrieved == 0
    assert retrieval.is_empty
    assert retrieval.results == []


def test_relevance_threshold_combined_with_k():
    """Verify K capping occurs after threshold filtering."""
    candidates = [
        make_vector_result("c_1", 0.10),
        make_vector_result("c_2", 0.20),
        make_vector_result("c_3", 0.30),
        make_vector_result("c_4", 0.40),
        make_vector_result("c_5", 0.80),  # Exceeds threshold
    ]
    sim_results = make_similarity_results(candidates)

    # Threshold keeps c_1, c_2, c_3, c_4. k=2 selects c_1, c_2.
    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=2, max_distance=0.50)

    assert retrieval.total_candidates == 5
    assert retrieval.total_retrieved == 2
    assert [r.id for r in retrieval.results] == ["c_1", "c_2"]


# ===========================================================================
# 4. METRIC & PROVENANCE PRESERVATION
# ===========================================================================


@pytest.mark.parametrize(
    "metric",
    ["cosine_distance", "l2_distance", "inner_product"],
)
def test_retriever_supported_metrics(metric):
    """Verify all supported distance metrics are handled correctly."""
    candidates = [
        make_vector_result("c_1", 0.10),
        make_vector_result("c_2", 0.30),
    ]
    sim_results = make_similarity_results(candidates, metric=metric)

    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=2)

    assert retrieval.distance_metric == metric
    assert len(retrieval.results) == 2


def test_retriever_preserves_full_provenance():
    """Verify document text and metadata are completely preserved without alteration."""
    raw_item = VectorSearchResult(
        id="chunk_aapl_10k_001",
        distance=0.18,
        document="Net sales for fiscal 2024 were $391,035 million.",
        metadata={
            "document_id": "doc_999",
            "ticker": "AAPL",
            "chunk_index": 42,
            "page_number": 27,
            "section": "Item 7 - MD&A",
        },
    )
    sim_results = make_similarity_results([raw_item])

    retriever = TopKRetriever()
    retrieval = retriever.retrieve(sim_results, k=1)

    result_item = retrieval.results[0]
    assert result_item.id == "chunk_aapl_10k_001"
    assert result_item.distance == 0.18
    assert result_item.document == "Net sales for fiscal 2024 were $391,035 million."
    assert result_item.document_id == "doc_999"
    assert result_item.ticker == "AAPL"
    assert result_item.metadata["chunk_index"] == 42
    assert result_item.metadata["page_number"] == 27
    assert result_item.metadata["section"] == "Item 7 - MD&A"


def test_retriever_accepts_raw_list_input():
    """Verify retriever accepts raw list of VectorSearchResult items."""
    candidates = [
        make_vector_result("c_1", 0.20),
        make_vector_result("c_2", 0.10),
    ]
    retriever = TopKRetriever()
    retrieval = retriever.retrieve(
        candidates,
        k=2,
        collection_name="custom_col",
        distance_metric="l2_distance",
    )

    assert retrieval.collection_name == "custom_col"
    assert retrieval.distance_metric == "l2_distance"
    assert [r.id for r in retrieval.results] == ["c_2", "c_1"]


def test_retriever_empty_candidates():
    """Verify retriever behavior when input candidates list is empty."""
    retriever = TopKRetriever()
    retrieval = retriever.retrieve([], k=5)

    assert retrieval.total_candidates == 0
    assert retrieval.total_retrieved == 0
    assert retrieval.is_empty
    assert retrieval.results == []


# ===========================================================================
# 5. CONFIGURATION INHERITANCE & CONVENIENCE FUNCTION
# ===========================================================================


def test_retriever_uses_default_config():
    """Verify retriever uses constructor default_config when not overridden."""
    default_cfg = TopKConfig(k=2, max_distance=0.30)
    retriever = TopKRetriever(default_config=default_cfg)

    candidates = [
        make_vector_result("c_1", 0.10),
        make_vector_result("c_2", 0.20),
        make_vector_result("c_3", 0.25),
        make_vector_result("c_4", 0.50),  # Exceeds max_distance
    ]
    retrieval = retriever.retrieve(make_similarity_results(candidates))

    # default k=2, default max_distance=0.30
    assert retrieval.k == 2
    assert retrieval.total_retrieved == 2
    assert [r.id for r in retrieval.results] == ["c_1", "c_2"]


def test_convenience_function_retrieve_top_k():
    """Verify standalone retrieve_top_k functional helper."""
    candidates = [
        make_vector_result("c_1", 0.35),
        make_vector_result("c_2", 0.15),
    ]
    sim_results = make_similarity_results(candidates)

    result = retrieve_top_k(sim_results, k=1)
    assert isinstance(result, TopKRetrievalResult)
    assert result.total_retrieved == 1
    assert result.results[0].id == "c_2"


# ===========================================================================
# 6. ERROR HANDLING & VALIDATION
# ===========================================================================


@pytest.mark.parametrize("bad_k", [0, -1, -10, 2.5, "3", True, False])
def test_retriever_rejects_invalid_k(bad_k):
    """Verify retriever raises InvalidKError on invalid k."""
    retriever = TopKRetriever()
    with pytest.raises(InvalidKError):
        retriever.retrieve([], k=bad_k)  # type: ignore


@pytest.mark.parametrize(
    "bad_threshold",
    [-0.1, -10.0, float("nan"), float("inf"), float("-inf"), True, False, "0.5"],
)
def test_retriever_rejects_invalid_threshold(bad_threshold):
    """Verify retriever raises InvalidRelevanceThresholdError on invalid threshold."""
    retriever = TopKRetriever()
    with pytest.raises(InvalidRelevanceThresholdError):
        retriever.retrieve([], max_distance=bad_threshold)  # type: ignore


def test_retriever_rejects_unsupported_metric():
    """Verify retriever raises UnsupportedDistanceMetricError on unknown metric."""
    candidates = [make_vector_result("c_1", 0.1)]
    sim_results = make_similarity_results(candidates, metric="levenshtein_distance")

    retriever = TopKRetriever()
    with pytest.raises(UnsupportedDistanceMetricError):
        retriever.retrieve(sim_results)


def test_retriever_rejects_invalid_search_results_type():
    """Verify retriever raises MalformedSearchResultError when input is not valid."""
    retriever = TopKRetriever()
    with pytest.raises(MalformedSearchResultError):
        retriever.retrieve("not_a_valid_result_object")  # type: ignore


def test_retriever_rejects_malformed_candidate_elements():
    """Verify retriever raises MalformedSearchResultError for non-model elements."""
    retriever = TopKRetriever()
    with pytest.raises(MalformedSearchResultError):
        retriever.retrieve([{"id": "dict_instead_of_model", "distance": 0.1}])  # type: ignore


def test_retriever_rejects_non_finite_candidate_distance():
    """Verify retriever raises error if candidate distance is NaN or Inf."""
    bad_result = make_vector_result("c_nan", 0.5)
    object.__setattr__(bad_result, "distance", float("nan"))
    retriever = TopKRetriever()
    with pytest.raises(MalformedSearchResultError):
        retriever.retrieve([bad_result])


def test_exceptions_inherit_from_top_k_retrieval_error():
    """Verify all custom exceptions inherit from TopKRetrievalError."""
    assert issubclass(InvalidKError, TopKRetrievalError)
    assert issubclass(InvalidRelevanceThresholdError, TopKRetrievalError)
    assert issubclass(UnsupportedDistanceMetricError, TopKRetrievalError)
    assert issubclass(MalformedSearchResultError, TopKRetrievalError)


# ===========================================================================
# 7. SAFE OPERATIONAL LOGGING
# ===========================================================================


def test_retriever_safe_operational_logging(caplog):
    """Verify retrieval logs summary statistics and never leaks raw document content."""
    secret_text = "CONFIDENTIAL_FINANCIAL_METRIC_98765"
    candidates = [
        make_vector_result("c_1", 0.10, text=secret_text),
        make_vector_result("c_2", 0.20, text="Public text"),
    ]
    sim_results = make_similarity_results(candidates, collection_name="company_AAPL")

    retriever = TopKRetriever()
    with caplog.at_level(logging.INFO, logger="app.services.top_k_retriever"):
        retrieval = retriever.retrieve(sim_results, k=2, max_distance=0.5)

    assert retrieval.total_retrieved == 2

    # Check logged messages
    log_text = caplog.text
    assert "Top-K retrieval completed for 'company_AAPL'" in log_text
    assert "candidates=2" in log_text
    assert "retrieved=2" in log_text
    assert "threshold=0.5" in log_text
    # Ensure document content is NOT in logs
    assert secret_text not in log_text
