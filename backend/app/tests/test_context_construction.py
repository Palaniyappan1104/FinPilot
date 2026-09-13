"""Unit tests for Phase 9.11: Context Construction.

Covers:
- ContextChunk and RetrievalContext model schemas, validation, and immutability
- Valid context construction from TopKRetrievalResult
- Multiple retrieved chunks, single chunk, and empty retrieval results
- Exact text preservation without rewriting, summarizing, or truncating
- Strict Phase 9.10 ordering preservation
- Complete provenance preservation (chunk_id, doc_id, ticker, pages, etc.)
- Distance and distance metric preservation
- Configurable context budgeting (flag vs. typed exception)
- Mandatory provenance validation (require_provenance flag)
- Input validation and typed exceptions
- Metadata immutability and preservation
- Helper properties (is_empty, document_ids, tickers, sources)
- Functional helper build_retrieval_context
- Safe operational logging without leaking chunk text
"""

import logging
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.context import ContextChunk, ContextConfig, RetrievalContext
from app.models.retrieval import TopKRetrievalResult
from app.models.vector_store import VectorSearchResult
from app.services.context_builder import (
    ContextBudgetExceededError,
    ContextBuilder,
    ContextConstructionError,
    InconsistentContextError,
    InvalidChunkDataError,
    MalformedRetrievalResultError,
    MissingProvenanceError,
    build_retrieval_context,
)

# ===========================================================================
# FIXTURES & HELPERS
# ===========================================================================


def make_vector_search_result(
    chunk_id: str = "chunk_001",
    distance: float = 0.15,
    document: str = "Apple Inc. reports record Q4 services revenue of $25.0B.",
    doc_id: str = "doc_aapl_10k_2024",
    ticker: str = "AAPL",
    doc_type: str = "10-K",
    pages: str = "15,16",
    section: str = "Item 7 - MD&A",
    source: str = "AAPL_2024_10K.pdf",
    chunk_index: int = 1,
) -> VectorSearchResult:
    """Construct a valid VectorSearchResult for testing."""
    return VectorSearchResult(
        id=chunk_id,
        distance=distance,
        document=document,
        metadata={
            "document_id": doc_id,
            "ticker": ticker,
            "document_type": doc_type,
            "page_numbers": pages,
            "section_name": section,
            "source_document": source,
            "chunk_index": chunk_index,
        },
        distance_metric="cosine_distance",
    )


def make_top_k_result(
    results: list,
    collection_name: str = "company_AAPL",
    metric: str = "cosine_distance",
    k: int = 5,
    threshold: float = None,
) -> TopKRetrievalResult:
    """Construct a valid TopKRetrievalResult for testing."""
    return TopKRetrievalResult(
        collection_name=collection_name,
        k=k,
        results=results,
        total_candidates=len(results),
        total_retrieved=len(results),
        distance_metric=metric,
        max_distance_threshold=threshold,
        retrieved_at=datetime.now(timezone.utc),
    )


# ===========================================================================
# 1. MODEL SCHEMA & VALIDATION TESTS
# ===========================================================================


def test_context_chunk_valid():
    """Verify ContextChunk instantiation and properties."""
    chunk = ContextChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        source="doc_1.pdf",
        ticker="AAPL",
        document_type="10-K",
        pages=[1, 2],
        section="MD&A",
        distance=0.12,
        text="Financial statement notes.",
        chunk_index=1,
        metadata={"custom": "value"},
    )
    assert chunk.chunk_id == "chunk_1"
    assert chunk.document_id == "doc_1"
    assert chunk.ticker == "AAPL"
    assert chunk.character_count == len("Financial statement notes.")
    assert chunk.word_count == 3


def test_context_chunk_immutability():
    """Verify ContextChunk is frozen."""
    chunk = ContextChunk(
        chunk_id="c1",
        distance=0.1,
        text="Sample text",
    )
    with pytest.raises(ValidationError):
        chunk.text = "Modified text"  # type: ignore


@pytest.mark.parametrize("empty_id", ["", "   "])
def test_context_chunk_rejects_empty_id(empty_id):
    """Verify ContextChunk rejects empty or whitespace-only chunk_id."""
    with pytest.raises(ValidationError):
        ContextChunk(chunk_id=empty_id, distance=0.1, text="Text")


@pytest.mark.parametrize("empty_text", ["", "   "])
def test_context_chunk_rejects_empty_text(empty_text):
    """Verify ContextChunk rejects empty or whitespace-only text."""
    with pytest.raises(ValidationError):
        ContextChunk(chunk_id="c1", distance=0.1, text=empty_text)


@pytest.mark.parametrize(
    "bad_dist",
    [float("nan"), float("inf"), float("-inf"), True, False, "0.15"],
)
def test_context_chunk_rejects_non_finite_distance(bad_dist):
    """Verify ContextChunk rejects non-finite or non-float distance."""
    with pytest.raises(ValidationError):
        ContextChunk(chunk_id="c1", distance=bad_dist, text="Valid text")


def test_context_config_defaults():
    """Verify ContextConfig defaults."""
    cfg = ContextConfig()
    assert cfg.max_characters is None
    assert cfg.raise_on_budget_exceeded is False
    assert cfg.require_provenance is False


def test_context_config_validation():
    """Verify ContextConfig validates positive max_characters."""
    with pytest.raises(ValidationError):
        ContextConfig(max_characters=0)
    with pytest.raises(ValidationError):
        ContextConfig(max_characters=-10)
    with pytest.raises(ValidationError):
        ContextConfig(max_characters=True)  # type: ignore


# ===========================================================================
# 2. CONTEXT CONSTRUCTION: BASIC & MULTI-CHUNK
# ===========================================================================


def test_build_context_with_multiple_chunks():
    """Verify construction with multiple Top-K retrieved chunks."""
    r1 = make_vector_search_result(
        chunk_id="c1",
        distance=0.10,
        document="First chunk text.",
        doc_id="doc_A",
        ticker="AAPL",
    )
    r2 = make_vector_search_result(
        chunk_id="c2",
        distance=0.25,
        document="Second chunk text with more details.",
        doc_id="doc_B",
        ticker="MSFT",
    )
    top_k = make_top_k_result([r1, r2], collection_name="multi_company")

    builder = ContextBuilder()
    context = builder.build(
        top_k,
        query="What were the revenues?",
    )

    assert isinstance(context, RetrievalContext)
    assert context.query == "What were the revenues?"
    assert context.collection_name == "multi_company"
    assert context.distance_metric == "cosine_distance"
    assert context.total_retrieved == 2
    assert context.has_evidence is True
    assert not context.is_empty
    assert len(context.chunks) == 2

    # Check first chunk
    assert context.chunks[0].chunk_id == "c1"
    assert context.chunks[0].text == "First chunk text."
    assert context.chunks[0].distance == 0.10
    assert context.chunks[0].document_id == "doc_A"
    assert context.chunks[0].ticker == "AAPL"

    # Check second chunk
    assert context.chunks[1].chunk_id == "c2"
    assert context.chunks[1].text == "Second chunk text with more details."
    assert context.chunks[1].distance == 0.25
    assert context.chunks[1].document_id == "doc_B"
    assert context.chunks[1].ticker == "MSFT"

    # Check totals
    expected_chars = len("First chunk text.") + len(
        "Second chunk text with more details."
    )
    assert context.total_characters == expected_chars
    assert context.total_words == len("First chunk text.".split()) + len(
        "Second chunk text with more details.".split()
    )


def test_build_context_with_single_chunk():
    """Verify construction with a single retrieved chunk."""
    r = make_vector_search_result(
        chunk_id="c_single",
        distance=0.05,
        document="Standalone single chunk.",
    )
    top_k = make_top_k_result([r])

    builder = ContextBuilder()
    context = builder.build(top_k)

    assert context.total_retrieved == 1
    assert context.has_evidence is True
    assert not context.is_empty
    assert context.chunks[0].chunk_id == "c_single"


# ===========================================================================
# 3. TEXT & ORDERING PRESERVATION (ZERO MUTATION)
# ===========================================================================


def test_exact_text_preservation_no_alteration():
    """Verify chunk text is preserved character-for-character without change."""
    verbatim_text = (
        "  \t\nItem 8. Consolidated Financial Statements.\n"
        "Net income was $96,995M compared to $99,803M in 2023.   \n\r"
    )
    r = make_vector_search_result(document=verbatim_text)
    top_k = make_top_k_result([r])

    builder = ContextBuilder()
    context = builder.build(top_k)

    assert context.chunks[0].text == verbatim_text
    assert (
        context.chunks[0].text is verbatim_text
        or context.chunks[0].text == verbatim_text
    )


def test_exact_ordering_preservation():
    """Verify chunks maintain the exact sequence provided by Top-K retrieval."""
    r_first = make_vector_search_result(chunk_id="rank_1", distance=0.10)
    r_second = make_vector_search_result(chunk_id="rank_2", distance=0.20)
    r_third = make_vector_search_result(chunk_id="rank_3", distance=0.30)
    top_k = make_top_k_result([r_first, r_second, r_third])

    builder = ContextBuilder()
    context = builder.build(top_k)

    assert [c.chunk_id for c in context.chunks] == ["rank_1", "rank_2", "rank_3"]
    assert [c.distance for c in context.chunks] == [0.10, 0.20, 0.30]


def test_metadata_immutability():
    """Verify metadata dictionary is preserved without in-place mutation."""
    original_meta = {
        "document_id": "doc_10k",
        "ticker": "AAPL",
        "source_document": "report.pdf",
        "page_numbers": "1,2",
    }
    r = VectorSearchResult(
        id="c1",
        distance=0.15,
        document="Some text content.",
        metadata=original_meta,
    )
    top_k = make_top_k_result([r])

    builder = ContextBuilder()
    context = builder.build(top_k)

    # Modify original metadata dictionary
    original_meta["mutated_key"] = "SHOULD_NOT_LEAK"

    assert "mutated_key" not in context.chunks[0].metadata
    assert context.chunks[0].metadata["ticker"] == "AAPL"


# ===========================================================================
# 4. PROVENANCE EXTRACTION & CITATION FIELDS
# ===========================================================================


def test_complete_provenance_preservation():
    """Verify all citation provenance fields are correctly mapped from metadata."""
    r = VectorSearchResult(
        id="chunk_sec_042",
        distance=0.18,
        document="Operating margin expanded 120 bps.",
        metadata={
            "document_id": "doc_sec_2024",
            "ticker": "NVDA",
            "document_type": "10-Q",
            "page_numbers": "12,13,14",
            "start_page": 12,
            "end_page": 14,
            "section_name": "Operating Results",
            "source_document": "NVDA_Q3_2024.pdf",
            "chunk_index": 42,
        },
    )
    top_k = make_top_k_result([r])

    builder = ContextBuilder()
    context = builder.build(top_k)
    chunk = context.chunks[0]

    assert chunk.chunk_id == "chunk_sec_042"
    assert chunk.document_id == "doc_sec_2024"
    assert chunk.ticker == "NVDA"
    assert chunk.document_type == "10-Q"
    assert chunk.pages == [12, 13, 14]
    assert chunk.section == "Operating Results"
    assert chunk.source == "NVDA_Q3_2024.pdf"
    assert chunk.chunk_index == 42
    assert chunk.distance == 0.18


def test_retrieval_context_helper_properties():
    """Verify helper properties of RetrievalContext."""
    r1 = make_vector_search_result(
        chunk_id="c1", doc_id="doc_A", ticker="AAPL", source="aapl.pdf"
    )
    r2 = make_vector_search_result(
        chunk_id="c2", doc_id="doc_B", ticker="MSFT", source="msft.pdf"
    )
    r3 = make_vector_search_result(
        chunk_id="c3", doc_id="doc_A", ticker="AAPL", source="aapl.pdf"
    )
    top_k = make_top_k_result([r1, r2, r3])

    builder = ContextBuilder()
    context = builder.build(top_k)

    assert context.document_ids == ["doc_A", "doc_B"]
    assert context.tickers == ["AAPL", "MSFT"]
    assert context.sources == ["aapl.pdf", "msft.pdf"]


# ===========================================================================
# 5. EMPTY RETRIEVAL HANDLING (EXPLICIT NO-EVIDENCE)
# ===========================================================================


def test_empty_retrieval_result_handling():
    """Verify empty TopKRetrievalResult produces an explicit no-evidence context."""
    empty_top_k = make_top_k_result([], collection_name="empty_collection")

    builder = ContextBuilder()
    context = builder.build(empty_top_k, query="Query with no matches")

    assert context.is_empty is True
    assert context.has_evidence is False
    assert context.total_retrieved == 0
    assert context.total_characters == 0
    assert context.total_words == 0
    assert context.chunks == []
    assert context.document_ids == []
    assert context.tickers == []
    assert context.sources == []
    assert context.query == "Query with no matches"


# ===========================================================================
# 6. CONTEXT BUDGETING (SIZE HANDLING)
# ===========================================================================


def test_context_budget_within_limit():
    """Verify context under character budget has exceeds_budget=False."""
    r = make_vector_search_result(document="Short text.")
    top_k = make_top_k_result([r])

    cfg = ContextConfig(max_characters=100)
    builder = ContextBuilder(default_config=cfg)
    context = builder.build(top_k)

    assert context.exceeds_budget is False
    assert context.total_characters == len("Short text.")


def test_context_budget_exceeded_flag_mode():
    """Verify when limit exceeded and raise=False, exceeds_budget is True."""
    full_text = "This is a longer document chunk that will exceed the configured limit."
    r = make_vector_search_result(document=full_text)
    top_k = make_top_k_result([r])

    cfg = ContextConfig(max_characters=20, raise_on_budget_exceeded=False)
    builder = ContextBuilder(default_config=cfg)
    context = builder.build(top_k)

    assert context.exceeds_budget is True
    # Verify text was NOT truncated or modified
    assert context.chunks[0].text == full_text
    assert context.total_characters == len(full_text)


def test_context_budget_exceeded_raises_error():
    """Verify when limit exceeded and raise=True, raises typed error."""
    r = make_vector_search_result(document="Text exceeding threshold.")
    top_k = make_top_k_result([r])

    cfg = ContextConfig(max_characters=10, raise_on_budget_exceeded=True)
    builder = ContextBuilder(default_config=cfg)

    with pytest.raises(ContextBudgetExceededError) as exc_info:
        builder.build(top_k)
    assert exc_info.value.code == "CONTEXT_BUDGET_EXCEEDED"


# ===========================================================================
# 7. PROVENANCE ENFORCEMENT (require_provenance)
# ===========================================================================


def test_require_provenance_passes_when_present():
    """Verify require_provenance succeeds when document_id is present."""
    r = make_vector_search_result(doc_id="doc_valid_123")
    top_k = make_top_k_result([r])

    cfg = ContextConfig(require_provenance=True)
    builder = ContextBuilder(default_config=cfg)
    context = builder.build(top_k)

    assert context.chunks[0].document_id == "doc_valid_123"


def test_require_provenance_raises_when_missing():
    """Verify require_provenance raises error when document_id is absent."""
    # Chunk with empty document_id in metadata
    r = VectorSearchResult(
        id="c_no_doc",
        distance=0.2,
        document="Some chunk without doc id.",
        metadata={},
    )
    top_k = make_top_k_result([r])

    cfg = ContextConfig(require_provenance=True)
    builder = ContextBuilder(default_config=cfg)

    with pytest.raises(MissingProvenanceError) as exc_info:
        builder.build(top_k)
    assert exc_info.value.code == "MISSING_PROVENANCE"


# ===========================================================================
# 8. ERROR HANDLING & VALIDATION
# ===========================================================================


def test_rejects_non_top_k_retrieval_result():
    """Verify builder raises MalformedRetrievalResultError on invalid input."""
    builder = ContextBuilder()
    with pytest.raises(MalformedRetrievalResultError):
        builder.build("not_a_retrieval_result")  # type: ignore


@pytest.mark.parametrize("bad_col", ["", "   ", None])
def test_rejects_empty_collection_name(bad_col):
    """Verify builder raises InconsistentContextError on empty collection name."""
    top_k = TopKRetrievalResult(
        collection_name="valid_col",
        k=5,
        results=[],
        total_candidates=0,
        total_retrieved=0,
    )
    object.__setattr__(top_k, "collection_name", bad_col)

    builder = ContextBuilder()
    with pytest.raises(InconsistentContextError):
        builder.build(top_k)


def test_rejects_unsupported_distance_metric():
    """Verify builder raises InconsistentContextError on unsupported metric."""
    top_k = TopKRetrievalResult(
        collection_name="valid_col",
        k=5,
        results=[],
        total_candidates=0,
        total_retrieved=0,
        distance_metric="cosine_distance",
    )
    object.__setattr__(top_k, "distance_metric", "unsupported_metric_xyz")

    builder = ContextBuilder()
    with pytest.raises(InconsistentContextError):
        builder.build(top_k)


def test_rejects_non_vector_search_result_in_candidates():
    """Verify builder raises error if candidate is not VectorSearchResult."""
    top_k = TopKRetrievalResult(
        collection_name="valid_col",
        k=5,
        results=[],
        total_candidates=1,
        total_retrieved=1,
    )
    object.__setattr__(top_k, "results", [{"not": "a_vector_search_result"}])

    builder = ContextBuilder()
    with pytest.raises(MalformedRetrievalResultError):
        builder.build(top_k)


def test_rejects_empty_chunk_text():
    """Verify builder raises error if chunk document text is whitespace."""
    r = make_vector_search_result(document="Valid text")
    object.__setattr__(r, "document", "   ")
    top_k = make_top_k_result([r])

    builder = ContextBuilder()
    with pytest.raises(InvalidChunkDataError):
        builder.build(top_k)


def test_rejects_non_finite_chunk_distance():
    """Verify builder raises InvalidChunkDataError if chunk distance is non-finite."""
    r = make_vector_search_result()
    object.__setattr__(r, "distance", float("nan"))
    top_k = make_top_k_result([r])

    builder = ContextBuilder()
    with pytest.raises(InvalidChunkDataError):
        builder.build(top_k)


def test_exceptions_inherit_from_context_construction_error():
    """Verify all custom exceptions inherit from ContextConstructionError."""
    assert issubclass(MalformedRetrievalResultError, ContextConstructionError)
    assert issubclass(InvalidChunkDataError, ContextConstructionError)
    assert issubclass(MissingProvenanceError, ContextConstructionError)
    assert issubclass(InconsistentContextError, ContextConstructionError)
    assert issubclass(ContextBudgetExceededError, ContextConstructionError)


# ===========================================================================
# 9. DETERMINISM & CONVENIENCE FUNCTION
# ===========================================================================


def test_deterministic_repeated_construction():
    """Verify repeated construction from same input yields identical output."""
    r1 = make_vector_search_result(chunk_id="c1", distance=0.10)
    r2 = make_vector_search_result(chunk_id="c2", distance=0.20)
    top_k = make_top_k_result([r1, r2])

    builder = ContextBuilder()
    c1 = builder.build(top_k, query="deterministic test")
    c2 = builder.build(top_k, query="deterministic test")

    assert c1.collection_name == c2.collection_name
    assert c1.total_characters == c2.total_characters
    assert c1.total_words == c2.total_words
    assert [c.chunk_id for c in c1.chunks] == [c.chunk_id for c in c2.chunks]
    assert [c.distance for c in c1.chunks] == [c.distance for c in c2.chunks]
    assert [c.text for c in c1.chunks] == [c.text for c in c2.chunks]


def test_convenience_function_build_retrieval_context():
    """Verify standalone build_retrieval_context function."""
    r = make_vector_search_result(chunk_id="c_func", distance=0.15)
    top_k = make_top_k_result([r])

    context = build_retrieval_context(
        retrieval_result=top_k,
        query="Quick test",
        max_characters=5000,
    )

    assert isinstance(context, RetrievalContext)
    assert context.query == "Quick test"
    assert context.total_retrieved == 1
    assert context.chunks[0].chunk_id == "c_func"


# ===========================================================================
# 10. SAFE OPERATIONAL LOGGING
# ===========================================================================


def test_context_builder_safe_operational_logging(caplog):
    """Verify context builder logs summary metrics and never leaks chunk text."""
    secret_text = "CONFIDENTIAL_MERGER_DISCUSSION_CODE_778899"
    r = make_vector_search_result(document=secret_text)
    top_k = make_top_k_result([r], collection_name="company_AAPL")

    builder = ContextBuilder()
    with caplog.at_level(logging.INFO, logger="app.services.context_builder"):
        context = builder.build(top_k, query="Confidential query")

    assert context.total_retrieved == 1
    log_text = caplog.text
    assert "Context constructed for 'company_AAPL'" in log_text
    assert "chunks=1" in log_text
    assert "has_evidence=True" in log_text
    # Ensure raw secret document text is NOT leaked in logs
    assert secret_text not in log_text
