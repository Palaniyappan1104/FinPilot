"""Unit and integration tests for Phase 9.15: Handling Irrelevant Queries.

Covers:
1. Empty context handling (has_evidence=False, 0 chunks)
2. Completely irrelevant query / distant retrieval chunks
3. All chunks above distance threshold
4. All chunks below distance threshold
5. Mixed relevance chunks (partial relevance filtering)
6. Exact threshold boundary (distance == max_distance)
7. Off-by-boundary tests (distance = max_distance - 0.0001 vs + 0.0001)
8. Configurable max_distance threshold
9. Configurable min_relevant_chunks requirement
10. require_all_relevant policy flag
11. best_distance metric calculation
12. Accurate candidate chunk counts (total, relevant, rejected)
13. Immutability of original RetrievalContext
14. Derived context provenance and ranking preservation
15. Zero LLM calls when context is irrelevant or empty
16. Structured insufficient_evidence output schema adherence
17. Relevant query proceeds to LLM generation
18. Grounding validation remains active on filtered derived context
19. Zero external API calls / zero web search
20. Malformed context input rejection (TypeError)
21. RelevanceConfig validation (finite float, positive integer, types)
22. Convenience function check_retrieval_relevance parity
23. ResearchAnalystAgent.run() integration with AgentResult
24. ResearchAnalystInput per-query relevance_config override
25. Custom agent-level default relevance_config initialization
26. Deterministic repeated evaluation
"""

from typing import Any, List, Optional
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.agents.base import AgentResult
from app.agents.research import ResearchAnalystAgent
from app.agents.research_schema import (
    InvalidProvenanceError,
    ResearchAnalysisOutput,
    ResearchAnalystInput,
)
from app.core.llm.base import LLMProvider, LLMResponse
from app.models.context import ContextChunk, RetrievalContext
from app.models.relevance import (
    REASON_INSUFFICIENT_RELEVANT_CHUNKS,
    REASON_NO_EVIDENCE_WITHIN_THRESHOLD,
    REASON_NO_RETRIEVED_EVIDENCE,
    REASON_NOT_ALL_CHUNKS_RELEVANT,
    REASON_SUFFICIENT_EVIDENCE,
    RelevanceConfig,
)
from app.services.relevance_checker import (
    RelevanceChecker,
    check_retrieval_relevance,
)

# ---------------------------------------------------------------------------
# Test Helpers and Fixtures
# ---------------------------------------------------------------------------


class MockLLMProvider(LLMProvider):
    """Deterministic offline Mock LLM provider for testing."""

    def __init__(self, response_text: str = "", name: str = "mock_provider") -> None:
        super().__init__()
        self.response_text = response_text
        self.call_count = 0
        self.last_prompt = ""
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        schema: Optional[Any] = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_prompt = prompt
        return LLMResponse(
            content=self.response_text,
            provider=self._name,
            model="mock-deterministic-llm",
        )


def make_chunk(
    chunk_id: str,
    distance: float,
    text: str = "Quarterly revenue grew by 15% year-over-year.",
    doc_id: str = "doc-aapl-2023",
    source: str = "AAPL_10K_2023.pdf",
    pages: Optional[List[int]] = None,
    ticker: str = "AAPL",
    chunk_index: int = 0,
) -> ContextChunk:
    """Helper to build a deterministic ContextChunk."""
    return ContextChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        source=source,
        ticker=ticker,
        doc_type="10-K",
        pages=pages or [10],
        section="Item 7",
        distance=distance,
        text=text,
        chunk_index=chunk_index,
        metadata={"year": 2023, "quarter": "Q4"},
    )


def make_context(
    chunks: List[ContextChunk],
    query: str = "What was Apple's revenue growth?",
    collection_name: str = "sec_filings",
) -> RetrievalContext:
    """Helper to build a deterministic RetrievalContext."""
    has_evidence = len(chunks) > 0
    total_chars = sum(c.character_count for c in chunks)
    total_words = sum(c.word_count for c in chunks)
    return RetrievalContext(
        query=query,
        collection_name=collection_name,
        distance_metric="cosine_distance",
        total_retrieved=len(chunks),
        total_characters=total_chars,
        total_words=total_words,
        chunks=chunks,
        has_evidence=has_evidence,
        max_distance_threshold=None,
        exceeds_budget=False,
        built_at="2026-09-14T00:00:00Z",
        retrieval_metadata={"source_system": "chromadb_test"},
    )


# ---------------------------------------------------------------------------
# 1. RelevanceConfig Model Tests
# ---------------------------------------------------------------------------


def test_relevance_config_defaults() -> None:
    """Verify default RelevanceConfig values and frozen immutability."""
    cfg = RelevanceConfig()
    assert cfg.max_distance == 0.65
    assert cfg.min_relevant_chunks == 1
    assert cfg.require_all_relevant is False

    with pytest.raises(ValidationError):
        cfg.max_distance = 0.50  # type: ignore[misc]


def test_relevance_config_valid_custom_values() -> None:
    """Verify custom configuration initialization."""
    cfg = RelevanceConfig(
        max_distance=0.45,
        min_relevant_chunks=3,
        require_all_relevant=True,
    )
    assert cfg.max_distance == 0.45
    assert cfg.min_relevant_chunks == 3
    assert cfg.require_all_relevant is True


@pytest.mark.parametrize(
    "invalid_dist", [-0.1, -1.0, float("nan"), float("inf"), "invalid", True]
)
def test_relevance_config_invalid_distance(invalid_dist: Any) -> None:
    """Verify rejection of invalid distance thresholds."""
    with pytest.raises(ValidationError):
        RelevanceConfig(max_distance=invalid_dist)


@pytest.mark.parametrize("invalid_chunks", [0, -1, -5, "two", 1.5, True])
def test_relevance_config_invalid_min_chunks(invalid_chunks: Any) -> None:
    """Verify rejection of invalid min_relevant_chunks."""
    with pytest.raises(ValidationError):
        RelevanceConfig(min_relevant_chunks=invalid_chunks)


# ---------------------------------------------------------------------------
# 2. RelevanceChecker Unit Tests
# ---------------------------------------------------------------------------


def test_empty_retrieval_context_handling() -> None:
    """Verify empty context (0 chunks) produces is_relevant=False and rejection."""
    empty_ctx = make_context(chunks=[])
    checker = RelevanceChecker()
    result = checker.check_relevance(empty_ctx)

    assert result.is_relevant is False
    assert result.reason == REASON_NO_RETRIEVED_EVIDENCE
    assert result.total_chunks == 0
    assert result.relevant_chunks_count == 0
    assert result.rejected_chunks_count == 0
    assert result.best_distance is None
    assert result.derived_context is None
    assert result.relevant_chunks == []
    assert result.rejected_chunks == []


def test_all_chunks_above_threshold_rejected() -> None:
    """Verify all chunks exceeding max_distance are rejected."""
    c1 = make_chunk("c1", distance=0.72)
    c2 = make_chunk("c2", distance=0.85)
    c3 = make_chunk("c3", distance=0.91)
    ctx = make_context(
        [c1, c2, c3], query="Best luxury beach hotels in Honolulu Hawaii"
    )

    checker = RelevanceChecker()
    result = checker.check_relevance(ctx)

    assert result.is_relevant is False
    assert result.reason == REASON_NO_EVIDENCE_WITHIN_THRESHOLD
    assert result.total_chunks == 3
    assert result.relevant_chunks_count == 0
    assert result.rejected_chunks_count == 3
    assert result.best_distance == 0.72
    assert result.threshold_used == 0.65
    assert result.derived_context is None
    assert len(result.rejected_chunks) == 3


def test_all_chunks_below_threshold_accepted() -> None:
    """Verify all chunks within max_distance are accepted."""
    c1 = make_chunk("c1", distance=0.30)
    c2 = make_chunk("c2", distance=0.45)
    ctx = make_context([c1, c2])

    checker = RelevanceChecker()
    result = checker.check_relevance(ctx)

    assert result.is_relevant is True
    assert result.reason == REASON_SUFFICIENT_EVIDENCE
    assert result.total_chunks == 2
    assert result.relevant_chunks_count == 2
    assert result.rejected_chunks_count == 0
    assert result.best_distance == 0.30
    assert result.derived_context is not None
    assert len(result.derived_context.chunks) == 2
    assert result.derived_context.chunks[0].chunk_id == "c1"
    assert result.derived_context.chunks[1].chunk_id == "c2"


def test_mixed_relevance_partial_filtering() -> None:
    """Verify partial relevance: weak chunks filtered out while relevant survive."""
    c1 = make_chunk("c1", distance=0.35, text="Apple Services revenue was $85B.")
    c2 = make_chunk(
        "c2", distance=0.78, text="Random irrelevant legal boilerplate disclosure."
    )
    c3 = make_chunk("c3", distance=0.55, text="Gross margin expanded to 44.1%.")
    ctx = make_context([c1, c2, c3])

    checker = RelevanceChecker()
    result = checker.check_relevance(ctx)

    assert result.is_relevant is True
    assert result.reason == REASON_SUFFICIENT_EVIDENCE
    assert result.total_chunks == 3
    assert result.relevant_chunks_count == 2
    assert result.rejected_chunks_count == 1
    assert result.best_distance == 0.35

    # Check derived context
    derived = result.derived_context
    assert derived is not None
    assert derived.total_retrieved == 2
    assert len(derived.chunks) == 2
    assert [c.chunk_id for c in derived.chunks] == ["c1", "c3"]
    assert derived.chunks[0].distance == 0.35
    assert derived.chunks[1].distance == 0.55
    assert derived.retrieval_metadata["relevance_filtered"] is True
    assert derived.retrieval_metadata["original_chunks_count"] == 3


def test_original_context_immutability() -> None:
    """Verify original RetrievalContext is never mutated by relevance filtering."""
    c1 = make_chunk("c1", distance=0.40)
    c2 = make_chunk("c2", distance=0.88)
    ctx = make_context([c1, c2])

    checker = RelevanceChecker()
    _ = checker.check_relevance(ctx)

    # Original context remains strictly unchanged
    assert len(ctx.chunks) == 2
    assert ctx.total_retrieved == 2
    assert ctx.chunks[0].chunk_id == "c1"
    assert ctx.chunks[1].chunk_id == "c2"
    assert "relevance_filtered" not in ctx.retrieval_metadata


def test_exact_distance_boundary() -> None:
    """Verify exact boundary: chunk with distance == max_distance is relevant."""
    c_exact = make_chunk("c_exact", distance=0.65)
    ctx = make_context([c_exact])

    checker = RelevanceChecker(RelevanceConfig(max_distance=0.65))
    result = checker.check_relevance(ctx)

    assert result.is_relevant is True
    assert result.relevant_chunks_count == 1
    assert result.rejected_chunks_count == 0


def test_distance_off_by_boundary() -> None:
    """Verify precision around threshold: 0.6499 is accepted, 0.6501 is rejected."""
    c_just_below = make_chunk("c_below", distance=0.6499)
    c_just_above = make_chunk("c_above", distance=0.6501)
    ctx = make_context([c_just_below, c_just_above])

    checker = RelevanceChecker(RelevanceConfig(max_distance=0.65))
    result = checker.check_relevance(ctx)

    assert result.is_relevant is True
    assert result.relevant_chunks_count == 1
    assert result.rejected_chunks_count == 1
    assert result.relevant_chunks[0].chunk_id == "c_below"
    assert result.rejected_chunks[0].chunk_id == "c_above"


def test_configurable_max_distance() -> None:
    """Verify that customizing max_distance strictly alters the filtering boundary."""
    c1 = make_chunk("c1", distance=0.55)
    ctx = make_context([c1])

    # With strict max_distance=0.50 -> rejected
    strict_checker = RelevanceChecker(RelevanceConfig(max_distance=0.50))
    strict_res = strict_checker.check_relevance(ctx)
    assert strict_res.is_relevant is False
    assert strict_res.reason == REASON_NO_EVIDENCE_WITHIN_THRESHOLD

    # With lenient max_distance=0.60 -> accepted
    lenient_checker = RelevanceChecker(RelevanceConfig(max_distance=0.60))
    lenient_res = lenient_checker.check_relevance(ctx)
    assert lenient_res.is_relevant is True
    assert lenient_res.reason == REASON_SUFFICIENT_EVIDENCE


def test_min_relevant_chunks_policy() -> None:
    """Verify min_relevant_chunks requirement."""
    c1 = make_chunk("c1", distance=0.30)  # relevant
    c2 = make_chunk("c2", distance=0.75)  # weak
    ctx = make_context([c1, c2])

    # Require at least 2 relevant chunks -> fails
    cfg = RelevanceConfig(max_distance=0.65, min_relevant_chunks=2)
    checker = RelevanceChecker(cfg)
    result = checker.check_relevance(ctx)

    assert result.is_relevant is False
    assert result.reason == REASON_INSUFFICIENT_RELEVANT_CHUNKS
    assert result.relevant_chunks_count == 1
    assert result.min_chunks_required == 2
    assert result.derived_context is None


def test_require_all_relevant_policy() -> None:
    """Verify require_all_relevant=True rejects if any chunk exceeds threshold."""
    c1 = make_chunk("c1", distance=0.30)
    c2 = make_chunk("c2", distance=0.70)
    ctx = make_context([c1, c2])

    cfg = RelevanceConfig(max_distance=0.65, require_all_relevant=True)
    checker = RelevanceChecker(cfg)
    result = checker.check_relevance(ctx)

    assert result.is_relevant is False
    assert result.reason == REASON_NOT_ALL_CHUNKS_RELEVANT
    assert result.rejected_chunks_count == 1


def test_derived_context_preserves_ranking_and_provenance() -> None:
    """Verify derived context preserves exact chunk ordering, text, and provenance."""
    c1 = make_chunk("c1", distance=0.20, text="First text", chunk_index=1, pages=[1])
    c2 = make_chunk(
        "c2", distance=0.80, text="Second weak text", chunk_index=2, pages=[2]
    )
    c3 = make_chunk("c3", distance=0.40, text="Third text", chunk_index=3, pages=[3])
    ctx = make_context([c1, c2, c3])

    checker = RelevanceChecker()
    result = checker.check_relevance(ctx)

    derived = result.derived_context
    assert derived is not None
    assert len(derived.chunks) == 2
    assert derived.chunks[0].chunk_id == "c1"
    assert derived.chunks[0].text == "First text"
    assert derived.chunks[0].pages == [1]
    assert derived.chunks[1].chunk_id == "c3"
    assert derived.chunks[1].text == "Third text"
    assert derived.chunks[1].pages == [3]


def test_relevance_checker_type_error_on_invalid_context() -> None:
    """Verify TypeError when non-RetrievalContext object is supplied."""
    checker = RelevanceChecker()
    with pytest.raises(TypeError, match="must be an instance of RetrievalContext"):
        checker.check_relevance("not a context")  # type: ignore[arg-type]


def test_convenience_function_parity() -> None:
    """Verify check_retrieval_relevance returns the same result as RelevanceChecker."""
    c1 = make_chunk("c1", distance=0.45)
    ctx = make_context([c1])

    res1 = check_retrieval_relevance(ctx)
    res2 = RelevanceChecker().check_relevance(ctx)

    assert res1.is_relevant == res2.is_relevant
    assert res1.reason == res2.reason
    assert res1.best_distance == res2.best_distance


# ---------------------------------------------------------------------------
# 3. ResearchAnalystAgent Integration Tests
# ---------------------------------------------------------------------------


def test_irrelevant_query_skips_llm_call() -> None:
    """Verify that when context is irrelevant, agent does NOT call the LLM."""
    mock_llm = MockLLMProvider()
    agent = ResearchAnalystAgent(provider=mock_llm)

    c1 = make_chunk("c1", distance=0.88, text="Completely unrelated document.")
    ctx = make_context([c1], query="What was the weather in Maui?")

    output = agent.analyze(query="What was the weather in Maui?", context=ctx)

    # LLM must not be called
    assert mock_llm.call_count == 0
    assert output.insufficient_evidence is True
    assert output.confidence == 0.0
    assert output.key_findings == []
    assert output.evidence == []
    assert "Insufficient evidence" in output.answer
    assert "No retrieved chunks met the maximum distance threshold" in (
        output.insufficient_reason or ""
    )


def test_empty_retrieval_context_skips_llm_call() -> None:
    """Verify that when context is empty, agent does NOT call the LLM."""
    mock_llm = MockLLMProvider()
    agent = ResearchAnalystAgent(provider=mock_llm)
    empty_ctx = make_context(chunks=[], query="Find quarterly report")

    output = agent.analyze(query="Find quarterly report", context=empty_ctx)

    assert mock_llm.call_count == 0
    assert output.insufficient_evidence is True
    assert output.confidence == 0.0
    assert "empty retrieval" in (output.insufficient_reason or "")


def test_relevant_query_calls_llm_with_derived_context() -> None:
    """Verify relevant query calls LLM and prompt has ONLY relevant chunks."""
    # LLM response conforming to ResearchAnalysisOutput
    import json

    payload = {
        "query": "What was Apple's revenue?",
        "answer": "Apple revenue was $85B according to Q4 disclosures.",
        "key_findings": [
            {
                "claim": "Apple revenue reached $85B.",
                "evidence": [
                    {
                        "chunk_id": "c1",
                        "document_id": "doc-aapl-2023",
                        "source_document": "AAPL_10K_2023.pdf",
                        "page_numbers": [10],
                    }
                ],
            }
        ],
        "evidence": [
            {
                "chunk_id": "c1",
                "document_id": "doc-aapl-2023",
                "source_document": "AAPL_10K_2023.pdf",
                "page_numbers": [10],
            }
        ],
        "confidence": 0.95,
        "insufficient_evidence": False,
        "insufficient_reason": None,
    }
    mock_llm = MockLLMProvider(response_text=json.dumps(payload))
    agent = ResearchAnalystAgent(provider=mock_llm)

    c1 = make_chunk("c1", distance=0.30, text="Apple revenue reached $85B.", pages=[10])
    c2 = make_chunk(
        "c2",
        distance=0.85,
        text="Weak irrelevant chunk that must not appear in prompt.",
    )
    ctx = make_context([c1, c2])

    output = agent.analyze(query="What was Apple's revenue?", context=ctx)

    assert mock_llm.call_count == 1
    assert output.insufficient_evidence is False
    assert output.confidence == 0.95
    assert len(output.key_findings) == 1
    assert output.evidence[0].chunk_id == "c1"

    # Verify that the weak chunk was NOT included in the LLM prompt!
    assert "c1" in mock_llm.last_prompt
    assert "Weak irrelevant chunk" not in mock_llm.last_prompt
    assert "c2" not in mock_llm.last_prompt


def test_grounding_validation_still_enforced_on_relevant_context() -> None:
    """Verify citations to filtered-out chunks fail grounding validation."""
    import json

    # LLM hallucinates citation to filtered out chunk 'c2'
    hallucinated_payload = {
        "query": "What was Apple's revenue?",
        "answer": "Apple revenue was $85B.",
        "key_findings": [
            {
                "claim": "Apple revenue reached $85B.",
                "evidence": [
                    {
                        "chunk_id": "c2",
                        "document_id": "doc-aapl-2023",
                        "source_document": "AAPL_10K_2023.pdf",
                        "page_numbers": [10],
                    }
                ],
            }
        ],
        "evidence": [
            {
                "chunk_id": "c2",
                "document_id": "doc-aapl-2023",
                "source_document": "AAPL_10K_2023.pdf",
                "page_numbers": [10],
            }
        ],
        "confidence": 0.90,
        "insufficient_evidence": False,
        "insufficient_reason": None,
    }
    mock_llm = MockLLMProvider(response_text=json.dumps(hallucinated_payload))
    agent = ResearchAnalystAgent(provider=mock_llm)

    c1 = make_chunk("c1", distance=0.30)
    c2 = make_chunk("c2", distance=0.85)  # filtered out by relevance checker
    ctx = make_context([c1, c2])

    # Grounding validation should reject citation to c2
    with pytest.raises(
        InvalidProvenanceError, match="does not exist in RetrievalContext"
    ):
        agent.analyze(query="What was Apple's revenue?", context=ctx)


def test_agent_run_method_with_irrelevant_query() -> None:
    """Verify BaseAgent.run() returns AgentResult.create_success on irrelevant query."""
    mock_llm = MockLLMProvider()
    agent = ResearchAnalystAgent(provider=mock_llm)

    c1 = make_chunk("c1", distance=0.95)
    ctx = make_context([c1])
    input_data = ResearchAnalystInput(query="Irrelevant question", context=ctx)

    result: AgentResult = agent.run(input_data)

    assert result.success is True
    assert isinstance(result.data, ResearchAnalysisOutput)
    assert result.data.insufficient_evidence is True
    assert result.confidence == 0.0
    assert mock_llm.call_count == 0


def test_agent_run_with_per_query_relevance_config_override() -> None:
    """Verify ResearchAnalystInput.relevance_config overrides agent default."""
    mock_llm = MockLLMProvider()
    agent = ResearchAnalystAgent(
        provider=mock_llm,
        relevance_config=RelevanceConfig(max_distance=0.65),
    )

    c1 = make_chunk("c1", distance=0.55)  # <= 0.65, but > 0.50
    ctx = make_context([c1])

    # Override with strict max_distance=0.50
    strict_cfg = RelevanceConfig(max_distance=0.50)
    input_data = ResearchAnalystInput(
        query="Test query",
        context=ctx,
        relevance_config=strict_cfg,
    )

    result = agent.run(input_data)

    assert result.success is True
    assert result.data.insufficient_evidence is True
    assert mock_llm.call_count == 0


def test_agent_constructor_custom_relevance_config() -> None:
    """Verify agent initialized with custom relevance_config enforces it by default."""
    mock_llm = MockLLMProvider()
    agent = ResearchAnalystAgent(
        provider=mock_llm,
        relevance_config=RelevanceConfig(max_distance=0.40),
    )
    assert agent.relevance_config.max_distance == 0.40
    assert agent.relevance_checker.default_config.max_distance == 0.40

    c1 = make_chunk("c1", distance=0.50)
    ctx = make_context([c1])

    output = agent.analyze(query="Some query", context=ctx)
    assert output.insufficient_evidence is True
    assert mock_llm.call_count == 0


def test_no_external_financial_or_web_calls_on_irrelevant_query() -> None:
    """Verify zero external financial API or web search calls are made."""
    mock_llm = MockLLMProvider()
    agent = ResearchAnalystAgent(provider=mock_llm)
    c1 = make_chunk("c1", distance=0.99)
    ctx = make_context([c1])

    with patch("urllib.request.urlopen") as mock_url, patch("requests.get") as mock_get:
        output = agent.analyze(query="What is the forecast?", context=ctx)
        assert mock_url.call_count == 0
        assert mock_get.call_count == 0
        assert output.insufficient_evidence is True


def test_repeated_deterministic_relevance_evaluation() -> None:
    """Verify that running relevance check 50 times produces identical results."""
    c1 = make_chunk("c1", distance=0.42)
    c2 = make_chunk("c2", distance=0.78)
    ctx = make_context([c1, c2])

    checker = RelevanceChecker()
    first_res = checker.check_relevance(ctx)

    for _ in range(50):
        res = checker.check_relevance(ctx)
        assert res.is_relevant == first_res.is_relevant
        assert res.reason == first_res.reason
        assert res.relevant_chunks_count == first_res.relevant_chunks_count
        assert res.best_distance == first_res.best_distance
