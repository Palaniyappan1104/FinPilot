"""Unit tests for Phase 9.13: Source/Evidence Tracking.

All tests run 100% offline and deterministically without live APIs or network calls.

Covers all 25+ required test scenarios:
1. Create evidence from valid RetrievalContext
2. Deterministic evidence ID format
3. Same chunk produces same evidence ID across calls
4. Different chunks produce different evidence IDs
5. Multiple documents support
6. Multiple chunks per document
7. Resolve by evidence ID
8. Resolve by chunk ID
9. Missing evidence resolution failure (EvidenceNotFoundError)
10. Missing chunk resolution failure (EvidenceNotFoundError)
11. Missing document_id on chunk fails validation (EvidenceValidationError)
12. Missing source_document on chunk fails validation (EvidenceValidationError)
13. Invalid page metadata (non-positive / strings) rejected
14. Mismatched document ID citation rejected (ConflictingProvenanceError)
15. Conflicting provenance for same chunk ID rejected (ConflictingProvenanceError)
16. Duplicate chunk handling and deduplication preserves unique records
17. Preservation of retrieval ranking and ordering
18. Preservation of source document
19. Preservation of page numbers
20. Preservation of section
21. Preservation of retrieval distance
22. Fabricated provenance citation rejection (FabricatedProvenanceError)
23. Malformed evidence rejection (empty strings, non-finite distance)
24. Research Analyst integration and finding traceability
25. Empty RetrievalContext handling
26. Raw document text is not leaked into evidence metadata
27. Safe operational logging
28. Flexible resolve method (string ID vs ResearchEvidenceRef)
"""

import logging
from typing import List, Optional

import pytest
from pydantic import ValidationError

from app.agents.research import ResearchAnalystAgent
from app.agents.research_schema import (
    ResearchAnalysisOutput,
    ResearchEvidenceRef,
    ResearchFinding,
)
from app.models.context import ContextChunk, RetrievalContext
from app.models.evidence import (
    ConflictingProvenanceError,
    EvidenceNotFoundError,
    EvidenceValidationError,
    FabricatedProvenanceError,
    ResearchEvidence,
    compute_evidence_id,
)
from app.services.evidence_tracker import EvidenceTracker, track_evidence

# ===========================================================================
# TEST FIXTURES & FACTORY HELPERS
# ===========================================================================


def make_context_chunk(
    chunk_id: str,
    document_id: str = "doc_apple_10k",
    text: str = "Operating margin reached 31.2% in fiscal 2024.",
    source: str = "AAPL_2024_10K.pdf",
    ticker: Optional[str] = "AAPL",
    doc_type: Optional[str] = "10-K",
    pages: Optional[List[int]] = None,
    section: Optional[str] = "Item 7: MD&A",
    distance: float = 0.12,
    chunk_index: int = 1,
) -> ContextChunk:
    """Helper to build a deterministic ContextChunk."""
    return ContextChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source=source,
        ticker=ticker,
        document_type=doc_type,
        pages=pages if pages is not None else [21, 22],
        section=section,
        distance=distance,
        text=text,
        chunk_index=chunk_index,
        metadata={
            "source_document": source,
            "ticker": ticker,
            "document_type": doc_type,
            "internal_score": 0.95,
        },
    )


def make_retrieval_context(
    chunks: List[ContextChunk],
    query: str = "What was operating margin?",
    collection_name: str = "financial_docs_test",
) -> RetrievalContext:
    """Helper to build a RetrievalContext from ContextChunks."""
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
        has_evidence=len(chunks) > 0,
    )


# ===========================================================================
# 1. CREATE EVIDENCE FROM VALID RETRIEVAL CONTEXT
# ===========================================================================


def test_create_evidence_from_valid_context():
    """Scenario 1: Create evidence items from valid RetrievalContext."""
    chunk = make_context_chunk(chunk_id="chunk_1")
    context = make_retrieval_context([chunk])

    tracker = EvidenceTracker(context=context)

    assert tracker.total_evidence == 1
    assert tracker.total_documents == 1
    assert not tracker.is_empty
    evidence = tracker.resolve_by_chunk_id("chunk_1")
    assert evidence.chunk_id == "chunk_1"
    assert evidence.document_id == "doc_apple_10k"
    assert evidence.source_document == "AAPL_2024_10K.pdf"


# ===========================================================================
# 2, 3, 4. DETERMINISTIC EVIDENCE ID STRATEGY
# ===========================================================================


def test_deterministic_evidence_id_format():
    """Scenario 2: Verify deterministic evidence ID format."""
    eid = compute_evidence_id("doc_123", "chunk_456")
    assert eid == "ev_doc_123_chunk_456"


def test_same_chunk_produces_same_evidence_id():
    """Scenario 3: Same chunk produces identical evidence ID across repeated runs."""
    eid1 = compute_evidence_id("doc_alpha", "c_001")
    eid2 = compute_evidence_id("doc_alpha", "c_001")
    assert eid1 == eid2 == "ev_doc_alpha_c_001"


def test_different_chunks_produce_different_evidence_ids():
    """Scenario 4: Different chunks produce distinct evidence IDs."""
    eid1 = compute_evidence_id("doc_alpha", "c_001")
    eid2 = compute_evidence_id("doc_alpha", "c_002")
    eid3 = compute_evidence_id("doc_beta", "c_001")
    assert eid1 != eid2
    assert eid1 != eid3
    assert eid2 != eid3


def test_compute_evidence_id_validation():
    """Verify compute_evidence_id rejects empty/whitespace arguments."""
    with pytest.raises(EvidenceValidationError):
        compute_evidence_id("", "chunk_1")
    with pytest.raises(EvidenceValidationError):
        compute_evidence_id("doc_1", "   ")


# ===========================================================================
# 5, 6. MULTI-DOCUMENT AND MULTI-CHUNK SUPPORT
# ===========================================================================


def test_multiple_documents_and_chunks():
    """Scenarios 5 & 6: Support multiple documents and multiple chunks per document."""
    c_a1 = make_context_chunk(chunk_id="a_1", document_id="doc_A", source="DocA.pdf")
    c_a2 = make_context_chunk(chunk_id="a_2", document_id="doc_A", source="DocA.pdf")
    c_b1 = make_context_chunk(chunk_id="b_1", document_id="doc_B", source="DocB.pdf")
    c_b2 = make_context_chunk(chunk_id="b_2", document_id="doc_B", source="DocB.pdf")

    context = make_retrieval_context([c_a1, c_a2, c_b1, c_b2])
    tracker = track_evidence(context)

    assert tracker.total_evidence == 4
    assert tracker.total_documents == 2
    assert set(tracker.document_ids) == {"doc_A", "doc_B"}

    doc_a_evidence = tracker.get_document_evidence("doc_A")
    doc_b_evidence = tracker.get_document_evidence("doc_B")
    assert len(doc_a_evidence) == 2
    assert len(doc_b_evidence) == 2
    assert [e.chunk_id for e in doc_a_evidence] == ["a_1", "a_2"]
    assert [e.chunk_id for e in doc_b_evidence] == ["b_1", "b_2"]


# ===========================================================================
# 7, 8, 9, 10. RESOLUTION BY EVIDENCE_ID AND CHUNK_ID
# ===========================================================================


def test_resolve_by_evidence_id():
    """Scenario 7: Resolve evidence by deterministic evidence_id."""
    chunk = make_context_chunk(chunk_id="c_target", document_id="doc_target")
    context = make_retrieval_context([chunk])
    tracker = EvidenceTracker(context)

    expected_eid = "ev_doc_target_c_target"
    assert tracker.has_evidence_id(expected_eid)
    evidence = tracker.resolve_by_evidence_id(expected_eid)
    assert evidence.chunk_id == "c_target"
    assert evidence.document_id == "doc_target"


def test_resolve_by_chunk_id():
    """Scenario 8: Resolve evidence by chunk_id."""
    chunk = make_context_chunk(chunk_id="c_target", document_id="doc_target")
    context = make_retrieval_context([chunk])
    tracker = EvidenceTracker(context)

    assert tracker.has_chunk("c_target")
    evidence = tracker.resolve_by_chunk_id("c_target")
    assert evidence.evidence_id == "ev_doc_target_c_target"


def test_resolve_missing_evidence_id_raises_not_found():
    """Scenario 9: Resolving non-existent evidence_id raises EvidenceNotFoundError."""
    chunk = make_context_chunk(chunk_id="c1")
    tracker = EvidenceTracker(make_retrieval_context([chunk]))

    with pytest.raises(EvidenceNotFoundError) as exc_info:
        tracker.resolve_by_evidence_id("ev_nonexistent_chunk")
    assert "ev_nonexistent_chunk" in str(exc_info.value)


def test_resolve_missing_chunk_id_raises_not_found():
    """Scenario 10: Resolving non-existent chunk_id raises EvidenceNotFoundError."""
    chunk = make_context_chunk(chunk_id="c1")
    tracker = EvidenceTracker(make_retrieval_context([chunk]))

    with pytest.raises(EvidenceNotFoundError) as exc_info:
        tracker.resolve_by_chunk_id("chunk_nonexistent")
    assert "chunk_nonexistent" in str(exc_info.value)


# ===========================================================================
# 11, 12, 13. PROVENANCE VALIDATION
# ===========================================================================


def test_missing_document_id_fails_validation():
    """Scenario 11: Chunk with missing document_id raises EvidenceValidationError."""
    chunk = make_context_chunk(chunk_id="c1", document_id="")
    context = make_retrieval_context([chunk])
    tracker = EvidenceTracker()

    with pytest.raises(EvidenceValidationError) as exc_info:
        tracker.register_context(context)
    assert "mandatory document_id" in str(exc_info.value)


def test_missing_source_document_fails_validation():
    """Scenario 12: Chunk with missing source_document raises
    EvidenceValidationError.
    """
    # Chunk with source=None and empty metadata
    chunk = ContextChunk(
        chunk_id="c1",
        document_id="doc1",
        source=None,
        distance=0.1,
        text="Sample text content.",
        metadata={},
    )
    context = make_retrieval_context([chunk])
    tracker = EvidenceTracker()

    with pytest.raises(EvidenceValidationError) as exc_info:
        tracker.register_context(context)
    assert "mandatory source_document" in str(exc_info.value)


def test_invalid_page_metadata_rejected():
    """Scenario 13: Non-positive or invalid page numbers rejected."""
    with pytest.raises(ValidationError):
        ResearchEvidence(
            evidence_id="ev_d1_c1",
            chunk_id="c1",
            document_id="d1",
            source_document="doc.pdf",
            page_numbers=[0],  # 0 is invalid (must be >= 1)
            distance=0.1,
        )

    with pytest.raises(ValidationError):
        ResearchEvidence(
            evidence_id="ev_d1_c1",
            chunk_id="c1",
            document_id="d1",
            source_document="doc.pdf",
            page_numbers=[-3],  # negative is invalid
            distance=0.1,
        )


# ===========================================================================
# 14, 15. MISMATCHED / CONFLICTING PROVENANCE
# ===========================================================================


def test_mismatched_document_id_citation_rejected():
    """Scenario 14: Citation with wrong document_id raises
    ConflictingProvenanceError.
    """
    chunk = make_context_chunk(chunk_id="c1", document_id="doc_real")
    tracker = EvidenceTracker(make_retrieval_context([chunk]))

    citation = ResearchEvidenceRef(
        chunk_id="c1",
        document_id="doc_WRONG",
        source_document="AAPL_2024_10K.pdf",
        page_numbers=[21],
    )

    with pytest.raises(ConflictingProvenanceError) as exc_info:
        tracker.verify_citation(citation)
    assert "does not match registered document_id" in str(exc_info.value)


def test_conflicting_provenance_for_same_chunk_rejected():
    """Scenario 15: Same chunk registered with conflicting document raises error."""
    chunk1 = make_context_chunk(chunk_id="c1", document_id="doc_A", source="A.pdf")
    chunk2 = make_context_chunk(chunk_id="c1", document_id="doc_B", source="B.pdf")

    context = make_retrieval_context([chunk1, chunk2])
    tracker = EvidenceTracker()

    with pytest.raises(ConflictingProvenanceError) as exc_info:
        tracker.register_context(context)
    assert "Conflicting provenance detected" in str(exc_info.value)


# ===========================================================================
# 16, 17. DEDUPLICATION AND ORDER PRESERVATION
# ===========================================================================


def test_duplicate_chunk_deduplication():
    """Scenario 16: Duplicate chunk with identical provenance is deduplicated."""
    chunk1 = make_context_chunk(chunk_id="c1", document_id="doc_A", distance=0.1)
    chunk2 = make_context_chunk(chunk_id="c1", document_id="doc_A", distance=0.1)

    context = make_retrieval_context([chunk1, chunk2])
    tracker = EvidenceTracker(context)

    assert tracker.total_evidence == 1
    assert len(tracker.get_all_evidence()) == 1


def test_preservation_of_retrieval_ranking():
    """Scenario 17: Preserves Top-K retrieval ordering."""
    c1 = make_context_chunk(chunk_id="rank_1", distance=0.05)
    c2 = make_context_chunk(chunk_id="rank_2", distance=0.15)
    c3 = make_context_chunk(chunk_id="rank_3", distance=0.25)

    context = make_retrieval_context([c1, c2, c3])
    tracker = EvidenceTracker(context)

    evidence_list = tracker.get_all_evidence()
    assert [e.chunk_id for e in evidence_list] == ["rank_1", "rank_2", "rank_3"]
    assert [e.distance for e in evidence_list] == [0.05, 0.15, 0.25]


# ===========================================================================
# 18, 19, 20, 21. PRESERVATION OF METADATA ATTRIBUTES
# ===========================================================================


def test_provenance_preservation():
    """Scenarios 18, 19, 20, 21: Preserves source document, pages, section, distance."""
    chunk = make_context_chunk(
        chunk_id="chunk_detail",
        document_id="doc_msft_q3",
        source="MSFT_FY24Q3.pdf",
        ticker="MSFT",
        doc_type="10-Q",
        pages=[42, 43],
        section="Liquidity and Capital Resources",
        distance=0.082,
        chunk_index=7,
    )
    context = make_retrieval_context([chunk])
    tracker = EvidenceTracker(context)

    evidence = tracker.resolve_by_chunk_id("chunk_detail")
    # 18. Source document
    assert evidence.source_document == "MSFT_FY24Q3.pdf"
    # 19. Page numbers
    assert evidence.page_numbers == [42, 43]
    # 20. Section
    assert evidence.section == "Liquidity and Capital Resources"
    # 21. Retrieval distance
    assert evidence.distance == 0.082
    assert evidence.ticker == "MSFT"
    assert evidence.document_type == "10-Q"
    assert evidence.chunk_index == 7

    # Test to_provenance_dict()
    pdict = evidence.to_provenance_dict()
    assert pdict["evidence_id"] == "ev_doc_msft_q3_chunk_detail"
    assert pdict["chunk_id"] == "chunk_detail"
    assert pdict["page_numbers"] == [42, 43]


# ===========================================================================
# 22. FABRICATED PROVENANCE REJECTION
# ===========================================================================


def test_fabricated_provenance_rejection():
    """Scenario 22: Cited chunk_id not present in registry is rejected."""
    chunk = make_context_chunk(chunk_id="chunk_real")
    tracker = EvidenceTracker(make_retrieval_context([chunk]))

    fake_citation = ResearchEvidenceRef(
        chunk_id="chunk_FABRICATED_999",
        document_id="doc_apple_10k",
        source_document="AAPL_2024_10K.pdf",
        page_numbers=[21],
    )

    with pytest.raises(FabricatedProvenanceError) as exc_info:
        tracker.verify_citation(fake_citation)
    assert "chunk_FABRICATED_999" in str(exc_info.value)


# ===========================================================================
# 23. MALFORMED EVIDENCE REJECTION
# ===========================================================================


def test_malformed_evidence_rejected():
    """Scenario 23: Malformed fields (empty IDs, non-finite distance) raise
    ValidationError.
    """
    with pytest.raises(ValidationError):
        ResearchEvidence(
            evidence_id="",  # Empty
            chunk_id="c1",
            document_id="d1",
            source_document="doc.pdf",
            distance=0.1,
        )

    with pytest.raises(ValidationError):
        ResearchEvidence(
            evidence_id="ev_1",
            chunk_id="   ",  # Whitespace only
            document_id="d1",
            source_document="doc.pdf",
            distance=0.1,
        )

    with pytest.raises(ValidationError):
        ResearchEvidence(
            evidence_id="ev_1",
            chunk_id="c1",
            document_id="d1",
            source_document="doc.pdf",
            distance=float("nan"),  # NaN not allowed
        )

    with pytest.raises(ValidationError):
        ResearchEvidence(
            evidence_id="ev_1",
            chunk_id="c1",
            document_id="d1",
            source_document="doc.pdf",
            distance=float("inf"),  # Infinity not allowed
        )


# ===========================================================================
# 24. RESEARCH ANALYST COMPATIBILITY
# ===========================================================================


def test_research_analyst_compatibility():
    """Scenario 24: Finding -> ResearchEvidenceRef -> EvidenceTracker ->
    ResearchEvidence.
    """
    chunk = make_context_chunk(
        chunk_id="chunk_fin_1",
        document_id="doc_nvda_2024",
        source="NVDA_10K.pdf",
        pages=[15, 16],
    )
    context = make_retrieval_context([chunk])

    agent = ResearchAnalystAgent()
    tracker = agent.create_evidence_tracker(context)

    # Simulate structured output from ResearchAnalystAgent
    finding = ResearchFinding(
        claim="Data center revenue grew 217% year-over-year.",
        evidence=[
            ResearchEvidenceRef(
                chunk_id="chunk_fin_1",
                document_id="doc_nvda_2024",
                source_document="NVDA_10K.pdf",
                page_numbers=[15],
            )
        ],
    )
    output = ResearchAnalysisOutput(
        query="What drove NVDA revenue growth?",
        answer="NVDA revenue growth was primarily driven by data center demand.",
        key_findings=[finding],
        evidence=finding.evidence,
        confidence=0.92,
        insufficient_evidence=False,
    )

    # Trace finding -> evidence ref -> EvidenceTracker -> exact source provenance
    verified_evidence = tracker.verify_analysis_output(output)
    assert len(verified_evidence) == 1
    assert verified_evidence[0].evidence_id == "ev_doc_nvda_2024_chunk_fin_1"
    assert verified_evidence[0].source_document == "NVDA_10K.pdf"
    assert verified_evidence[0].page_numbers == [15, 16]


# ===========================================================================
# 25. EMPTY RETRIEVAL CONTEXT
# ===========================================================================


def test_empty_retrieval_context():
    """Scenario 25: Empty RetrievalContext creates an empty tracker safely."""
    context = make_retrieval_context([])
    tracker = EvidenceTracker(context)

    assert tracker.is_empty
    assert tracker.total_evidence == 0
    assert tracker.total_documents == 0
    assert tracker.get_all_evidence() == []
    assert tracker.get_document_evidence("any_doc") == []


# ===========================================================================
# 26. NO RAW TEXT LEAKAGE INTO METADATA
# ===========================================================================


def test_raw_text_not_duplicated_into_evidence_metadata():
    """Scenario 26: Full raw text is omitted from ResearchEvidence metadata."""
    chunk = make_context_chunk(
        chunk_id="c1",
        text=(
            "A very long secret document passage that should not be duplicated "
            "in metadata."
        ),
    )
    # inject text into raw metadata dictionary
    chunk.metadata["raw_text"] = "should be filtered"
    chunk.metadata["text"] = "should be filtered"

    context = make_retrieval_context([chunk])
    tracker = EvidenceTracker(context)

    evidence = tracker.resolve_by_chunk_id("c1")
    assert "raw_text" not in evidence.metadata
    assert "text" not in evidence.metadata
    assert "document_text" not in evidence.metadata


# ===========================================================================
# 27. SAFE OPERATIONAL LOGGING
# ===========================================================================


def test_safe_operational_logging(caplog):
    """Scenario 27: Registration logs only counts and IDs without leaking
    credentials.
    """
    chunk = make_context_chunk(chunk_id="c_log_test", document_id="doc_log")
    context = make_retrieval_context([chunk])

    with caplog.at_level(logging.INFO):
        EvidenceTracker(context)

    log_text = caplog.text
    assert "Evidence registration complete: 1 total evidence items" in log_text
    # Confirm no API keys or passwords in log
    assert "api_key" not in log_text.lower()
    assert "secret" not in log_text.lower()


# ===========================================================================
# 28. FLEXIBLE RESOLVE METHOD
# ===========================================================================


def test_flexible_resolve_method():
    """Scenario 28: resolve() handles evidence_id, chunk_id, or ResearchEvidenceRef."""
    chunk = make_context_chunk(chunk_id="chunk_flex", document_id="doc_flex")
    tracker = EvidenceTracker(make_retrieval_context([chunk]))

    # Resolve by evidence_id string
    e1 = tracker.resolve("ev_doc_flex_chunk_flex")
    assert e1.chunk_id == "chunk_flex"

    # Resolve by chunk_id string
    e2 = tracker.resolve("chunk_flex")
    assert e2.evidence_id == "ev_doc_flex_chunk_flex"

    # Resolve by ResearchEvidenceRef
    ref = ResearchEvidenceRef(
        chunk_id="chunk_flex",
        document_id="doc_flex",
        source_document="AAPL_2024_10K.pdf",
        page_numbers=[21],
    )
    e3 = tracker.resolve(ref)
    assert e3.chunk_id == "chunk_flex"

    # Unsupported type raises EvidenceValidationError
    with pytest.raises(EvidenceValidationError):
        tracker.resolve(12345)  # type: ignore[arg-type]
