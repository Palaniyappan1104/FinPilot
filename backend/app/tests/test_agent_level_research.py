"""Agent-level end-to-end and integration tests for Research Analyst (Phase 9.16).

Verifies the complete Research Vault / Research Analyst pipeline:
1. End-to-end RAG pipeline:
   Document creation -> chunking -> metadata -> embeddings -> ChromaDB storage ->
   query embedding -> similarity search -> Top-K retrieval -> context construction ->
   Research Analyst Agent -> EvidenceTracker resolution
2. Relevant evidence synthesis & grounding
3. Irrelevant evidence rejection & zero-LLM short circuiting (Phase 9.15)
4. Mixed relevance chunk filtering into derived context
5. Empty retrieval context handling
6. Provenance & evidence citation validation (Phase 9.13)
7. Safety / prohibited investment advice rejection
8. Provider failure handling & error isolation
9. BaseAgent.run() execution contract and AgentResult verification
10. Edge cases: Context budget exceeded, duplicate chunks, UTF-8 / multi-lingual symbols
"""

import json
import math
from typing import Any, List, Optional

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.agents.research import ResearchAnalystAgent
from app.agents.research_schema import (
    InvalidProvenanceError,
    ResearchAnalysisOutput,
    ResearchAnalystInput,
    ResearchValidationError,
)
from app.core.llm.base import LLMProvider, LLMResponse
from app.core.llm.exceptions import LLMError
from app.models.context import ContextChunk, RetrievalContext
from app.models.embeddings import (
    ChunkEmbeddingRequest,
    ChunkEmbeddingResult,
    EmbeddingModelInfo,
)
from app.models.relevance import RelevanceConfig
from app.models.retrieval import TopKConfig
from app.models.vector_store import VectorRecord
from app.providers.embedding import EmbeddingProvider
from app.services.context_builder import ContextBuilder
from app.services.similarity_search import SimilaritySearchService
from app.services.top_k_retriever import TopKRetriever
from app.storage.chroma_vector_store import ChromaVectorStore

# ===========================================================================
# DETERMINISTIC TEST FIXTURES & MOCKS
# ===========================================================================


class MockAgentLLMProvider(LLMProvider):
    """Deterministic offline Mock LLM provider for agent-level tests."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_agent_llm_provider",
    ) -> None:
        super().__init__()
        self.responses = list(responses or [])
        self.call_count = 0
        self.prompts_received: List[str] = []
        self.fail_with = fail_with
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
        self.prompts_received.append(prompt)

        if self.fail_with is not None:
            raise self.fail_with

        if not self.responses:
            raise RuntimeError("MockAgentLLMProvider: No response queued.")

        resp_idx = min(self.call_count - 1, len(self.responses) - 1)
        return LLMResponse(
            content=self.responses[resp_idx],
            provider=self._name,
            model="mock-deterministic-model",
        )


class Deterministic16dEmbeddingProvider(EmbeddingProvider):
    """Deterministic 16-dimensional embedding provider for offline pipeline testing."""

    # 4 research topics mapped to dimensions 0..3;
    # dimensions 4..15 for lexical residuals
    TOPICS = [
        ["cloud", "azure", "server", "datacenter", "infrastructure"],  # Topic 0
        ["revenue", "growth", "margin", "earnings", "operating"],  # Topic 1
        ["dividend", "shareholder", "repurchase", "capital", "yield"],  # Topic 2
        ["antitrust", "regulation", "compliance", "lawsuit", "fine"],  # Topic 3
    ]

    def __init__(self, dimensions: int = 16) -> None:
        self._dimensions = dimensions

    @property
    def provider_name(self) -> str:
        return "deterministic_test"

    @property
    def model_name(self) -> str:
        return "test-16d"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider=self.provider_name,
            model_name=self.model_name,
            dimensions=self.dimensions,
        )

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * 16
        tokens = text.lower().split()
        for token in tokens:
            matched = False
            for t_idx, kws in enumerate(self.TOPICS):
                if any(kw in token for kw in kws):
                    vec[t_idx] += 2.0
                    matched = True
                    break
            if not matched:
                bucket = 4 + (hash(token) % 12)
                vec[bucket] += 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            return [round(v / norm, 6) for v in vec]
        return [1.0] + [0.0] * 15

    def embed_query(self, query: str) -> List[float]:
        return self._embed_text(query)

    def embed_single(self, request: ChunkEmbeddingRequest) -> ChunkEmbeddingResult:
        vec = self._embed_text(request.text)
        return ChunkEmbeddingResult(
            chunk_id=request.chunk_id,
            document_id=request.document_id,
            ticker=request.ticker,
            document_type=request.document_type,
            page_numbers=request.page_numbers,
            start_page=request.start_page,
            end_page=request.end_page,
            chunk_index=request.chunk_index,
            text=request.text,
            embedding=vec,
            embedding_model=self.model_name,
        )

    def embed_batch(
        self, requests: List[ChunkEmbeddingRequest]
    ) -> List[ChunkEmbeddingResult]:
        return [self.embed_single(r) for r in requests]

    def embed_chunk(self, request: ChunkEmbeddingRequest) -> ChunkEmbeddingResult:
        return self.embed_single(request)

    def embed_chunks(
        self, requests: List[ChunkEmbeddingRequest]
    ) -> List[ChunkEmbeddingResult]:
        return self.embed_batch(requests)


def create_in_memory_chroma() -> chromadb.ClientAPI:
    """Create an ephemeral in-memory Chroma client."""
    return chromadb.Client(ChromaSettings(is_persistent=False, allow_reset=True))


def make_test_chunk(
    chunk_id: str,
    distance: float,
    text: str,
    doc_id: str = "doc_msft_2024",
    source: str = "MSFT_10K_2024.pdf",
    pages: Optional[List[int]] = None,
    ticker: str = "MSFT",
    section: str = "Item 7",
    chunk_index: int = 0,
) -> ContextChunk:
    """Helper to build a deterministic ContextChunk."""
    return ContextChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        source=source,
        ticker=ticker,
        doc_type="10-K",
        pages=pages or [24],
        section=section,
        distance=distance,
        text=text,
        chunk_index=chunk_index,
        metadata={"ticker": ticker, "source": source},
    )


def make_test_retrieval_context(
    chunks: List[ContextChunk],
    query: str = "What was Microsoft Cloud revenue growth?",
    collection_name: str = "vault_test",
    exceeds_budget: bool = False,
) -> RetrievalContext:
    """Helper to assemble a valid RetrievalContext."""
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
        exceeds_budget=exceeds_budget,
        built_at="2026-09-14T00:00:00Z",
        retrieval_metadata={"source": "test_agent_level_research"},
    )


# ===========================================================================
# 1. END-TO-END VAULT PIPELINE INTEGRATION
# ===========================================================================


class TestEndToEndVaultPipeline:
    """End-to-end integration: store -> query -> retrieve -> context -> agent answer."""

    def test_end_to_end_relevant_research_flow(self) -> None:
        """Verify full path from vector store storage through Research Analyst."""
        chroma_client = create_in_memory_chroma()
        collection_name = "vault_e2e_msft"
        vector_store = ChromaVectorStore(client=chroma_client)
        embed_provider = Deterministic16dEmbeddingProvider()

        # 1. Prepare raw document chunks and embeddings
        raw_chunks = [
            (
                "msft_chunk_001",
                (
                    "Microsoft Cloud revenue reached $137.4 billion, "
                    "growing 23% year-over-year."
                ),
                24,
                "Item 7",
            ),
            (
                "msft_chunk_002",
                (
                    "Operating income increased 24% to $109.4 billion "
                    "driven by cloud expansion."
                ),
                25,
                "Item 7",
            ),
        ]

        records: List[VectorRecord] = []
        for cid, txt, page, sect in raw_chunks:
            records.append(
                VectorRecord(
                    id=cid,
                    embedding=embed_provider.embed_query(txt),
                    document=txt,
                    metadata={
                        "chunk_id": cid,
                        "document_id": "doc_msft_2024",
                        "source": "MSFT_10K_2024.pdf",
                        "ticker": "MSFT",
                        "document_type": "10-K",
                        "page_numbers": str(page),
                        "section_name": sect,
                        "chunk_index": 0,
                    },
                )
            )
        vector_store.upsert_records(collection_name=collection_name, records=records)

        # 2. Query embedding and similarity search
        query = "What was Microsoft Cloud revenue and growth rate?"
        query_vector = embed_provider.embed_query(query)

        search_service = SimilaritySearchService(vector_store=vector_store)
        search_result = search_service.search(
            collection_name=collection_name,
            query=query_vector,
            n_results=2,
        )

        # 3. Top-K retrieval
        retriever = TopKRetriever()
        top_k_result = retriever.retrieve(
            search_results=search_result,
            config=TopKConfig(k=2, max_distance=0.65),
        )

        # 4. Context construction
        context_builder = ContextBuilder()
        context = context_builder.build(
            retrieval_result=top_k_result,
            query=query,
        )

        assert context.has_evidence is True
        assert len(context.chunks) == 2

        # 5. Research Analyst Agent Execution with Mock LLM
        valid_llm_payload = {
            "query": query,
            "answer": (
                "Microsoft Cloud revenue grew 23% year-over-year to $137.4 billion "
                "with operating income expanding 24%."
            ),
            "key_findings": [
                {
                    "claim": "Microsoft Cloud revenue grew 23% to $137.4 billion.",
                    "evidence": [
                        {
                            "chunk_id": "msft_chunk_001",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [24],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "msft_chunk_001",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [24],
                }
            ],
            "confidence": 0.95,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(valid_llm_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        agent_result = agent.run(ResearchAnalystInput(query=query, context=context))

        assert agent_result.success is True
        output: ResearchAnalysisOutput = agent_result.data
        assert output.insufficient_evidence is False
        assert output.confidence == 0.95
        assert len(output.key_findings) == 1
        assert output.evidence[0].chunk_id == "msft_chunk_001"
        assert mock_llm.call_count == 1

        # 6. Verify EvidenceTracker resolution
        tracker = agent.create_evidence_tracker(context)
        tracked_ev = tracker.resolve_by_chunk_id("msft_chunk_001")
        assert tracked_ev.document_id == "doc_msft_2024"
        assert tracked_ev.source_document == "MSFT_10K_2024.pdf"
        assert tracked_ev.page_numbers == [24]

    def test_end_to_end_irrelevant_query_pipeline_short_circuit(self) -> None:
        """Verify off-topic query through full pipeline short-circuits without LLM."""
        chroma_client = create_in_memory_chroma()
        vector_store = ChromaVectorStore(client=chroma_client)
        embed_provider = Deterministic16dEmbeddingProvider()
        collection_name = "vault_e2e_irrelevant"

        # Seed only cloud financial content
        txt = "Cloud infrastructure revenue increased by 20%."
        rec = VectorRecord(
            id="c_fin_01",
            embedding=embed_provider.embed_query(txt),
            document=txt,
            metadata={
                "chunk_id": "c_fin_01",
                "document_id": "doc_fin",
                "source": "Fin.pdf",
                "page_numbers": "1",
            },
        )
        vector_store.upsert_records(collection_name=collection_name, records=[rec])

        # Query about completely unrelated domain (weather / vacation)
        query = "What is the best snorkeling spot in Maui during winter?"
        query_vector = embed_provider.embed_query(query)

        search_service = SimilaritySearchService(vector_store=vector_store)
        search_result = search_service.search(
            collection_name=collection_name,
            query=query_vector,
            n_results=1,
        )

        # Retrieve top-k without pre-filtering distance so context
        # carries the distant chunk
        top_k_result = TopKRetriever().retrieve(
            search_results=search_result,
            config=TopKConfig(k=1, max_distance=None),
        )

        context = ContextBuilder().build(
            retrieval_result=top_k_result,
            query=query,
        )

        # Agent relevance gate must short-circuit
        mock_llm = MockAgentLLMProvider()
        agent = ResearchAnalystAgent(provider=mock_llm)

        res = agent.run(ResearchAnalystInput(query=query, context=context))

        assert res.success is True
        assert res.data.insufficient_evidence is True
        assert res.data.confidence == 0.0
        assert mock_llm.call_count == 0


# ===========================================================================
# 2. RELEVANT EVIDENCE SYNTHESIS & GROUNDING
# ===========================================================================


class TestRelevantEvidenceSynthesis:
    """Verifies that relevant RetrievalContext is synthesized and grounded."""

    def test_prompt_contains_exact_retrieved_evidence(self) -> None:
        """Verify formatted prompt provided to LLM contains the retrieved chunk text."""
        chunk = make_test_chunk(
            chunk_id="c_exact_01",
            distance=0.22,
            text="Operating margins widened 320 basis points to 44.5%.",
            source="NVDA_10K.pdf",
            pages=[18],
        )
        context = make_test_retrieval_context([chunk])

        llm_payload = {
            "query": "What happened to operating margins?",
            "answer": "Operating margins widened by 320 basis points to 44.5%.",
            "key_findings": [
                {
                    "claim": "Margins widened 320 bps to 44.5%.",
                    "evidence": [
                        {
                            "chunk_id": "c_exact_01",
                            "document_id": "doc_msft_2024",
                            "source_document": "NVDA_10K.pdf",
                            "page_numbers": [18],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "c_exact_01",
                    "document_id": "doc_msft_2024",
                    "source_document": "NVDA_10K.pdf",
                    "page_numbers": [18],
                }
            ],
            "confidence": 0.92,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(llm_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        output = agent.analyze(
            query="What happened to operating margins?",
            context=context,
        )

        assert mock_llm.call_count == 1
        prompt = mock_llm.prompts_received[0]
        assert "Operating margins widened 320 basis points" in prompt
        assert "c_exact_01" in prompt
        assert output.insufficient_evidence is False

    def test_multi_chunk_multi_finding_synthesis(self) -> None:
        """Verify research answer synthesizing evidence from multiple chunks."""
        c1 = make_test_chunk(
            "c1", 0.20, "Data Center revenue grew 112% to $22.6B.", pages=[12]
        )
        c2 = make_test_chunk(
            "c2", 0.35, "Free cash flow was $14.9B vs $5.3B prior year.", pages=[14]
        )
        context = make_test_retrieval_context([c1, c2])

        llm_payload = {
            "query": "Summarize data center growth and cash flow.",
            "answer": (
                "Data center revenue surged 112% and free cash flow reached $14.9B."
            ),
            "key_findings": [
                {
                    "claim": "Data Center revenue surged 112% to $22.6B.",
                    "evidence": [
                        {
                            "chunk_id": "c1",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [12],
                        }
                    ],
                },
                {
                    "claim": "Free cash flow reached $14.9B.",
                    "evidence": [
                        {
                            "chunk_id": "c2",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [14],
                        }
                    ],
                },
            ],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [12],
                },
                {
                    "chunk_id": "c2",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [14],
                },
            ],
            "confidence": 0.94,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(llm_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        output = agent.analyze(
            query="Summarize data center growth and cash flow.",
            context=context,
        )

        assert len(output.key_findings) == 2
        assert len(output.evidence) == 2


# ===========================================================================
# 3. IRRELEVANT EVIDENCE REJECTION (PHASE 9.15 INTEGRATION)
# ===========================================================================


class TestIrrelevantEvidenceRejection:
    """Verifies that low-relevance retrieval returns insufficient evidence."""

    def test_distant_chunks_rejected_without_llm(self) -> None:
        """Verify context where all chunks exceed max_distance does not trigger LLM."""
        c1 = make_test_chunk("c_weak1", 0.75, "Random unrelated disclosure.", pages=[5])
        c2 = make_test_chunk(
            "c_weak2", 0.88, "Boilerplate litigation disclosure.", pages=[6]
        )
        context = make_test_retrieval_context([c1, c2])

        mock_llm = MockAgentLLMProvider()
        agent = ResearchAnalystAgent(provider=mock_llm)

        output = agent.analyze(query="What is the dividend policy?", context=context)

        assert mock_llm.call_count == 0
        assert output.insufficient_evidence is True
        assert output.confidence == 0.0
        assert output.key_findings == []
        assert output.evidence == []
        assert "No retrieved chunks met the maximum distance threshold" in (
            output.insufficient_reason or ""
        )

    def test_custom_agent_relevance_threshold_applied(self) -> None:
        """Verify agent configured with custom max_distance enforces threshold."""
        c1 = make_test_chunk("c1", 0.52, "Moderate match chunk.")
        context = make_test_retrieval_context([c1])

        mock_llm = MockAgentLLMProvider()
        strict_agent = ResearchAnalystAgent(
            provider=mock_llm,
            relevance_config=RelevanceConfig(max_distance=0.45),
        )

        output = strict_agent.analyze(query="Some query", context=context)
        assert output.insufficient_evidence is True
        assert mock_llm.call_count == 0


# ===========================================================================
# 4. MIXED RELEVANCE & WEAK CHUNK FILTERING
# ===========================================================================


class TestMixedRelevanceFiltering:
    """Verifies weak chunks (>0.65) filtered while relevant chunks (<=0.65) proceed."""

    def test_weak_chunk_withheld_from_prompt(self) -> None:
        """Verify that only relevant chunks appear in LLM prompt and active context."""
        c_good = make_test_chunk(
            "c_good", 0.30, "Strongly relevant revenue data.", pages=[1]
        )
        c_bad = make_test_chunk("c_bad", 0.82, "Unrelated weak chunk noise.", pages=[2])
        context = make_test_retrieval_context([c_good, c_bad])

        llm_payload = {
            "query": "What is revenue?",
            "answer": "Revenue data is strong.",
            "key_findings": [
                {
                    "claim": "Revenue is strong.",
                    "evidence": [
                        {
                            "chunk_id": "c_good",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [1],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "c_good",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [1],
                }
            ],
            "confidence": 0.90,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(llm_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        output = agent.analyze(query="What is revenue?", context=context)

        assert mock_llm.call_count == 1
        prompt = mock_llm.prompts_received[0]
        assert "c_good" in prompt
        assert "c_bad" not in prompt
        assert "Unrelated weak chunk noise" not in prompt
        assert output.insufficient_evidence is False

    def test_citing_filtered_chunk_raises_provenance_error(self) -> None:
        """Verify that citing a filtered-out chunk raises InvalidProvenanceError."""
        c_good = make_test_chunk(
            "c_good", 0.30, "Strongly relevant revenue data.", pages=[1]
        )
        c_bad = make_test_chunk("c_bad", 0.82, "Unrelated weak chunk noise.", pages=[2])
        context = make_test_retrieval_context([c_good, c_bad])

        # Model attempts to cite c_bad (which was filtered out)
        hallucinated_payload = {
            "query": "What is revenue?",
            "answer": "Revenue is strong.",
            "key_findings": [
                {
                    "claim": "Revenue is strong.",
                    "evidence": [
                        {
                            "chunk_id": "c_bad",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [2],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "c_bad",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [2],
                }
            ],
            "confidence": 0.80,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(hallucinated_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        with pytest.raises(
            InvalidProvenanceError, match="does not exist in RetrievalContext"
        ):
            agent.analyze(query="What is revenue?", context=context)


# ===========================================================================
# 5. EMPTY RETRIEVAL & ZERO-EVIDENCE HANDLING
# ===========================================================================


class TestEmptyRetrievalHandling:
    """Verifies handling when zero chunks are retrieved."""

    def test_empty_retrieval_context_returns_insufficient_evidence(self) -> None:
        """Verify empty RetrievalContext returns insufficient evidence without LLM."""
        empty_context = make_test_retrieval_context(chunks=[])
        mock_llm = MockAgentLLMProvider()
        agent = ResearchAnalystAgent(provider=mock_llm)

        output = agent.analyze(query="Any question", context=empty_context)

        assert mock_llm.call_count == 0
        assert output.insufficient_evidence is True
        assert output.confidence == 0.0
        assert output.key_findings == []
        assert output.evidence == []
        assert "no evidence chunks" in (output.insufficient_reason or "")


# ===========================================================================
# 6. EVIDENCE PROVENANCE & GROUNDING VALIDATION
# ===========================================================================


class TestProvenanceAndGrounding:
    """Verifies strict deterministic provenance enforcement."""

    def test_invalid_chunk_id_citation_rejected(self) -> None:
        """Verify citation to non-existent chunk_id is rejected."""
        chunk = make_test_chunk("c1", 0.25, "Operating cash flow was $10B.", pages=[5])
        context = make_test_retrieval_context([chunk])

        bad_payload = {
            "query": "Cash flow?",
            "answer": "Cash flow was $10B.",
            "key_findings": [
                {
                    "claim": "Cash flow was $10B.",
                    "evidence": [
                        {
                            "chunk_id": "non_existent_chunk",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [5],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "non_existent_chunk",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [5],
                }
            ],
            "confidence": 0.85,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(bad_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        with pytest.raises(
            InvalidProvenanceError, match="does not exist in RetrievalContext"
        ):
            agent.analyze(query="Cash flow?", context=context)

    def test_mismatched_page_number_citation_rejected(self) -> None:
        """Verify citation with page number not matching chunk pages is rejected."""
        chunk = make_test_chunk("c1", 0.25, "Operating cash flow was $10B.", pages=[5])
        context = make_test_retrieval_context([chunk])

        bad_page_payload = {
            "query": "Cash flow?",
            "answer": "Cash flow was $10B.",
            "key_findings": [
                {
                    "claim": "Cash flow was $10B.",
                    "evidence": [
                        {
                            "chunk_id": "c1",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [
                                99
                            ],  # page 99 does not exist on chunk pages [5]
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [99],
                }
            ],
            "confidence": 0.85,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(bad_page_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        with pytest.raises(InvalidProvenanceError, match="not present in chunk pages"):
            agent.analyze(query="Cash flow?", context=context)

    def test_unsupported_finding_without_evidence_rejected(self) -> None:
        """Verify finding with empty evidence citation list is rejected."""
        chunk = make_test_chunk("c1", 0.25, "Operating cash flow was $10B.", pages=[5])
        context = make_test_retrieval_context([chunk])

        no_ev_payload = {
            "query": "Cash flow?",
            "answer": "Cash flow was $10B.",
            "key_findings": [
                {
                    "claim": "Unsubstantiated factual claim.",
                    "evidence": [],  # empty evidence list
                }
            ],
            "evidence": [],
            "confidence": 0.85,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(no_ev_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        with pytest.raises(ResearchValidationError, match="has no supporting evidence"):
            agent.analyze(query="Cash flow?", context=context)


# ===========================================================================
# 7. SAFETY & PROHIBITED INVESTMENT ADVICE CONSTRAINTS
# ===========================================================================


class TestSafetyAndDecisionSupportConstraints:
    """Verifies that investment recommendations / price targets are rejected."""

    @pytest.mark.parametrize(
        "forbidden_phrase",
        [
            "We issue a buy recommendation for MSFT.",
            "Our strong sell rating is based on debt.",
            "Investors must buy at current levels.",
            "Target price is $450 per share.",
            "Strong buy recommendation for the stock.",
        ],
    )
    def test_prohibited_phrases_in_answer_rejected(self, forbidden_phrase: str) -> None:
        """Verify prohibited investment advice in answer raises ValidationError."""
        chunk = make_test_chunk("c1", 0.20, "Operating cash flow was $10B.", pages=[5])
        context = make_test_retrieval_context([chunk])

        advice_payload = {
            "query": "Investment opinion?",
            "answer": forbidden_phrase,
            "key_findings": [
                {
                    "claim": "Operating cash flow was $10B.",
                    "evidence": [
                        {
                            "chunk_id": "c1",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [5],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [5],
                }
            ],
            "confidence": 0.90,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(advice_payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        with pytest.raises(
            ResearchValidationError, match="Prohibited recommendation phrase"
        ):
            agent.analyze(query="Investment opinion?", context=context)


# ===========================================================================
# 8. PROVIDER FAILURE HANDLING & ERROR ISOLATION
# ===========================================================================


class TestProviderFailureHandling:
    """Verifies graceful handling of upstream model and parser errors."""

    def test_llm_provider_failure_returns_failure_result(self) -> None:
        """Verify LLM network failure produces AgentResult failure without crash."""
        chunk = make_test_chunk("c1", 0.20, "Operating cash flow was $10B.")
        context = make_test_retrieval_context([chunk])

        failing_llm = MockAgentLLMProvider(
            fail_with=LLMError("Upstream connection timeout to inference provider.")
        )
        agent = ResearchAnalystAgent(provider=failing_llm)

        result = agent.run(ResearchAnalystInput(query="Cash flow?", context=context))

        assert result.success is False
        assert result.confidence == 0.0
        assert "Upstream connection timeout" in (result.error or "")

    def test_malformed_llm_json_returns_failure_result(self) -> None:
        """Verify unparseable model response produces clean failure result."""
        chunk = make_test_chunk("c1", 0.20, "Operating cash flow was $10B.")
        context = make_test_retrieval_context([chunk])

        malformed_llm = MockAgentLLMProvider(
            responses=["This is not JSON text at all."]
        )
        agent = ResearchAnalystAgent(provider=malformed_llm)

        result = agent.run(ResearchAnalystInput(query="Cash flow?", context=context))

        assert result.success is False
        assert result.confidence == 0.0
        assert result.error is not None


# ===========================================================================
# 9. AGENT EXECUTION WRAPPER CONTRACT (BaseAgent.run())
# ===========================================================================


class TestAgentExecutionWrapper:
    """Verifies public agent execution contract."""

    def test_run_with_valid_dict_payload(self) -> None:
        """Verify run() accepts raw dictionary and converts to ResearchAnalystInput."""
        chunk = make_test_chunk("c1", 0.30, "Cloud revenue grew 20%.", pages=[10])
        context = make_test_retrieval_context([chunk])

        payload = {
            "query": "Cloud growth?",
            "answer": "Cloud revenue grew 20%.",
            "key_findings": [
                {
                    "claim": "Cloud revenue grew 20%.",
                    "evidence": [
                        {
                            "chunk_id": "c1",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [10],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [10],
                }
            ],
            "confidence": 0.90,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        dict_input = {
            "query": "Cloud growth?",
            "context": context,
        }
        result = agent.run(dict_input)

        assert result.success is True
        assert isinstance(result.data, ResearchAnalysisOutput)
        assert result.data.answer == "Cloud revenue grew 20%."

    def test_run_with_invalid_input_type(self) -> None:
        """Verify run() fails when input is neither ResearchAnalystInput nor dict."""
        mock_llm = MockAgentLLMProvider()
        agent = ResearchAnalystAgent(provider=mock_llm)

        result = agent.run("invalid string payload")  # type: ignore[arg-type]

        assert result.success is False
        assert "must be an instance of ResearchAnalystInput or dict" in (
            result.error or ""
        )

    def test_per_query_relevance_config_override_in_run(self) -> None:
        """Verify relevance_config in ResearchAnalystInput overrides agent defaults."""
        chunk = make_test_chunk("c1", 0.55, "Some financial text.")
        context = make_test_retrieval_context([chunk])

        mock_llm = MockAgentLLMProvider()
        agent = ResearchAnalystAgent(provider=mock_llm)

        # Stricter override: max_distance=0.50
        input_data = ResearchAnalystInput(
            query="Financial text?",
            context=context,
            relevance_config=RelevanceConfig(max_distance=0.50),
        )
        result = agent.run(input_data)

        assert result.success is True
        assert result.data.insufficient_evidence is True
        assert mock_llm.call_count == 0


# ===========================================================================
# 10. EDGE CASES (PLAN.MD 9.16.2)
# ===========================================================================


class TestVaultEdgeCases:
    """Verifies edge cases: budget, duplicate chunks, and UTF-8 multilingual text."""

    def test_context_budget_exceeded_fails_safely(self) -> None:
        """Verify context exceeding character budget raises ResearchValidationError."""
        chunk = make_test_chunk("c1", 0.20, "Standard length chunk text.")
        context = make_test_retrieval_context([chunk], exceeds_budget=True)

        mock_llm = MockAgentLLMProvider()
        agent = ResearchAnalystAgent(provider=mock_llm)

        with pytest.raises(
            ResearchValidationError, match="exceeds configured character budget"
        ):
            agent.analyze(query="Some query", context=context)

    def test_duplicate_chunks_handled_deterministically(self) -> None:
        """Verify context containing duplicate chunk IDs is handled without crash."""
        c1 = make_test_chunk("c1", 0.25, "Segment operating margin was 35%.", pages=[8])
        c1_dup = make_test_chunk(
            "c1", 0.25, "Segment operating margin was 35%.", pages=[8]
        )
        context = make_test_retrieval_context([c1, c1_dup])

        payload = {
            "query": "Segment margin?",
            "answer": "Segment operating margin was 35%.",
            "key_findings": [
                {
                    "claim": "Segment operating margin was 35%.",
                    "evidence": [
                        {
                            "chunk_id": "c1",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [8],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "c1",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [8],
                }
            ],
            "confidence": 0.90,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        output = agent.analyze(query="Segment margin?", context=context)
        assert output.insufficient_evidence is False

        # Evidence tracker deduplication check
        tracker = agent.create_evidence_tracker(context)
        assert tracker.total_evidence == 1

    def test_utf8_multilingual_and_currency_symbols(self) -> None:
        """Verify chunks with European & Asian currency symbols (€, ¥) work properly."""
        unicode_text = (
            "Europäische Erlöse stiegen auf €12,5 Mrd. und ¥350 Mrd. in Asien."
        )
        chunk = make_test_chunk("c_utf8", 0.20, unicode_text, pages=[15])
        context = make_test_retrieval_context([chunk])

        payload = {
            "query": "European and Asian revenue?",
            "answer": "European revenue rose to €12,5 Mrd. and ¥350 Mrd. in Asia.",
            "key_findings": [
                {
                    "claim": "European revenue was €12,5 Mrd.",
                    "evidence": [
                        {
                            "chunk_id": "c_utf8",
                            "document_id": "doc_msft_2024",
                            "source_document": "MSFT_10K_2024.pdf",
                            "page_numbers": [15],
                        }
                    ],
                }
            ],
            "evidence": [
                {
                    "chunk_id": "c_utf8",
                    "document_id": "doc_msft_2024",
                    "source_document": "MSFT_10K_2024.pdf",
                    "page_numbers": [15],
                }
            ],
            "confidence": 0.92,
            "insufficient_evidence": False,
            "insufficient_reason": None,
        }
        mock_llm = MockAgentLLMProvider(responses=[json.dumps(payload)])
        agent = ResearchAnalystAgent(provider=mock_llm)

        output = agent.analyze(query="European and Asian revenue?", context=context)

        assert output.insufficient_evidence is False
        assert "€12,5 Mrd." in output.answer
        assert "¥350 Mrd." in output.answer
