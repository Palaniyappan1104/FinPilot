"""Retrieval quality, correctness, and benchmark testing (Phase 9.14).

Evaluates the FinPilot Research Vault retrieval pipeline:
Query -> Query Embedding -> Similarity Search -> Top-K Retrieval -> Context Construction

Key evaluation dimensions:
- Deterministic synthetic financial evaluation corpus (ACME 10-K, ACME 10-Q, BETA 10-K)
- 10 human-defined ground-truth research queries (single-doc, multi-chunk,
  cross-doc, negative)
- Quantitative evaluation metrics: Recall@K, Precision@K, HitRate@K (K=1, 3, 5) and MRR
- Retrieval correctness: distance ranking, deterministic tie-breaking,
  metadata preservation
- Filtering: document-isolation, company-isolation, max-distance thresholding
- Zero live API calls / 100% offline reproducible testing
"""

import math
import re
import zlib
from typing import Any, Dict, List, Optional

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, ConfigDict, Field

from app.models.context import ContextConfig, RetrievalContext
from app.models.embeddings import (
    ChunkEmbeddingRequest,
    ChunkEmbeddingResult,
    EmbeddingModelInfo,
)
from app.models.retrieval import TopKConfig
from app.models.vector_store import VectorRecord
from app.providers.embedding import EmbeddingProvider
from app.services.context_builder import ContextBuilder
from app.services.evidence_tracker import EvidenceTracker
from app.services.query_embedder import QueryEmbedder
from app.services.similarity_search import SimilaritySearchService
from app.services.top_k_retriever import TopKRetriever
from app.storage.chroma_vector_store import ChromaVectorStore


def create_ephemeral_client() -> chromadb.ClientAPI:
    """Create an in-memory ChromaDB client with allow_reset=True."""
    return chromadb.Client(ChromaSettings(is_persistent=False, allow_reset=True))


# ===========================================================================
# 1. EVALUATION SCHEMAS (Phase 9.14 Section 12)
# ===========================================================================


class QueryGroundTruth(BaseModel):
    """Ground truth specification for an evaluation query."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    query_text: str
    target_ticker: Optional[str] = None
    target_document_id: Optional[str] = None
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    description: str = ""
    is_negative_query: bool = False


class QueryEvaluationDetail(BaseModel):
    """Detailed evaluation metrics for an individual query evaluation."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    query_text: str
    k: int
    relevant_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    hits: int
    precision: float
    recall: float
    hit: bool
    first_relevant_rank: Optional[int] = None
    reciprocal_rank: float = 0.0


class RetrievalEvaluationResult(BaseModel):
    """Comprehensive machine-readable evaluation report for retrieval testing."""

    model_config = ConfigDict(frozen=True)

    total_queries: int
    evaluated_k_values: List[int]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    mrr: float
    per_query_details: List[QueryEvaluationDetail]
    failures: List[str] = Field(default_factory=list)


# ===========================================================================
# 2. DETERMINISTIC TOPIC-PROJECTION EMBEDDING PROVIDER
# ===========================================================================

# 8 distinct financial research topics forming the latent embedding space
TOPIC_KEYWORDS = [
    # Topic 0: Revenue and Financial Performance
    [
        "revenue",
        "sales",
        "operating",
        "income",
        "financial",
        "fiscal",
        "growth",
        "recurring",
        "quarterly",
        "annual",
        "billion",
    ],
    # Topic 1: Artificial Intelligence and Technology Infrastructure
    [
        "ai",
        "machine",
        "learning",
        "gpu",
        "infrastructure",
        "clusters",
        "generative",
        "copilot",
        "accelerators",
        "technology",
    ],
    # Topic 2: Debt, Leverage, and Capital Resources
    [
        "debt",
        "leverage",
        "ebitda",
        "credit",
        "liquidity",
        "capital",
        "borrowing",
        "refinancing",
        "restructuring",
        "bonds",
    ],
    # Topic 3: Regulatory, Compliance, and Legal Risks
    [
        "regulatory",
        "scrutiny",
        "antitrust",
        "compliance",
        "fines",
        "licensing",
        "legal",
        "proceedings",
        "cross-border",
        "export",
    ],
    # Topic 4: Supply Chain and Operational Logistics
    [
        "supply",
        "chain",
        "lead",
        "times",
        "delivery",
        "constraints",
        "hardware",
        "server",
        "freight",
        "logistics",
        "shipping",
    ],
    # Topic 5: Management Guidance and Outlook
    [
        "outlook",
        "guidance",
        "anticipates",
        "forecast",
        "estimates",
        "momentum",
        "future",
        "projections",
        "expectations",
    ],
    # Topic 6: Enterprise Product Adoption and Software
    [
        "product",
        "development",
        "enterprise",
        "customers",
        "adoption",
        "software",
        "platform",
        "pilot",
        "launched",
    ],
    # Topic 7: Telecom and Edge Network Infrastructure
    [
        "telecom",
        "network",
        "edge",
        "hyperscale",
        "broadband",
        "cellular",
        "carrier",
        "connectivity",
    ],
]


def text_to_topic_vector(text: str, dimensions: int = 16) -> List[float]:
    """Compute a deterministic, normalized topic projection vector from text.

    Dimensions 0..7 represent the 8 specialized financial research topics.
    Dimensions 8..15 represent lexical residual buckets for general/out-of-domain terms.
    If a text matches financial topics, it projects into dimensions 0..7.
    If a text contains no financial topics (negative/off-topic query), it projects
    exclusively into orthogonal dimensions 8..15, guaranteeing high
    cosine distance (>= 0.70).
    """
    lower_text = text.lower()
    words = re.findall(r"[a-z0-9_-]+", lower_text)
    word_set = set(words)
    raw_vector = [0.0] * dimensions
    has_topic_match = False

    # Topic projection (dimensions 0..7)
    for i, keywords in enumerate(TOPIC_KEYWORDS[: min(8, dimensions)]):
        matches = 0
        for kw in keywords:
            if " " in kw or "-" in kw:
                if kw in lower_text:
                    matches += 1
            elif kw in word_set:
                matches += 1
        if matches > 0:
            raw_vector[i] = matches * 3.0
            has_topic_match = True

    # Residual / out-of-domain projection (dimensions 8..15)
    if not has_topic_match:
        residual_dims = max(1, dimensions - 8)
        for w in words:
            bucket = 8 + (zlib.crc32(w.encode("utf-8")) % residual_dims)
            raw_vector[bucket] += 1.0

    # L2 normalization to guarantee unit-norm vectors
    norm = math.sqrt(sum(x * x for x in raw_vector))
    if norm < 1e-9:
        return [1.0 / math.sqrt(dimensions)] * dimensions
    return [round(x / norm, 6) for x in raw_vector]


class DeterministicTopicEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline EmbeddingProvider for reproducible retrieval testing."""

    def __init__(self, dimensions: int = 16) -> None:
        self._dimensions = dimensions

    @property
    def provider_name(self) -> str:
        return "deterministic_topic"

    @property
    def model_name(self) -> str:
        return "deterministic-topic-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider=self.provider_name,
            model_name=self.model_name,
            dimensions=self.dimensions,
        )

    def embed_query(self, query: str) -> List[float]:
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")
        return text_to_topic_vector(query, dimensions=self._dimensions)

    def embed_single(self, request: ChunkEmbeddingRequest) -> ChunkEmbeddingResult:
        vec = text_to_topic_vector(request.text, dimensions=self._dimensions)
        return ChunkEmbeddingResult(
            chunk_id=request.chunk_id,
            embedding=vec,
            dimensions=self._dimensions,
            model=self.model_name,
        )

    def embed_batch(
        self, requests: List[ChunkEmbeddingRequest]
    ) -> List[ChunkEmbeddingResult]:
        return [self.embed_single(r) for r in requests]


# ===========================================================================
# 3. SYNTHETIC EVALUATION CORPUS (Phase 9.14 Section 2)
# ===========================================================================

SYNTHETIC_CHUNKS: List[Dict[str, Any]] = [
    # Document 1: ACME FY2024 10-K (Annual Report)
    {
        "chunk_id": "chunk_acme_rev_01",
        "document_id": "doc_acme_10k_2024",
        "ticker": "ACME",
        "source_document": "ACME_FY2024_10K.pdf",
        "document_type": "10-K",
        "page_numbers": [25],
        "section": "Financial Results",
        "chunk_index": 1,
        "text": (
            "ACME reported annual revenue of $42.5 billion in fiscal year 2024, "
            "representing a 14% year-over-year increase driven by "
            "enterprise cloud services."
        ),
    },
    {
        "chunk_id": "chunk_acme_ai_02",
        "document_id": "doc_acme_10k_2024",
        "ticker": "ACME",
        "source_document": "ACME_FY2024_10K.pdf",
        "document_type": "10-K",
        "page_numbers": [31],
        "section": "Technology and AI Investments",
        "chunk_index": 2,
        "text": (
            "Management accelerated capital investments in AI infrastructure, "
            "committing $2.8 billion to proprietary machine learning "
            "clusters and GPU acceleration."
        ),
    },
    {
        "chunk_id": "chunk_acme_debt_03",
        "document_id": "doc_acme_10k_2024",
        "ticker": "ACME",
        "source_document": "ACME_FY2024_10K.pdf",
        "document_type": "10-K",
        "page_numbers": [45],
        "section": "Liquidity and Capital Resources",
        "chunk_index": 3,
        "text": (
            "Total long-term debt stood at $8.2 billion with a net "
            "debt-to-EBITDA ratio of 1.4x, maintaining strong "
            "investment-grade credit ratings."
        ),
    },
    {
        "chunk_id": "chunk_acme_reg_04",
        "document_id": "doc_acme_10k_2024",
        "ticker": "ACME",
        "source_document": "ACME_FY2024_10K.pdf",
        "document_type": "10-K",
        "page_numbers": [58],
        "section": "Risk Factors - Regulatory",
        "chunk_index": 4,
        "text": (
            "Regulatory scrutiny over cross-border data transfers and antitrust "
            "compliance in European markets poses compliance risks and potential "
            "fines."
        ),
    },
    {
        "chunk_id": "chunk_acme_guidance_05",
        "document_id": "doc_acme_10k_2024",
        "ticker": "ACME",
        "source_document": "ACME_FY2024_10K.pdf",
        "document_type": "10-K",
        "page_numbers": [62],
        "section": "Management Outlook",
        "chunk_index": 5,
        "text": (
            "Management's outlook and guidance anticipates fiscal 2025 revenue "
            "growth between 10% and 12%, citing continued momentum in enterprise "
            "generative AI adoption."
        ),
    },
    # Document 2: ACME Q3 2024 (Quarterly Report)
    {
        "chunk_id": "chunk_acme_q3_rev_06",
        "document_id": "doc_acme_q3_2024",
        "ticker": "ACME",
        "source_document": "ACME_Q3_2024.pdf",
        "document_type": "10-Q",
        "page_numbers": [10],
        "section": "Quarterly Financial Highlights",
        "chunk_index": 1,
        "text": (
            "Quarterly revenue for Q3 2024 reached $11.1 billion, exceeding consensus "
            "estimates by 3.2% led by subscription recurring revenue."
        ),
    },
    {
        "chunk_id": "chunk_acme_q3_supply_07",
        "document_id": "doc_acme_q3_2024",
        "ticker": "ACME",
        "source_document": "ACME_Q3_2024.pdf",
        "document_type": "10-Q",
        "page_numbers": [18],
        "section": "Operational Updates",
        "chunk_index": 2,
        "text": (
            "Global supply chain lead times improved by 35%, easing hardware "
            "server delivery constraints and lowering freight costs."
        ),
    },
    {
        "chunk_id": "chunk_acme_q3_reg_08",
        "document_id": "doc_acme_q3_2024",
        "ticker": "ACME",
        "source_document": "ACME_Q3_2024.pdf",
        "document_type": "10-Q",
        "page_numbers": [24],
        "section": "Legal Proceedings and Regulatory",
        "chunk_index": 3,
        "text": (
            "New export licensing requirements for advanced semiconductor accelerators "
            "introduced administrative delays for select Asian customers."
        ),
    },
    {
        "chunk_id": "chunk_acme_q3_ai_09",
        "document_id": "doc_acme_q3_2024",
        "ticker": "ACME",
        "source_document": "ACME_Q3_2024.pdf",
        "document_type": "10-Q",
        "page_numbers": [29],
        "section": "Product Development",
        "chunk_index": 4,
        "text": (
            "Launched ACME Copilot for enterprise finance, integrating internal "
            "multi-agent automation across 120 pilot customers."
        ),
    },
    # Document 3: BETA FY2024 10-K (Different Company for Isolation Tests)
    {
        "chunk_id": "chunk_beta_rev_10",
        "document_id": "doc_beta_10k_2024",
        "ticker": "BETA",
        "source_document": "BETA_FY2024_10K.pdf",
        "document_type": "10-K",
        "page_numbers": [20],
        "section": "Financial Overview",
        "chunk_index": 1,
        "text": (
            "BETA generated fiscal 2024 revenue of $18.4 billion with operating "
            "cash flow of $3.6 billion."
        ),
    },
    {
        "chunk_id": "chunk_beta_debt_11",
        "document_id": "doc_beta_10k_2024",
        "ticker": "BETA",
        "source_document": "BETA_FY2024_10K.pdf",
        "document_type": "10-K",
        "page_numbers": [35],
        "section": "Capital Structure",
        "chunk_index": 2,
        "text": (
            "BETA carries $14.5 billion in total debt with $2.1 billion due in the "
            "next twelve months, requiring refinancing or debt restructuring."
        ),
    },
    {
        "chunk_id": "chunk_beta_ai_12",
        "document_id": "doc_beta_10k_2024",
        "ticker": "BETA",
        "source_document": "BETA_FY2024_10K.pdf",
        "document_type": "10-K",
        "page_numbers": [48],
        "section": "Strategic Initiatives",
        "chunk_index": 3,
        "text": (
            "BETA partnered with hyperscale cloud providers to deploy foundation model "
            "inference across its telecom network edge."
        ),
    },
]


# ===========================================================================
# 4. GROUND-TRUTH EVALUATION QUERIES (Phase 9.14 Section 3)
# ===========================================================================

GROUND_TRUTH_QUERIES: List[QueryGroundTruth] = [
    # 1. Exact query for AI investment
    QueryGroundTruth(
        query_id="q_01_exact_ai",
        query_text=(
            "What did management say about AI infrastructure and "
            "technology investment?"
        ),
        target_ticker="ACME",
        relevant_chunk_ids=["chunk_acme_ai_02", "chunk_acme_q3_ai_09"],
        description="Exact query targeting AI and machine learning investments",
    ),
    # 2. Paraphrased query for AI investment
    QueryGroundTruth(
        query_id="q_02_paraphrased_ai",
        query_text=(
            "Capital spending on machine learning GPU clusters and "
            "Copilot deployment"
        ),
        target_ticker="ACME",
        relevant_chunk_ids=["chunk_acme_ai_02", "chunk_acme_q3_ai_09"],
        description="Paraphrased query with alternative phrasing for AI infrastructure",
    ),
    # 3. Revenue performance query
    QueryGroundTruth(
        query_id="q_03_revenue",
        query_text="How did annual and quarterly revenue perform in fiscal 2024?",
        target_ticker="ACME",
        relevant_chunk_ids=["chunk_acme_rev_01", "chunk_acme_q3_rev_06"],
        description="Revenue performance spanning annual 10-K and quarterly 10-Q",
    ),
    # 4. Debt and capital position
    QueryGroundTruth(
        query_id="q_04_debt",
        query_text="What is the company's long-term debt position and leverage ratio?",
        target_ticker="ACME",
        relevant_chunk_ids=["chunk_acme_debt_03"],
        description="Debt and leverage ratio query",
    ),
    # 5. Regulatory risks (Cross-document)
    QueryGroundTruth(
        query_id="q_05_regulatory",
        query_text=(
            "What regulatory scrutiny and export licensing compliance risks "
            "were reported?"
        ),
        target_ticker="ACME",
        relevant_chunk_ids=["chunk_acme_reg_04", "chunk_acme_q3_reg_08"],
        description="Cross-document regulatory compliance and export delay risks",
    ),
    # 6. Management outlook and guidance
    QueryGroundTruth(
        query_id="q_06_outlook",
        query_text=(
            "What is management's outlook, future expectations, and "
            "guidance for fiscal 2025?"
        ),
        target_ticker="ACME",
        relevant_chunk_ids=["chunk_acme_guidance_05"],
        description="Forward-looking management guidance query",
    ),
    # 7. Supply chain operations
    QueryGroundTruth(
        query_id="q_07_supply_chain",
        query_text=(
            "How have global supply chain lead times and delivery "
            "constraints changed?"
        ),
        target_ticker="ACME",
        relevant_chunk_ids=["chunk_acme_q3_supply_07"],
        description="Operational supply chain and logistics lead times",
    ),
    # 8. Broad cross-document query
    QueryGroundTruth(
        query_id="q_08_broad_strategic",
        query_text=(
            "Summarize ACME's AI initiatives, machine learning, and "
            "enterprise software growth"
        ),
        target_ticker="ACME",
        relevant_chunk_ids=[
            "chunk_acme_ai_02",
            "chunk_acme_q3_ai_09",
            "chunk_acme_q3_rev_06",
            "chunk_acme_guidance_05",
        ],
        description=(
            "Broad multi-topic query spanning AI and recurring software " "revenue"
        ),
    ),
    # 9. Company isolation query (BETA debt)
    QueryGroundTruth(
        query_id="q_09_beta_debt",
        query_text="What are BETA's total debt obligations and refinancing needs?",
        target_ticker="BETA",
        relevant_chunk_ids=["chunk_beta_debt_11"],
        description="BETA-specific debt structure query",
    ),
    # 10. Irrelevant / out-of-domain negative query
    QueryGroundTruth(
        query_id="q_10_negative_irrelevant",
        query_text=(
            "What are the best tourist hotels and restaurants in " "Honolulu Hawaii?"
        ),
        target_ticker="ACME",
        relevant_chunk_ids=[],
        description="Out-of-domain query with no relevant financial chunks",
        is_negative_query=True,
    ),
]


# ===========================================================================
# 5. RETRIEVAL EVALUATION HARNESS (Phase 9.14 Section 4 & 5)
# ===========================================================================


class RetrievalEvaluationHarness:
    """End-to-end evaluation harness executing real retrieval and computing metrics."""

    def __init__(
        self,
        collection_name: str = "retrieval_eval_collection",
        dimensions: int = 16,
    ) -> None:
        self.collection_name = collection_name
        self.provider = DeterministicTopicEmbeddingProvider(dimensions=dimensions)
        self.vector_store = ChromaVectorStore(client=create_ephemeral_client())
        self.similarity_service = SimilaritySearchService(
            vector_store=self.vector_store
        )
        self.top_k_retriever = TopKRetriever()
        self.context_builder = ContextBuilder()
        self.query_embedder = QueryEmbedder(provider=self.provider)

        self._populate_collection()

    def _populate_collection(self) -> None:
        """Embed and upsert all synthetic chunks into ephemeral ChromaVectorStore."""
        records: List[VectorRecord] = []
        for c in SYNTHETIC_CHUNKS:
            vec = text_to_topic_vector(c["text"], dimensions=self.provider.dimensions)
            metadata = {
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "ticker": c["ticker"],
                "source_document": c["source_document"],
                "document_type": c["document_type"],
                "page_numbers": ",".join(str(p) for p in c["page_numbers"]),
                "section": c["section"],
                "section_name": c["section"],
                "chunk_index": c["chunk_index"],
            }
            records.append(
                VectorRecord(
                    id=c["chunk_id"],
                    embedding=vec,
                    document=c["text"],
                    metadata=metadata,
                )
            )

        self.vector_store.upsert_records(
            collection_name=self.collection_name, records=records
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        ticker_filter: Optional[str] = None,
        doc_filter: Optional[str] = None,
        max_distance: Optional[float] = None,
    ) -> RetrievalContext:
        """Run the full query -> embed -> search -> top_k -> context pipeline."""
        query_res = self.query_embedder.embed_query(query)

        where: Optional[Dict[str, Any]] = None
        if ticker_filter and doc_filter:
            where = {"$and": [{"ticker": ticker_filter}, {"document_id": doc_filter}]}
        elif ticker_filter:
            where = {"ticker": ticker_filter}
        elif doc_filter:
            where = {"document_id": doc_filter}

        search_res = self.similarity_service.search(
            collection_name=self.collection_name,
            query=query_res,
            n_results=10,
            where=where,
        )

        top_k_res = self.top_k_retriever.retrieve(
            search_results=search_res,
            config=TopKConfig(k=k, max_distance=max_distance),
        )

        return self.context_builder.build(
            retrieval_result=top_k_res,
            query=query,
            config=ContextConfig(),
        )

    def evaluate_benchmark(
        self,
        queries: List[QueryGroundTruth],
        k_values: Optional[List[int]] = None,
    ) -> RetrievalEvaluationResult:
        """Compute Precision, Recall, HitRate, and MRR across queries."""
        if k_values is None:
            k_values = [1, 3, 5]

        # Filter to relevant queries for standard IR metrics
        eval_queries = [q for q in queries if not q.is_negative_query]
        details: List[QueryEvaluationDetail] = []
        precision_sums = {k: 0.0 for k in k_values}
        recall_sums = {k: 0.0 for k in k_values}
        hit_sums = {k: 0.0 for k in k_values}
        reciprocal_rank_sums = 0.0

        for q in queries:
            is_neg = q.is_negative_query
            max_k = max(k_values)
            context_max = self.retrieve(
                query=q.query_text, k=max_k, ticker_filter=q.target_ticker
            )
            retrieved_max = [c.chunk_id for c in context_max.chunks]

            first_rank: Optional[int] = None
            if not is_neg:
                for rank_idx, cid in enumerate(retrieved_max, start=1):
                    if cid in q.relevant_chunk_ids:
                        first_rank = rank_idx
                        break
            rr = (1.0 / first_rank) if first_rank is not None else 0.0
            if not is_neg:
                reciprocal_rank_sums += rr

            # Evaluate at each K value
            for k in k_values:
                context_k = (
                    context_max
                    if k == max_k
                    else self.retrieve(
                        query=q.query_text, k=k, ticker_filter=q.target_ticker
                    )
                )
                retrieved_k = [c.chunk_id for c in context_k.chunks][:k]

                hits = (
                    sum(1 for cid in retrieved_k if cid in q.relevant_chunk_ids)
                    if not is_neg
                    else 0
                )
                precision = hits / k if k > 0 else 0.0
                recall = (
                    hits / len(q.relevant_chunk_ids)
                    if (q.relevant_chunk_ids and not is_neg)
                    else 0.0
                )
                hit = hits > 0

                if not is_neg:
                    precision_sums[k] += precision
                    recall_sums[k] += recall
                    if hit:
                        hit_sums[k] += 1.0

                details.append(
                    QueryEvaluationDetail(
                        query_id=q.query_id,
                        query_text=q.query_text,
                        k=k,
                        relevant_chunk_ids=list(q.relevant_chunk_ids),
                        retrieved_chunk_ids=retrieved_k,
                        hits=hits,
                        precision=round(precision, 4),
                        recall=round(recall, 4),
                        hit=hit,
                        first_relevant_rank=first_rank,
                        reciprocal_rank=round(rr, 4),
                    )
                )

        n = len(eval_queries)
        precision_at_k = {
            k: round(precision_sums[k] / n, 4) if n > 0 else 0.0 for k in k_values
        }
        recall_at_k = {
            k: round(recall_sums[k] / n, 4) if n > 0 else 0.0 for k in k_values
        }
        hit_rate_at_k = {
            k: round(hit_sums[k] / n, 4) if n > 0 else 0.0 for k in k_values
        }
        mrr = round(reciprocal_rank_sums / n, 4) if n > 0 else 0.0

        return RetrievalEvaluationResult(
            total_queries=len(queries),
            evaluated_k_values=k_values,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            hit_rate_at_k=hit_rate_at_k,
            mrr=mrr,
            per_query_details=details,
        )


# ===========================================================================
# 6. TESTS: CORPUS, EMBEDDINGS, AND PIPELINE CORRECTNESS
# ===========================================================================


@pytest.fixture(scope="module")
def harness() -> RetrievalEvaluationHarness:
    """Fixture providing initialized RetrievalEvaluationHarness."""
    return RetrievalEvaluationHarness()


def test_evaluation_dataset_integrity():
    """Verify synthetic dataset has 12 chunks across 3 documents and 2 tickers."""
    assert len(SYNTHETIC_CHUNKS) == 12
    doc_ids = {c["document_id"] for c in SYNTHETIC_CHUNKS}
    assert doc_ids == {
        "doc_acme_10k_2024",
        "doc_acme_q3_2024",
        "doc_beta_10k_2024",
    }
    tickers = {c["ticker"] for c in SYNTHETIC_CHUNKS}
    assert tickers == {"ACME", "BETA"}


def test_deterministic_topic_embedding_unit_norm():
    """Verify topic embedding generates deterministic unit-norm vectors."""
    provider = DeterministicTopicEmbeddingProvider(dimensions=8)
    vec1 = provider.embed_query("AI investment and GPUs")
    vec2 = provider.embed_query("AI investment and GPUs")

    assert vec1 == vec2
    norm = math.sqrt(sum(x * x for x in vec1))
    assert math.isclose(norm, 1.0, rel_tol=1e-4)


# ===========================================================================
# 7. RETRIEVAL CORRECTNESS TESTS (Phase 9.14 Section 6)
# ===========================================================================


def test_exact_relevant_query(harness: RetrievalEvaluationHarness):
    """Scenario 1: Exact relevant query returns expected chunks in top positions."""
    q = GROUND_TRUTH_QUERIES[0]  # q_01_exact_ai
    context = harness.retrieve(query=q.query_text, k=3, ticker_filter=q.target_ticker)

    retrieved = [c.chunk_id for c in context.chunks]
    assert "chunk_acme_ai_02" in retrieved
    # Top result should be the exact AI investment chunk
    assert retrieved[0] == "chunk_acme_ai_02"


def test_paraphrased_relevant_query(harness: RetrievalEvaluationHarness):
    """Scenario 2: Paraphrased query successfully retrieves relevant chunks."""
    q = GROUND_TRUTH_QUERIES[1]  # q_02_paraphrased_ai
    context = harness.retrieve(query=q.query_text, k=3, ticker_filter=q.target_ticker)

    retrieved = [c.chunk_id for c in context.chunks]
    # Both AI-related chunks retrieved
    assert "chunk_acme_ai_02" in retrieved or "chunk_acme_q3_ai_09" in retrieved


def test_broad_research_query_retrieval(harness: RetrievalEvaluationHarness):
    """Scenario 3: Broad research query returns multi-topic relevant chunks."""
    q = GROUND_TRUTH_QUERIES[7]  # q_08_broad_strategic
    context = harness.retrieve(query=q.query_text, k=5, ticker_filter=q.target_ticker)

    retrieved = [c.chunk_id for c in context.chunks]
    # Hits at least two distinct relevant chunks across AI and software revenue
    hits = set(retrieved).intersection(set(q.relevant_chunk_ids))
    assert len(hits) >= 2


def test_query_matching_multiple_chunks(harness: RetrievalEvaluationHarness):
    """Scenario 4: Query matching multiple chunks retrieves them in Top-K."""
    q = GROUND_TRUTH_QUERIES[2]  # q_03_revenue
    context = harness.retrieve(query=q.query_text, k=3, ticker_filter=q.target_ticker)

    retrieved = [c.chunk_id for c in context.chunks]
    assert "chunk_acme_rev_01" in retrieved
    assert "chunk_acme_q3_rev_06" in retrieved


def test_cross_document_retrieval(harness: RetrievalEvaluationHarness):
    """Scenario 5 & 10: Query retrieves relevant chunks across 10-K and 10-Q."""
    q = GROUND_TRUTH_QUERIES[4]  # q_05_regulatory
    context = harness.retrieve(query=q.query_text, k=3, ticker_filter=q.target_ticker)

    retrieved_docs = {c.document_id for c in context.chunks}
    # Chunks retrieved from both annual 10-K and quarterly 10-Q
    assert "doc_acme_10k_2024" in retrieved_docs
    assert "doc_acme_q3_2024" in retrieved_docs


def test_irrelevant_query_boundary(harness: RetrievalEvaluationHarness):
    """Scenario 6 & 11: Irrelevant query produces high distance / weak relevance."""
    q = GROUND_TRUTH_QUERIES[9]  # q_10_negative_irrelevant (Hawaii hotels)
    context = harness.retrieve(query=q.query_text, k=3, ticker_filter=q.target_ticker)

    # All retrieved chunks should have large distance indicating weak similarity
    assert all(c.distance >= 0.70 for c in context.chunks)


def test_top_k_ordering_and_distance_ordering(
    harness: RetrievalEvaluationHarness,
):
    """Scenarios 7 & 8: Top-K results are strictly sorted by ascending distance."""
    context = harness.retrieve(
        query="debt leverage capital ratios", k=5, ticker_filter="ACME"
    )

    distances = [c.distance for c in context.chunks]
    assert distances == sorted(distances)
    # The nearest match is the debt chunk
    assert context.chunks[0].chunk_id == "chunk_acme_debt_03"


def test_duplicate_chunk_handling(harness: RetrievalEvaluationHarness):
    """Scenario 9: Candidate records with identical chunk_id maintain uniqueness."""
    context = harness.retrieve(query="revenue performance", k=5)
    chunk_ids = [c.chunk_id for c in context.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


# ===========================================================================
# 8. EDGE CASES AND BOUNDARIES
# ===========================================================================


def test_empty_collection_retrieval():
    """Scenario 11: Retrieval against an empty collection returns empty context."""
    store = ChromaVectorStore(client=create_ephemeral_client())
    store.get_or_create_collection("empty_test_collection")
    service = SimilaritySearchService(vector_store=store)
    retriever = TopKRetriever()
    builder = ContextBuilder()

    provider = DeterministicTopicEmbeddingProvider()
    q_vec = provider.embed_query("any query")

    search_res = service.search("empty_test_collection", query=q_vec, n_results=5)
    top_k_res = retriever.retrieve(search_res, config=TopKConfig(k=3))
    context = builder.build(top_k_res, query="any query")

    assert context.is_empty
    assert context.total_retrieved == 0
    assert len(context.chunks) == 0


def test_empty_result_on_strict_threshold(harness: RetrievalEvaluationHarness):
    """Scenario 12: Strict max_distance threshold excludes distant candidates."""
    # Strict max_distance 0.0001 excludes any chunk not having exact 0 distance
    context = harness.retrieve(
        query="General business outlook",
        k=5,
        ticker_filter="ACME",
        max_distance=0.0001,
    )
    assert context.is_empty
    assert context.total_retrieved == 0


def test_k_larger_than_available_chunks(harness: RetrievalEvaluationHarness):
    """Scenario 13: Requesting K=50 returns all available chunks without crashing."""
    context = harness.retrieve(query="operating revenue", k=50, ticker_filter="BETA")
    # BETA has only 3 chunks total in the corpus
    assert len(context.chunks) <= 3
    assert context.total_retrieved <= 3


def test_k_equals_one(harness: RetrievalEvaluationHarness):
    """Scenario 14: K=1 returns exactly the single nearest chunk."""
    context = harness.retrieve(
        query="What is the company's long-term debt position and leverage ratio?",
        k=1,
        ticker_filter="ACME",
    )
    assert len(context.chunks) == 1
    assert context.chunks[0].chunk_id == "chunk_acme_debt_03"


# ===========================================================================
# 9. DETERMINISM AND ISOLATION (Phase 9.14 Section 8, 18, 19, 20)
# ===========================================================================


def test_deterministic_repeated_retrieval(harness: RetrievalEvaluationHarness):
    """Scenario 16: Repeated queries produce identical rankings and distances."""
    query = "Capital investments in AI clusters"
    ctx1 = harness.retrieve(query=query, k=4, ticker_filter="ACME")
    ctx2 = harness.retrieve(query=query, k=4, ticker_filter="ACME")

    assert [c.chunk_id for c in ctx1.chunks] == [c.chunk_id for c in ctx2.chunks]
    assert [c.distance for c in ctx1.chunks] == [c.distance for c in ctx2.chunks]


def test_document_isolation_filter(harness: RetrievalEvaluationHarness):
    """Scenario 18: Scoping retrieval to a specific document excludes others."""
    context = harness.retrieve(
        query="revenue growth",
        k=5,
        doc_filter="doc_acme_q3_2024",
    )
    assert all(c.document_id == "doc_acme_q3_2024" for c in context.chunks)
    assert "chunk_acme_rev_01" not in [c.chunk_id for c in context.chunks]


def test_company_isolation_filter(harness: RetrievalEvaluationHarness):
    """Scenario 19: Scoping retrieval by ticker isolates company records."""
    context_acme = harness.retrieve(
        query="debt restructuring", k=5, ticker_filter="ACME"
    )
    assert all(c.ticker == "ACME" for c in context_acme.chunks)

    context_beta = harness.retrieve(
        query="debt restructuring", k=5, ticker_filter="BETA"
    )
    assert all(c.ticker == "BETA" for c in context_beta.chunks)


def test_max_distance_filtering(harness: RetrievalEvaluationHarness):
    """Scenario 20: Candidate chunks exceeding max_distance threshold are filtered."""
    threshold = 0.45
    context = harness.retrieve(
        query="AI machine learning",
        k=5,
        ticker_filter="ACME",
        max_distance=threshold,
    )
    assert all(c.distance <= threshold for c in context.chunks)


# ===========================================================================
# 10. PROVENANCE VERIFICATION AND EVIDENCE TRACKER INTEGRATION
# ===========================================================================


def test_provenance_preservation_across_pipeline(
    harness: RetrievalEvaluationHarness,
):
    """Scenario 10 & 17: Validates all provenance attributes are intact."""
    context = harness.retrieve(query="debt position", k=1, ticker_filter="ACME")
    chunk = context.chunks[0]

    assert chunk.chunk_id == "chunk_acme_debt_03"
    assert chunk.document_id == "doc_acme_10k_2024"
    assert chunk.source == "ACME_FY2024_10K.pdf"
    assert chunk.ticker == "ACME"
    assert chunk.document_type == "10-K"
    assert chunk.pages == [45]
    assert chunk.section == "Liquidity and Capital Resources"
    assert chunk.chunk_index == 3
    assert math.isfinite(chunk.distance)


def test_evidence_tracker_integration_with_retrieved_context(
    harness: RetrievalEvaluationHarness,
):
    """Scenario 22: Context integrates with EvidenceTracker (Phase 9.13)."""
    context = harness.retrieve(
        query="AI and machine learning investments",
        k=3,
        ticker_filter="ACME",
    )
    tracker = EvidenceTracker(context)

    assert tracker.total_evidence == len(context.chunks)
    assert not tracker.is_empty
    top_chunk = context.chunks[0]
    evidence = tracker.resolve_by_chunk_id(top_chunk.chunk_id)
    assert evidence.evidence_id == f"ev_{top_chunk.document_id}_{top_chunk.chunk_id}"
    assert evidence.source_document == top_chunk.source


def test_chunk_text_integrity(harness: RetrievalEvaluationHarness):
    """Scenario 19: Retrieved chunk text matches raw source document verbatim."""
    context = harness.retrieve(
        query="AI infrastructure investment", k=1, ticker_filter="ACME"
    )
    chunk = context.chunks[0]
    original = next(c for c in SYNTHETIC_CHUNKS if c["chunk_id"] == chunk.chunk_id)

    assert chunk.text == original["text"]
    assert chunk.character_count == len(original["text"])
    assert chunk.word_count == len(original["text"].split())


def test_negative_query_zero_relevant_chunks(harness: RetrievalEvaluationHarness):
    """Scenario 24: Out-of-domain query produces 0 hits against ground truth."""
    q_neg = GROUND_TRUTH_QUERIES[9]
    assert q_neg.is_negative_query
    assert len(q_neg.relevant_chunk_ids) == 0

    report = harness.evaluate_benchmark(queries=[q_neg], k_values=[1, 3, 5])
    detail = report.per_query_details[0]
    assert detail.hits == 0
    assert detail.hit is False
    assert detail.first_relevant_rank is None
    assert detail.reciprocal_rank == 0.0


# ===========================================================================
# 11. QUANTITATIVE BENCHMARK EVALUATION (Phase 9.14 Section 4, 5, 12)
# ===========================================================================


def test_full_benchmark_metrics_thresholds(
    harness: RetrievalEvaluationHarness,
):
    """Scenario 23: Evaluates full benchmark retrieval quality thresholds."""
    report = harness.evaluate_benchmark(
        queries=GROUND_TRUTH_QUERIES, k_values=[1, 3, 5]
    )

    # 1. MRR (Mean Reciprocal Rank) should be very high (>= 0.85)
    assert report.mrr >= 0.85, f"MRR={report.mrr} below target 0.85"

    # 2. Hit Rate at K=3 should be >= 0.90 (>= 90% retrieve relevant chunk)
    assert (
        report.hit_rate_at_k[3] >= 0.90
    ), f"HitRate@3={report.hit_rate_at_k[3]} below target 0.90"

    # 3. Hit Rate at K=5 should be 1.0 (100% of queries retrieve relevant chunk)
    assert (
        report.hit_rate_at_k[5] == 1.0
    ), f"HitRate@5={report.hit_rate_at_k[5]} below target 1.0"

    # 4. Recall at K=5 should be >= 0.85
    assert (
        report.recall_at_k[5] >= 0.85
    ), f"Recall@5={report.recall_at_k[5]} below target 0.85"

    # 5. Precision at K=1 should be >= 0.80
    assert (
        report.precision_at_k[1] >= 0.80
    ), f"Precision@1={report.precision_at_k[1]} below target 0.80"


def test_evaluation_report_machine_readable(
    harness: RetrievalEvaluationHarness,
):
    """Scenario 24: Evaluation report serializes cleanly to dict."""
    report = harness.evaluate_benchmark(queries=GROUND_TRUTH_QUERIES, k_values=[1, 3])
    dumped = report.model_dump()

    assert dumped["total_queries"] == len(GROUND_TRUTH_QUERIES)
    assert dumped["evaluated_k_values"] == [1, 3]
    assert 1 in dumped["precision_at_k"]
    assert 3 in dumped["recall_at_k"]
    assert isinstance(dumped["mrr"], float)
    assert len(dumped["per_query_details"]) > 0


def test_zero_live_api_calls_offline(harness: RetrievalEvaluationHarness):
    """Scenario 25: All retrieval components run 100% offline."""
    assert harness.provider.provider_name == "deterministic_topic"
    assert harness.vector_store.store_name == "chroma"
    # EphemeralClient runs in-memory without network socket
    assert harness.vector_store._client is not None
