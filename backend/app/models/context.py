"""Domain models for Context Construction (Phase 9.11).

Defines:
- ContextChunk: Grounded chunk preserving exact text and provenance.
- ContextConfig: Configuration for context construction limits and rules.
- RetrievalContext: Structured container holding ordered context chunks,
  query reference, and retrieval metadata for the Research Analyst.
"""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContextChunk(BaseModel):
    """Grounded retrieved document chunk within structured context (Phase 9.11).

    Preserves exact unedited chunk text and complete citation metadata so downstream
    agents (Phase 9.12 Research Analyst) can trace and ground every insight.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique deterministic chunk identifier")
    document_id: Optional[str] = Field(
        default=None,
        description="Source document identifier",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source filename or document reference",
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Stock ticker or company identifier",
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Category of document (e.g. 10-K, 10-Q, 8-K)",
    )
    pages: List[int] = Field(
        default_factory=list,
        description="Source physical page numbers associated with chunk",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section heading or structural location in source document",
    )
    distance: float = Field(
        description="Raw distance score from vector search (lower = more relevant)",
    )
    text: str = Field(
        description="Exact raw text content of chunk, completely unedited",
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Sequence index of chunk within source document if known",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw vector store metadata dictionary for citation traceability",
    )

    @field_validator("chunk_id", mode="before")
    @classmethod
    def validate_chunk_id_non_empty(cls, v: Any) -> str:
        """Validate that chunk_id is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("chunk_id must be a non-empty string.")
        return v.strip()

    @field_validator("text", mode="before")
    @classmethod
    def validate_text_non_empty(cls, v: Any) -> str:
        """Validate that text is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("text must be a non-empty string.")
        return v

    @field_validator("distance", mode="before")
    @classmethod
    def validate_distance_finite(cls, v: Any) -> float:
        """Validate that distance is a finite float."""
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("distance must be a numeric float.")
        val = float(v)
        if not math.isfinite(val):
            raise ValueError("distance must be a finite float.")
        return val

    @property
    def character_count(self) -> int:
        """Return the exact length in characters of the chunk text."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Return the word count of the chunk text."""
        return len(self.text.split())


class ContextConfig(BaseModel):
    """Configuration for Context Construction (Phase 9.11).

    Controls:
    - max_characters: Optional total character budget across all chunks.
    - raise_on_budget_exceeded: If True, raises ContextBudgetExceededError when
      total characters exceed max_characters. If False, marks exceeds_budget=True
      on RetrievalContext without altering or truncating chunk text.
    - require_provenance: If True, requires document_id to be present on all chunks.
    """

    model_config = ConfigDict(frozen=True)

    max_characters: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional maximum total character budget across all chunks",
    )
    raise_on_budget_exceeded: bool = Field(
        default=False,
        description=(
            "If True, raises ContextBudgetExceededError when context exceeds "
            "max_characters. If False, marks context.exceeds_budget=True."
        ),
    )
    require_provenance: bool = Field(
        default=False,
        description=(
            "If True, requires document_id to be non-empty on all chunks; "
            "raises MissingProvenanceError otherwise."
        ),
    )

    @field_validator("max_characters", mode="before")
    @classmethod
    def validate_max_characters(cls, v: Any) -> Optional[int]:
        """Validate that max_characters is a positive integer or None."""
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(
                    f"max_characters must be an integer, got {type(v).__name__}."
                )
            if v <= 0:
                raise ValueError(f"max_characters must be greater than 0, got {v}.")
        return v


class RetrievalContext(BaseModel):
    """Structured, grounded context object built from Top-K retrieval (Phase 9.11).

    Aggregates Top-K retrieved chunks with retrieval provenance, configuration,
    and metadata for downstream consumption by the Research Analyst.
    Strictly preserves exact chunk text and ordering.
    """

    model_config = ConfigDict(frozen=True)

    query: Optional[str] = Field(
        default=None,
        description="Original query text or reference question if provided",
    )
    collection_name: str = Field(
        description="Target vector collection from which matches were retrieved",
    )
    distance_metric: str = Field(
        default="cosine_distance",
        description="Distance metric used by vector search",
    )
    total_retrieved: int = Field(
        ge=0,
        description="Total count of retrieved chunks included in context",
    )
    total_characters: int = Field(
        ge=0,
        description="Total character count across all included chunk texts",
    )
    total_words: int = Field(
        ge=0,
        description="Total word count across all included chunk texts",
    )
    chunks: List[ContextChunk] = Field(
        default_factory=list,
        description="Ordered list of context chunks preserving Phase 9.10 ranking",
    )
    has_evidence: bool = Field(
        description="True if context contains evidence chunks, False if empty",
    )
    max_distance_threshold: Optional[float] = Field(
        default=None,
        description="Maximum distance threshold applied during retrieval if any",
    )
    exceeds_budget: bool = Field(
        default=False,
        description="Whether total context size exceeded character budget",
    )
    built_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when context was constructed",
    )
    retrieval_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional retrieval metadata and parameters",
    )

    @property
    def is_empty(self) -> bool:
        """Return True if no evidence chunks were retrieved."""
        return len(self.chunks) == 0

    @property
    def document_ids(self) -> List[str]:
        """Return list of distinct source document IDs present in context."""
        seen = set()
        doc_ids = []
        for c in self.chunks:
            if c.document_id and c.document_id not in seen:
                seen.add(c.document_id)
                doc_ids.append(c.document_id)
        return doc_ids

    @property
    def tickers(self) -> List[str]:
        """Return list of distinct stock tickers present in context."""
        seen = set()
        tickers = []
        for c in self.chunks:
            if c.ticker and c.ticker not in seen:
                seen.add(c.ticker)
                tickers.append(c.ticker)
        return tickers

    @property
    def sources(self) -> List[str]:
        """Return list of distinct source document filenames/references."""
        seen = set()
        sources = []
        for c in self.chunks:
            if c.source and c.source not in seen:
                seen.add(c.source)
                sources.append(c.source)
        return sources
