"""Domain models for Top-K Retrieval (Phase 9.10).

Defines:
- TopKConfig: Configuration schema for Top-K retrieval policy and thresholding.
- TopKRetrievalResult: Output container holding ranked retrieved matches.
"""

import math
from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.vector_store import VectorSearchResult


class TopKConfig(BaseModel):
    """Configuration for Top-K retrieval and relevance thresholding (Phase 9.10).

    Controls:
    - k: Maximum number of nearest/most relevant records to retrieve (>= 1).
    - max_distance: Optional distance threshold (Phase 9.10.2). Candidate matches
      with distance > max_distance are filtered out as weak matches.
    """

    model_config = ConfigDict(frozen=True)

    k: int = Field(
        default=5,
        ge=1,
        description="Maximum number of relevant chunks to retrieve (must be >= 1)",
    )
    max_distance: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Optional maximum distance threshold. Candidate matches with distance "
            "> max_distance are excluded as weak matches."
        ),
    )

    @field_validator("k", mode="before")
    @classmethod
    def validate_k_is_positive_integer(cls, v: Any) -> int:
        """Validate that k is an integer strictly greater than zero."""
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"k must be an integer, got {type(v).__name__}.")
        if v <= 0:
            raise ValueError(f"k must be greater than 0, got {v}.")
        return v

    @field_validator("max_distance", mode="before")
    @classmethod
    def validate_max_distance_finite(cls, v: Any) -> Optional[float]:
        """Validate that max_distance is a non-negative finite float."""
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    "max_distance must be a numeric float or None, "
                    f"got {type(v).__name__}."
                )
            val = float(v)
            if not math.isfinite(val):
                raise ValueError("max_distance must be a finite float.")
            if val < 0.0:
                raise ValueError(f"max_distance cannot be negative, got {val}.")
            return val
        return None


class TopKRetrievalResult(BaseModel):
    """Container for the output of Top-K retrieval (Phase 9.10).

    Couples the ranked subset of VectorSearchResult matches with retrieval
    provenance, candidate counts, distance metric, and configuration parameters.
    """

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(
        description="Target vector collection from which matches were retrieved"
    )
    k: int = Field(
        ge=1,
        description="Configured maximum number of results requested",
    )
    results: List[VectorSearchResult] = Field(
        default_factory=list,
        description=(
            "Ranked list of retrieved matches ordered from most to least "
            "relevant (ascending distance)"
        ),
    )
    total_candidates: int = Field(
        ge=0,
        description="Total candidate matches evaluated from similarity search",
    )
    total_retrieved: int = Field(
        ge=0,
        description="Total number of results selected and returned (<= k)",
    )
    distance_metric: str = Field(
        default="cosine_distance",
        description="Distance metric used for relevance ordering",
    )
    max_distance_threshold: Optional[float] = Field(
        default=None,
        description="Distance threshold applied during retrieval if configured",
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when Top-K retrieval was executed",
    )

    @property
    def is_empty(self) -> bool:
        """Return True if no matches were retrieved."""
        return len(self.results) == 0

    @property
    def document_ids(self) -> List[str]:
        """Return list of distinct source document IDs present in retrieved chunks."""
        seen = set()
        doc_ids = []
        for r in self.results:
            d_id = r.document_id
            if d_id and d_id not in seen:
                seen.add(d_id)
                doc_ids.append(d_id)
        return doc_ids

    @property
    def tickers(self) -> List[str]:
        """Return list of distinct stock tickers present in retrieved chunks."""
        seen = set()
        tickers = []
        for r in self.results:
            t = r.ticker
            if t and t not in seen:
                seen.add(t)
                tickers.append(t)
        return tickers
