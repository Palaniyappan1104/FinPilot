"""Domain models for Relevance and Evidence Sufficiency Checking (Phase 9.15).

Defines:
- RelevanceConfig: Policy configuration for distance thresholding and evidence
  requirements.
- RelevanceCheckResult: Structured result of the relevance evaluation, including
  candidate counts, best distance, rejection reason, and derived filtered context.

Distance Semantics:
- The FinPilot retrieval pipeline operates strictly on DISTANCE metrics (specifically
  cosine distance in ChromaDB, where 0.0 represents identical vectors and 2.0
  represents opposite vectors).
- Lower distance indicates higher semantic relevance.
- A retrieved chunk is deemed relevant if and only if:
    chunk.distance <= max_distance
- There is NO scientifically universal semantic threshold; the default threshold
  (0.65) is an explicit application-level policy tuned for normalized embedding
  spaces to distinguish grounded financial disclosures from out-of-domain noise.
"""

import math
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.context import ContextChunk, RetrievalContext

# Standardized failure reason constants
REASON_SUFFICIENT_EVIDENCE = "sufficient_evidence"
REASON_NO_RETRIEVED_EVIDENCE = "no_retrieved_evidence"
REASON_NO_EVIDENCE_WITHIN_THRESHOLD = "no_evidence_within_relevance_threshold"
REASON_INSUFFICIENT_RELEVANT_CHUNKS = "insufficient_relevant_chunks"
REASON_NOT_ALL_CHUNKS_RELEVANT = "not_all_chunks_relevant"


class RelevanceConfig(BaseModel):
    """Configuration for retrieval relevance and sufficiency checking (Phase 9.15).

    Policy Parameters:
    - max_distance: Maximum acceptable distance for a chunk to be considered relevant.
      Chunks with distance <= max_distance are accepted; chunks with distance >
      max_distance are deemed weak / irrelevant. Default 0.65 (application policy).
    - min_relevant_chunks: Minimum number of chunks satisfying distance <= max_distance
      required for the retrieval context to be declared sufficient. Default is 1.
    - require_all_relevant: If True, ALL retrieved chunks must satisfy max_distance.
      If False (default), partial relevance is supported: weak chunks are filtered out
      and the context proceeds to the Research Analyst as long as min_relevant_chunks
      is satisfied.
    """

    model_config = ConfigDict(frozen=True)

    max_distance: float = Field(
        default=0.65,
        ge=0.0,
        description=(
            "Maximum acceptable retrieval distance (lower = more relevant). "
            "Chunks with distance <= max_distance are deemed relevant. "
            "Default 0.65 is an application policy parameter, not a universal constant."
        ),
    )
    min_relevant_chunks: int = Field(
        default=1,
        ge=1,
        description=(
            "Minimum number of chunks satisfying distance <= max_distance required "
            "for the evidence context to be deemed sufficient."
        ),
    )
    require_all_relevant: bool = Field(
        default=False,
        description=(
            "If True, requires all retrieved chunks to be within max_distance. "
            "If False, allows partial relevance if min_relevant_chunks is satisfied."
        ),
    )

    @field_validator("max_distance", mode="before")
    @classmethod
    def validate_max_distance_finite(cls, v: Any) -> float:
        """Validate that max_distance is a non-negative finite float."""
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"max_distance must be a numeric float, got {type(v).__name__}."
            )
        val = float(v)
        if not math.isfinite(val):
            raise ValueError("max_distance must be a finite float.")
        if val < 0.0:
            raise ValueError(f"max_distance cannot be negative, got {val}.")
        return val

    @field_validator("min_relevant_chunks", mode="before")
    @classmethod
    def validate_min_relevant_chunks_positive(cls, v: Any) -> int:
        """Validate that min_relevant_chunks is an integer >= 1."""
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"min_relevant_chunks must be an integer, got {type(v).__name__}."
            )
        if v < 1:
            raise ValueError(f"min_relevant_chunks must be at least 1, got {v}.")
        return v


class RelevanceCheckResult(BaseModel):
    """Structured machine-readable result of the relevance check (Phase 9.15).

    Carries explicit metadata, candidate counts, distance metrics, and the
    derived unmutated context containing only relevant chunks.
    """

    model_config = ConfigDict(frozen=True)

    is_relevant: bool = Field(
        description="Whether retrieved evidence is sufficiently relevant to proceed"
    )
    reason: str = Field(
        description="Deterministic reason code explaining the relevance decision"
    )
    total_chunks: int = Field(
        ge=0,
        description="Total candidate chunks evaluated from original RetrievalContext",
    )
    relevant_chunks_count: int = Field(
        ge=0,
        description="Number of chunks meeting the distance <= max_distance condition",
    )
    rejected_chunks_count: int = Field(
        ge=0,
        description="Number of chunks exceeding max_distance threshold",
    )
    best_distance: Optional[float] = Field(
        default=None,
        description="Minimum distance score among evaluated chunks (None if empty)",
    )
    threshold_used: float = Field(
        ge=0.0,
        description="max_distance threshold configured for this evaluation",
    )
    min_chunks_required: int = Field(
        ge=1,
        description="min_relevant_chunks policy requirement applied",
    )
    relevant_chunks: List[ContextChunk] = Field(
        default_factory=list,
        description=(
            "Subset of chunks satisfying distance <= threshold, in original order"
        ),
    )
    rejected_chunks: List[ContextChunk] = Field(
        default_factory=list,
        description="Subset of chunks exceeding threshold, in original order",
    )
    derived_context: Optional[RetrievalContext] = Field(
        default=None,
        description=(
            "New unmutated RetrievalContext containing ONLY relevant chunks when "
            "is_relevant is True; None when evidence is insufficient."
        ),
    )

    @field_validator("best_distance", mode="before")
    @classmethod
    def validate_best_distance_finite(cls, v: Any) -> Optional[float]:
        """Validate best_distance is finite if provided."""
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("best_distance must be a numeric float or None.")
            val = float(v)
            if not math.isfinite(val):
                raise ValueError("best_distance must be a finite float.")
            return val
        return None
