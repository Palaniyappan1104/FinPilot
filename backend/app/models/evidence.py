"""Domain models and exceptions for Source/Evidence Tracking (Phase 9.13).

Defines:
- ResearchEvidence: Immutable record representing a traceable evidence unit
  linking a retrieved chunk to its source document, physical pages, and metadata.
- compute_evidence_id: Deterministic evidence identifier generator.
- Typed exceptions for evidence tracking, validation, resolution, and provenance.
"""

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ===========================================================================
# TYPED EXCEPTIONS (Phase 9.13)
# ===========================================================================


class EvidenceTrackingError(Exception):
    """Base exception for all source/evidence tracking operations."""

    def __init__(self, message: str, code: str = "EVIDENCE_TRACKING_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class EvidenceValidationError(EvidenceTrackingError, ValueError):
    """Raised when evidence metadata is malformed or invalid."""

    def __init__(self, message: str, code: str = "EVIDENCE_VALIDATION_ERROR") -> None:
        super().__init__(message=message, code=code)


class EvidenceNotFoundError(EvidenceTrackingError, KeyError):
    """Raised when a cited evidence ID or chunk ID cannot be resolved."""

    def __init__(self, message: str, code: str = "EVIDENCE_NOT_FOUND") -> None:
        super().__init__(message=message, code=code)


class ConflictingProvenanceError(EvidenceTrackingError):
    """Raised when a chunk or evidence item has conflicting provenance."""

    def __init__(self, message: str, code: str = "CONFLICTING_PROVENANCE") -> None:
        super().__init__(message=message, code=code)


class FabricatedProvenanceError(EvidenceTrackingError):
    """Raised when a citation references non-existent or fabricated evidence."""

    def __init__(self, message: str, code: str = "FABRICATED_PROVENANCE") -> None:
        super().__init__(message=message, code=code)


# ===========================================================================
# DETERMINISTIC EVIDENCE ID
# ===========================================================================


def compute_evidence_id(document_id: str, chunk_id: str) -> str:
    """Compute a collision-safe deterministic evidence identifier.

    Format: 'ev_{document_id}_{chunk_id}'

    Args:
        document_id: Source document identifier.
        chunk_id: Chunk identifier within the document/vector collection.

    Returns:
        Deterministic evidence ID string.

    Raises:
        EvidenceValidationError: If document_id or chunk_id is empty/whitespace.
    """
    if not isinstance(document_id, str) or not document_id.strip():
        raise EvidenceValidationError(
            "document_id must be a non-empty string for evidence ID generation."
        )
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise EvidenceValidationError(
            "chunk_id must be a non-empty string for evidence ID generation."
        )
    return f"ev_{document_id.strip()}_{chunk_id.strip()}"


# ===========================================================================
# EVIDENCE MODEL
# ===========================================================================


class ResearchEvidence(BaseModel):
    """Traceable evidence unit linking retrieved chunk to source provenance.

    Phase 9.13: Represents a verified evidence item with full document, page,
    section, and retrieval distance provenance.
    Immutable to guarantee integrity throughout the research pipeline.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(
        description="Deterministic evidence identifier ('ev_{doc_id}_{chunk_id}')"
    )
    chunk_id: str = Field(
        description="Source chunk identifier matching vector store / context"
    )
    document_id: str = Field(description="Unique source document identifier")
    source_document: str = Field(
        description="Source document filename or equivalent reference"
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Associated stock ticker or company identifier",
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Classified document category (e.g. 10-K, annual_report)",
    )
    page_numbers: List[int] = Field(
        default_factory=list,
        description="1-indexed physical page numbers containing this evidence",
    )
    section: Optional[str] = Field(
        default=None,
        description="Detected section heading or structural location in document",
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Sequential index of chunk in source document if known",
    )
    distance: float = Field(
        description="Vector search retrieval distance (lower = more similar)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant vector store metadata dictionary (excluding raw text)",
    )

    @field_validator(
        "evidence_id", "chunk_id", "document_id", "source_document", mode="before"
    )
    @classmethod
    def validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        """Validate that essential provenance identifiers are non-empty strings."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("distance", mode="before")
    @classmethod
    def validate_distance_finite(cls, v: Any) -> float:
        """Validate that distance is a finite numeric float."""
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("distance must be a numeric float.")
        val = float(v)
        if not math.isfinite(val):
            raise ValueError("distance must be a finite float.")
        return val

    @field_validator("page_numbers", mode="before")
    @classmethod
    def validate_page_numbers(cls, v: Any) -> List[int]:
        """Validate that page numbers are positive 1-indexed integers."""
        if v is None:
            return []
        if not isinstance(v, (list, tuple)):
            raise ValueError("page_numbers must be a list of integers.")
        pages: List[int] = []
        for p in v:
            if isinstance(p, bool) or not isinstance(p, int) or p < 1:
                raise ValueError(f"Page numbers must be integers >= 1, got {p}.")
            pages.append(p)
        return pages

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Return clean provenance metadata for inspection, logging, and citation."""
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_document": self.source_document,
            "ticker": self.ticker,
            "document_type": self.document_type,
            "page_numbers": list(self.page_numbers),
            "section": self.section,
            "chunk_index": self.chunk_index,
            "distance": self.distance,
        }

    def to_evidence_ref(self) -> Any:
        """Convert this evidence item into a Phase 9.12 ResearchEvidenceRef."""
        from app.agents.research_schema import ResearchEvidenceRef

        return ResearchEvidenceRef(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            source_document=self.source_document,
            page_numbers=list(self.page_numbers),
        )
