"""Pydantic schemas and typed exceptions for Research Analyst Agent (Phase 9.12).

Defines:
- ResearchEvidenceRef: Citation reference pointing to a specific retrieved chunk.
- ResearchFinding: Structured factual claim supported by evidence citations.
- ResearchAnalysisOutput: Structured output returned by the Research Analyst Agent.
- ResearchAnalystInput: Validated input payload containing query and RetrievalContext.
- Typed exceptions for research validation and provenance enforcement.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.context import RetrievalContext
from app.models.relevance import RelevanceConfig

# ===========================================================================
# TYPED EXCEPTIONS (Phase 9.12)
# ===========================================================================


class ResearchAnalystError(Exception):
    """Base exception for Research Analyst Agent errors."""

    def __init__(self, message: str, code: str = "RESEARCH_ANALYST_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class ResearchValidationError(ResearchAnalystError):
    """Raised when research analysis output violates grounding or schema consistency."""

    def __init__(self, message: str, code: str = "RESEARCH_VALIDATION_ERROR") -> None:
        super().__init__(message=message, code=code)


class InvalidProvenanceError(ResearchValidationError):
    """Raised when an evidence citation references non-existent or
    conflicting provenance.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="INVALID_PROVENANCE_ERROR")


# ===========================================================================
# SCHEMAS
# ===========================================================================


class ResearchEvidenceRef(BaseModel):
    """Citation reference linking a factual claim to an exact retrieved chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        description="Exact chunk identifier matching a chunk in RetrievalContext"
    )
    document_id: str = Field(description="Source document identifier")
    source_document: str = Field(
        description="Source document filename or reference identifier"
    )
    page_numbers: List[int] = Field(
        default_factory=list,
        description="1-indexed physical page numbers containing the evidence",
    )

    @field_validator("chunk_id", "document_id", "source_document", mode="before")
    @classmethod
    def validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        """Validate that citation identifiers are non-empty strings."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()


class ResearchFinding(BaseModel):
    """Factual claim extracted from retrieved documents with evidence citations."""

    model_config = ConfigDict(frozen=True)

    claim: str = Field(
        description="Factual claim or insight grounded strictly in retrieved documents"
    )
    evidence: List[ResearchEvidenceRef] = Field(
        default_factory=list,
        description="Evidence references substantiating this specific claim",
    )

    @field_validator("claim", mode="before")
    @classmethod
    def validate_claim_non_empty(cls, v: Any) -> str:
        """Validate that finding claim is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("claim must be a non-empty string.")
        return v.strip()


class ResearchAnalysisOutput(BaseModel):
    """Structured output produced by the Research Analyst Agent (Phase 9.12).

    Contains the document-grounded research answer, key findings with citations,
    confidence score, and evidence sufficiency status.
    """

    query: str = Field(description="Original research question answered")
    answer: str = Field(
        description="Comprehensive answer synthesized strictly from retrieved evidence"
    )
    key_findings: List[ResearchFinding] = Field(
        default_factory=list,
        description="Key factual findings extracted from the retrieved documents",
    )
    evidence: List[ResearchEvidenceRef] = Field(
        default_factory=list,
        description="Complete list of distinct evidence chunks cited in the answer",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score reflecting evidence completeness (0.0 to 1.0)",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="Whether evidence was insufficient to answer the question",
    )
    insufficient_reason: Optional[str] = Field(
        default=None,
        description="Explanation when insufficient_evidence is True",
    )

    @field_validator("query", "answer", mode="before")
    @classmethod
    def validate_text_fields(cls, v: Any, info: Any) -> str:
        """Validate that query and answer are non-empty strings."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string.")
        return v.strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence_bounds(cls, v: Any) -> float:
        """Validate that confidence is a float strictly between 0.0 and 1.0."""
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"confidence must be a float, got {type(v).__name__}.")
        val = float(v)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {val}.")
        return val

    @field_validator("insufficient_reason")
    @classmethod
    def validate_insufficient_reason_consistency(
        cls, v: Optional[str], info: Any
    ) -> Optional[str]:
        """Enforce consistency between insufficient_evidence and insufficient_reason."""
        is_insufficient = info.data.get("insufficient_evidence", False)
        if is_insufficient:
            if not v or not v.strip():
                raise ValueError(
                    "insufficient_reason must be provided when "
                    "insufficient_evidence is True."
                )
            return v.strip()
        return v.strip() if v else None


class ResearchAnalystInput(BaseModel):
    """Validated input payload for ResearchAnalystAgent."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="User research query")
    context: RetrievalContext = Field(
        description="Structured retrieval context from Phase 9.11"
    )
    relevance_config: Optional[RelevanceConfig] = Field(
        default=None,
        description="Optional relevance check policy configuration (Phase 9.15)",
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_query_non_empty(cls, v: Any) -> str:
        """Validate query is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("query must be a non-empty string.")
        return v.strip()
