"""API request, response, and error schemas for FinPilot FastAPI Backend (Phase 15.2).

Defines typed Pydantic models for:
- Standardized API error responses and validation details (15.2.1, 15.2.2)
- Chat / conversational queries (15.1.2, 15.2.1)
- Company analysis requests (15.1.3, 15.2.1)
- Clarification answer submissions (15.1.4, 15.2.1)
- Research queries (15.1.6, 15.2.1)
- Analysis execution and status responses (15.1.7, 15.2.1)
- Report retrieval responses (15.1.8, 15.2.1)
- Health check and document upload response models (15.1.1, 15.1.5, 15.2.1)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.report_schema import FinalReport
from app.api.v1.health import HealthResponse
from app.models.documents import DocumentUploadResponse

# ---------------------------------------------------------------------------
# Validation Constants and Helpers (15.2.2)
# ---------------------------------------------------------------------------
TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def _normalize_and_validate_ticker(ticker: Optional[str]) -> Optional[str]:
    """Normalize and validate a stock ticker symbol.

    Args:
        ticker: Input stock ticker string or None.

    Returns:
        Optional[str]: Uppercased, stripped ticker string, or None if input was None.

    Raises:
        ValueError: If ticker is empty, whitespace, or contains invalid characters.
    """
    if ticker is None:
        return None
    clean = ticker.strip().upper()
    if not clean:
        raise ValueError("Ticker symbol cannot be empty or whitespace only.")
    if not TICKER_PATTERN.match(clean):
        raise ValueError(
            f"Invalid ticker symbol '{ticker}'. Expected 1-10 uppercase alphanumeric "
            f"characters (dots and hyphens allowed)."
        )
    return clean


# ---------------------------------------------------------------------------
# Error Response Models (15.2.1, 15.2.2)
# ---------------------------------------------------------------------------
class ValidationErrorDetail(BaseModel):
    """Detailed validation failure item matching FastAPI's sanitized error structure."""

    model_config = ConfigDict(extra="ignore")

    loc: List[Any] = Field(
        ..., description="Location / path of the invalid input parameter"
    )
    msg: str = Field(..., description="Human-readable validation failure message")
    type: str = Field(..., description="Validation error classification identifier")


class APIErrorDetail(BaseModel):
    """Standardized error payload encapsulated within API responses."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(
        ...,
        description=(
            "Machine-readable error classification code (e.g. VALIDATION_ERROR, "
            "HTTP_ERROR, INTERNAL_SERVER_ERROR)"
        ),
    )
    message: str = Field(..., description="Human-readable explanation of the error")
    details: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Detailed field-level validation errors or contextual diagnostic entries"
        ),
    )


class APIErrorEnvelope(BaseModel):
    """Top-level error response envelope returned on 4xx and 5xx failures."""

    model_config = ConfigDict(extra="ignore")

    error: APIErrorDetail = Field(..., description="Standard FinPilot error container")


# ---------------------------------------------------------------------------
# Request Models (15.2.1, 15.2.2)
# ---------------------------------------------------------------------------
class InvestorProfilePayload(BaseModel):
    """Structured investor profile payload with strict field constraints."""

    model_config = ConfigDict(extra="ignore")

    target_company: Optional[str] = Field(
        default=None, max_length=100, description="Target company name"
    )
    ticker: Optional[str] = Field(
        default=None, max_length=10, description="Company stock ticker symbol"
    )
    investment_goal: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Investment objective (e.g., capital preservation, growth)",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Target holding period (e.g., 3-5 years, long-term)",
    )
    capital_amount: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Available capital to invest in currency units "
            "(strictly positive finite number)"
        ),
    )
    risk_tolerance: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Risk tolerance profile (e.g., conservative, moderate, aggressive)",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def validate_ticker(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Ticker symbol must be a string.")
        return _normalize_and_validate_ticker(v)

    @field_validator("capital_amount", mode="after")
    @classmethod
    def validate_capital_amount(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Capital amount must be a finite numerical value.")
        if v > 1e15:
            raise ValueError("Capital amount exceeds maximum allowed threshold.")
        return v

    @field_validator("target_company", "investment_goal", "time_horizon", mode="before")
    @classmethod
    def sanitize_text_fields(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        clean = str(v).strip()
        return clean if clean else None

    @field_validator("risk_tolerance", mode="before")
    @classmethod
    def normalize_risk_tolerance(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        clean = str(v).strip().lower()
        return clean if clean else None


class ChatQueryRequest(BaseModel):
    """Request payload for the conversational chat/query endpoint (15.1.2, 15.2.1)."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Natural language user query",
    )
    investor_profile: Optional[InvestorProfilePayload] = Field(
        default=None, description="Optional existing investor constraints"
    )
    documents_available: bool = Field(
        default=False, description="Whether uploaded document context is available"
    )
    trace_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional client correlation / trace identifier",
    )

    @field_validator("query", mode="after")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Query cannot be empty or whitespace only.")
        return clean

    @field_validator("trace_id", mode="before")
    @classmethod
    def sanitize_trace_id(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        clean = str(v).strip()
        return clean if clean else None


class CompanyAnalysisRequest(BaseModel):
    """Request payload for company analysis endpoint (15.1.3, 15.2.1)."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Target equity ticker symbol (e.g., AAPL)",
    )
    target_company: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Target company name (e.g., Apple Inc.)",
    )
    query: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Optional specific investment query or objective",
    )
    investor_profile: Optional[InvestorProfilePayload] = Field(
        default=None, description="Investor financial constraints and preferences"
    )
    documents_available: bool = Field(
        default=False, description="Whether uploaded research filings are available"
    )
    clarification_answers: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional pre-supplied answers to missing parameters"
    )
    trace_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional correlation / trace identifier",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def validate_ticker_symbol(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Ticker symbol must be a string.")
        clean = _normalize_and_validate_ticker(v)
        if clean is None:
            raise ValueError("Ticker symbol is required.")
        return clean

    @field_validator("query", "target_company", "trace_id", mode="before")
    @classmethod
    def sanitize_optional_text(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        clean = str(v).strip()
        return clean if clean else None

    @field_validator("clarification_answers", mode="after")
    @classmethod
    def validate_clarification_answers_keys(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        for key in v.keys():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Clarification answer keys must be non-empty strings.")
        return v


class ClarificationSubmitRequest(BaseModel):
    """Request payload for submitting clarification answers (15.1.4, 15.2.1)."""

    model_config = ConfigDict(extra="ignore")

    clarification_answers: Dict[str, Any] = Field(
        ...,
        min_length=1,
        description="Key-value mapping of answers to required profile fields",
    )
    analysis_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional ID of the halted analysis session",
    )
    query: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Original or updated user prompt",
    )
    ticker: Optional[str] = Field(
        default=None,
        max_length=10,
        description="Stock ticker symbol if resuming without query",
    )
    investor_profile: Optional[InvestorProfilePayload] = Field(
        default=None, description="Baseline investor profile before clarification"
    )
    documents_available: bool = Field(
        default=False, description="Whether research document context is available"
    )
    trace_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional correlation / trace identifier",
    )

    @field_validator("clarification_answers", mode="after")
    @classmethod
    def validate_answers(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("Clarification answers cannot be empty.")
        for key in v.keys():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Clarification answer keys must be non-empty strings.")
        return v

    @field_validator("ticker", mode="before")
    @classmethod
    def validate_ticker(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Ticker symbol must be a string.")
        return _normalize_and_validate_ticker(v)

    @field_validator("query", "analysis_id", "trace_id", mode="before")
    @classmethod
    def sanitize_optional_text(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        clean = str(v).strip()
        return clean if clean else None


class ResearchQueryRequest(BaseModel):
    """Request payload for research query against documents (15.1.6, 15.2.1)."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Factual research question regarding documents or filings",
    )
    ticker: Optional[str] = Field(
        default=None,
        max_length=10,
        description="Associated stock ticker symbol",
    )
    documents_available: bool = Field(
        default=True,
        description="Set to true to activate research document grounding",
    )
    trace_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional correlation / trace identifier",
    )

    @field_validator("query", mode="after")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Query cannot be empty or whitespace only.")
        return clean

    @field_validator("ticker", mode="before")
    @classmethod
    def validate_ticker(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Ticker symbol must be a string.")
        return _normalize_and_validate_ticker(v)

    @field_validator("trace_id", mode="before")
    @classmethod
    def sanitize_trace_id(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        clean = str(v).strip()
        return clean if clean else None


# ---------------------------------------------------------------------------
# Response Models (15.2.1)
# ---------------------------------------------------------------------------
class AnalysisExecutionResponse(BaseModel):
    """Response returned upon running an analysis or chat query."""

    model_config = ConfigDict(extra="ignore")

    analysis_id: str = Field(..., description="Unique analysis session identifier")
    trace_id: str = Field(..., description="Correlated trace identifier")
    status: Literal["completed", "clarification_needed", "failed", "running"] = Field(
        ..., description="Overall workflow execution outcome"
    )
    clarification_needed: bool = Field(
        default=False,
        description="True if missing investor constraints halt the workflow",
    )
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="List of clarification questions when clarification is required",
    )
    investor_profile: Optional[Dict[str, Any]] = Field(
        default=None, description="Reflected investor profile captured during workflow"
    )
    report: Optional[FinalReport] = Field(
        default=None,
        description="Complete structured final investment report when completed",
    )
    report_id: Optional[str] = Field(
        default=None, description="Identifier to retrieve this completed report later"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the execution failed"
    )


class AnalysisStatusResponse(BaseModel):
    """Status tracking response for an analysis session (15.1.7, 15.2.1, 15.3.2)."""

    model_config = ConfigDict(extra="ignore")

    analysis_id: str = Field(..., description="Analysis session identifier")
    trace_id: str = Field(..., description="Correlated trace identifier")
    status: Literal["completed", "clarification_needed", "failed", "running"] = Field(
        ..., description="Current status of the analysis session"
    )
    ticker: Optional[str] = Field(
        default=None, description="Associated stock ticker symbol"
    )
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="Clarification questions if currently awaiting clarification",
    )
    report_id: Optional[str] = Field(
        default=None, description="Completed report identifier if available"
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    completed_at: Optional[str] = Field(
        default=None, description="ISO 8601 completion timestamp"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    progress_stage: Optional[str] = Field(
        default=None,
        description=(
            "Current workflow execution stage "
            "(e.g., 'running', 'completed', 'failed')"
        ),
    )


class ReportRetrievalResponse(BaseModel):
    """Response payload when fetching a completed report by ID (15.1.8, 15.2.1)."""

    model_config = ConfigDict(extra="ignore")

    report_id: str = Field(..., description="Report identifier")
    format: Literal["json", "markdown", "summary"] = Field(
        ..., description="Output rendering format of the retrieved report"
    )
    report: Optional[FinalReport] = Field(
        default=None,
        description="Full structured FinalReport model when format is 'json'",
    )
    rendered_content: Optional[str] = Field(
        default=None,
        description="Rendered markdown or executive summary text content",
    )


__all__ = [
    "ValidationErrorDetail",
    "APIErrorDetail",
    "APIErrorEnvelope",
    "InvestorProfilePayload",
    "ChatQueryRequest",
    "CompanyAnalysisRequest",
    "ClarificationSubmitRequest",
    "ResearchQueryRequest",
    "AnalysisExecutionResponse",
    "AnalysisStatusResponse",
    "ReportRetrievalResponse",
    "HealthResponse",
    "DocumentUploadResponse",
]
