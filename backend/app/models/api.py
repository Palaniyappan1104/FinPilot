"""API request and response schemas for FinPilot FastAPI Backend (Phase 15.1).

Defines typed Pydantic models for:
- Chat / conversational queries (15.1.2)
- Company analysis requests (15.1.3)
- Clarification answer submissions (15.1.4)
- Research queries (15.1.6)
- Analysis execution and status responses (15.1.7)
- Report retrieval responses (15.1.8)
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agents.report_schema import FinalReport


class InvestorProfilePayload(BaseModel):
    """Optional structured investor profile payload."""

    model_config = ConfigDict(extra="ignore")

    target_company: Optional[str] = Field(
        default=None, description="Target company name"
    )
    ticker: Optional[str] = Field(
        default=None, description="Company stock ticker symbol"
    )
    investment_goal: Optional[str] = Field(
        default=None,
        description="Investment objective (e.g., capital preservation, growth)",
    )
    time_horizon: Optional[str] = Field(
        default=None, description="Target holding period (e.g., 3-5 years, long-term)"
    )
    capital_amount: Optional[float] = Field(
        default=None, gt=0, description="Available capital to invest in currency units"
    )
    risk_tolerance: Optional[str] = Field(
        default=None,
        description="Risk tolerance profile (e.g., conservative, moderate, aggressive)",
    )


class ChatQueryRequest(BaseModel):
    """Request payload for the conversational chat/query endpoint (15.1.2)."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="Natural language user query")
    investor_profile: Optional[InvestorProfilePayload] = Field(
        default=None, description="Optional existing investor constraints"
    )
    documents_available: bool = Field(
        default=False, description="Whether uploaded document context is available"
    )
    trace_id: Optional[str] = Field(
        default=None, description="Optional client correlation / trace identifier"
    )


class CompanyAnalysisRequest(BaseModel):
    """Request payload for company analysis endpoint (15.1.3)."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Target equity ticker symbol (e.g., AAPL)",
    )
    target_company: Optional[str] = Field(
        default=None, description="Target company name (e.g., Apple Inc.)"
    )
    query: Optional[str] = Field(
        default=None, description="Optional specific investment query or objective"
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
        default=None, description="Optional correlation / trace identifier"
    )


class ClarificationSubmitRequest(BaseModel):
    """Request payload for submitting clarification answers (15.1.4)."""

    model_config = ConfigDict(extra="ignore")

    clarification_answers: Dict[str, Any] = Field(
        ...,
        min_length=1,
        description="Key-value mapping of answers to required profile fields",
    )
    analysis_id: Optional[str] = Field(
        default=None, description="Optional ID of the halted analysis session"
    )
    query: Optional[str] = Field(
        default=None, description="Original or updated user prompt"
    )
    ticker: Optional[str] = Field(
        default=None, description="Stock ticker symbol if resuming without query"
    )
    investor_profile: Optional[InvestorProfilePayload] = Field(
        default=None, description="Baseline investor profile before clarification"
    )
    documents_available: bool = Field(
        default=False, description="Whether research document context is available"
    )
    trace_id: Optional[str] = Field(
        default=None, description="Optional correlation / trace identifier"
    )


class ResearchQueryRequest(BaseModel):
    """Request payload for research query against documents (15.1.6)."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        min_length=1,
        description="Factual research question regarding documents or filings",
    )
    ticker: Optional[str] = Field(
        default=None, description="Associated stock ticker symbol"
    )
    documents_available: bool = Field(
        default=True,
        description="Set to true to activate research document grounding",
    )
    trace_id: Optional[str] = Field(
        default=None, description="Optional correlation / trace identifier"
    )


class AnalysisExecutionResponse(BaseModel):
    """Response returned upon running an analysis or chat query."""

    model_config = ConfigDict(extra="ignore")

    analysis_id: str = Field(..., description="Unique analysis session identifier")
    trace_id: str = Field(..., description="Correlated trace identifier")
    status: Literal["completed", "clarification_needed", "failed"] = Field(
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
    """Status tracking response for an analysis session (15.1.7)."""

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


class ReportRetrievalResponse(BaseModel):
    """Response payload when fetching a completed report by ID (15.1.8)."""

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
