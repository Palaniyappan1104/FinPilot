"""Pydantic schemas and input aggregation contracts for Report Aggregator (Phase 11.1).

Phase 11.1 establishes:
- 11.1.1: Unified analysis schema combining all specialist outputs + investor profile.
- 11.1.2: Per-specialist evidence/source reference preservation.

Key Architecture:
- `SpecialistStatus`: Explicit availability lifecycle (AVAILABLE, MISSING, etc.).
- `AggregatedEvidenceItem`: Unified evidence record preserving source specialist
  identity, reference ID, details, and optional document/chunk provenance.
- `AggregatorInvestorProfile`: Investor constraints and profile model.
- `ReportAggregatorInput`: Structured input contract for the Report Aggregator
  consuming all five specialists independently with partial availability.
- `UnifiedSpecialistAnalysis`: Unified analysis schema combining all specialist
  outputs, status tracking, and per-specialist evidence links without flattening.
- `ReportAggregatorError`, `ReportAggregatorValidationError`: Typed exceptions.
"""

import math
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.fundamental_schema import FundamentalAnalysisOutput
from app.agents.news_schema import NewsAnalysisOutput
from app.agents.research_schema import ResearchAnalysisOutput, ResearchEvidenceRef
from app.agents.risk_schema import RiskAnalysisOutput, RiskEvidenceRef
from app.agents.state import GraphState, InvestorProfile
from app.agents.technical_schema import TechnicalAnalysisOutput

# ===========================================================================
# ENUMS & CONSTANTS
# ===========================================================================


class SpecialistStatus(str, Enum):
    """Explicit availability and execution status for a specialist agent."""

    AVAILABLE = "available"  # Valid, successful specialist output provided
    MISSING = "missing"  # Specialist was not invoked, skipped, or not provided
    FAILED = "failed"  # Specialist execution failed or raised an error
    EMPTY = "empty"  # Specialist produced an empty or ungrounded assessment


SpecialistType = Literal[
    "technical",
    "fundamental",
    "news",
    "research",
    "risk",
]

ALL_SPECIALISTS: List[SpecialistType] = [
    "technical",
    "fundamental",
    "news",
    "research",
    "risk",
]

CORE_SPECIALISTS: List[SpecialistType] = [
    "technical",
    "fundamental",
    "news",
    "risk",
]

# Prohibited advisory patterns strictly disallowed in aggregator schemas
PROHIBITED_ADVICE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:buy|sell|strong\s+buy|strong\s+sell)\s+recommendation\b", re.I),
    re.compile(
        r"\brecommend(?:s|ed|ing)?\s+(?:to\s+)?(?:buy(?:ing)?|sell(?:ing)?|hold(?:ing)?)\b",
        re.I,
    ),
    re.compile(r"\b(?:target\s+price|price\s+target)\b", re.I),
    re.compile(r"\bguaranteed\s+(?:return|profit|gain)s?\b", re.I),
    re.compile(r"\brisk[- ]free\b", re.I),
    re.compile(r"\b(?:investors?\s+must|you\s+should)\s+(?:buy|sell|hold)\b", re.I),
]


# ===========================================================================
# TYPED EXCEPTIONS
# ===========================================================================


class ReportAggregatorError(Exception):
    """Base exception for Report Aggregator errors."""

    def __init__(self, message: str, code: str = "REPORT_AGGREGATOR_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class ReportAggregatorValidationError(ReportAggregatorError, ValueError):
    """Raised when aggregator inputs or schemas violate validation or safety rules."""

    def __init__(self, message: str, code: str = "AGGREGATOR_VALIDATION_ERROR") -> None:
        super().__init__(message=message, code=code)


# ===========================================================================
# EVIDENCE & PROVENANCE PRESERVATION (11.1.2)
# ===========================================================================


class AggregatedEvidenceItem(BaseModel):
    """Unified evidence record preserving exact specialist attribution and provenance.

    Fulfills Phase 11.1.2 requirement:
    Ensures that when evidence is aggregated, the source specialist, original
    reference key/id, factual content, and any document citations (document_id,
    chunk_id, page_numbers) are strictly retained and never flattened away.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    specialist: SpecialistType = Field(
        ...,
        description="Originating specialist analyst that produced this evidence.",
    )
    reference_id: str = Field(
        ...,
        description="Specialist-specific reference key (metric, article ID, chunk ID).",
    )
    detail: str = Field(
        ...,
        description="Factual observation, metric finding, excerpt, or evidence text.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Source document identifier if from document/research analysis.",
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Exact text chunk identifier if from research retrieval.",
    )
    page_numbers: List[int] = Field(
        default_factory=list,
        description="Physical page numbers if from document citations.",
    )

    @field_validator("reference_id", "detail")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned

    @classmethod
    def from_technical_evidence(
        cls, evidence_str: str, index: int = 0
    ) -> "AggregatedEvidenceItem":
        """Wrap a technical evidence string into an attributed evidence item."""
        cleaned = evidence_str.strip()
        if not cleaned:
            raise ValueError("Technical evidence string cannot be empty.")
        return cls(
            specialist="technical",
            reference_id=f"tech_ev_{index}",
            detail=cleaned,
        )

    @classmethod
    def from_fundamental_metric(
        cls, metric_name: str, explanation: str
    ) -> "AggregatedEvidenceItem":
        """Wrap fundamental metric and explanation into attributed evidence item."""
        return cls(
            specialist="fundamental",
            reference_id=metric_name.strip(),
            detail=explanation.strip(),
        )

    @classmethod
    def from_news_recent_item(
        cls, article_id: str, headline: str, source: str
    ) -> "AggregatedEvidenceItem":
        """Wrap a news article citation into an attributed evidence item."""
        return cls(
            specialist="news",
            reference_id=article_id.strip(),
            detail=f"[{source.strip()}] {headline.strip()}",
        )

    @classmethod
    def from_research_evidence(
        cls, ref: Union[ResearchEvidenceRef, Dict[str, Any]]
    ) -> "AggregatedEvidenceItem":
        """Wrap a ResearchEvidenceRef into an attributed evidence item."""
        if isinstance(ref, dict):
            ref = ResearchEvidenceRef.model_validate(ref)
        return cls(
            specialist="research",
            reference_id=ref.chunk_id,
            detail=f"Source: {ref.source_document}",
            document_id=ref.document_id,
            chunk_id=ref.chunk_id,
            page_numbers=list(ref.page_numbers),
        )

    @classmethod
    def from_risk_evidence(
        cls, ref: Union[RiskEvidenceRef, Dict[str, Any]]
    ) -> "AggregatedEvidenceItem":
        """Wrap a RiskEvidenceRef into an attributed evidence item."""
        if isinstance(ref, dict):
            ref = RiskEvidenceRef.model_validate(ref)
        return cls(
            specialist="risk",
            reference_id=f"{ref.source_type}:{ref.reference_id}",
            detail=ref.detail,
        )


# ===========================================================================
# SYNTHESIS, AGREEMENT & CONFLICT MODELS (Phase 11.2)
# ===========================================================================


class SynthesisFinding(BaseModel):
    """An area of consensus or agreement supported by multiple specialists."""

    model_config = ConfigDict(extra="ignore")

    topic: str = Field(
        ...,
        description="Concise topic of agreement (e.g., 'Cash Flow Strength').",
    )
    summary: str = Field(
        ...,
        description="Synthesized description grounded strictly in evidence.",
    )
    supporting_specialists: List[SpecialistType] = Field(
        ...,
        min_length=1,
        description="Specialist agents whose findings support this agreement.",
    )
    evidence: List[AggregatedEvidenceItem] = Field(
        default_factory=list,
        description="Specific attributed evidence items supporting this finding.",
    )

    @field_validator("topic", "summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned


class SignalConflict(BaseModel):
    """An identified tension or contradiction between two or more specialists."""

    model_config = ConfigDict(extra="ignore")

    topic: str = Field(
        ...,
        description=(
            "Nature of tension (e.g., 'Technical Strength vs Fundamental Weakness')."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Objective explanation of the conflict without artificial consensus."
        ),
    )
    specialist_positions: Dict[str, str] = Field(
        ...,
        description="Position/finding of each conflicting specialist.",
    )
    involved_specialists: List[SpecialistType] = Field(
        ...,
        min_length=2,
        description="At least two specialists exhibiting conflicting signals.",
    )
    evidence: List[AggregatedEvidenceItem] = Field(
        default_factory=list,
        description="Attributed evidence items illustrating the conflict.",
    )

    @field_validator("topic", "description")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned


class CrossSpecialistObservation(BaseModel):
    """A cross-specialist insight connecting signals across distinct domains."""

    model_config = ConfigDict(extra="ignore")

    observation: str = Field(
        ...,
        description="Factual cross-domain observation connecting specialist signals.",
    )
    connected_specialists: List[SpecialistType] = Field(
        ...,
        min_length=1,
        description="Specialists contributing to this cross-domain connection.",
    )
    evidence: List[AggregatedEvidenceItem] = Field(
        default_factory=list,
        description="Attributed evidence items substantiating the observation.",
    )

    @field_validator("observation")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned


# ===========================================================================
# CONSISTENCY VALIDATION SCHEMAS (Phase 11.3)
# ===========================================================================


class ConsistencyCheckStatus(str, Enum):
    """Evaluation status of an individual consistency check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ConsistencyCheckSeverity(str, Enum):
    """Severity of a detected consistency condition."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConsistencyIssue(BaseModel):
    """An individual consistency check result or detected issue."""

    model_config = ConfigDict(extra="ignore")

    check_id: str = Field(
        ...,
        description="Unique check identifier, e.g. 'CHK_ATTR_SPECIALIST_EXISTS'.",
    )
    status: ConsistencyCheckStatus = Field(
        ...,
        description="Outcome of check: passed, warning, or failed.",
    )
    severity: ConsistencyCheckSeverity = Field(
        ...,
        description="Severity level of the issue.",
    )
    affected_specialists: List[str] = Field(
        default_factory=list,
        description="Specialist identifiers associated with this issue.",
    )
    affected_evidence_refs: List[str] = Field(
        default_factory=list,
        description="Evidence reference IDs associated with this issue.",
    )
    message: str = Field(
        ...,
        description="Concise description of the condition found.",
    )
    explanation: str = Field(
        ...,
        description=(
            "Detailed explanation covering what was checked, expected, and found."
        ),
    )

    @field_validator("check_id", "message", "explanation")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned


class AggregationConsistencyReport(BaseModel):
    """Structured consistency validation report for Report Aggregator (Phase 11.3)."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        description="Uppercase ticker symbol for the target company.",
    )
    is_valid: bool = Field(
        ...,
        description="True if no checks failed with high or critical severity.",
    )
    passed_checks_count: int = Field(
        default=0,
        ge=0,
        description="Count of checks that passed.",
    )
    warnings_count: int = Field(
        default=0,
        ge=0,
        description="Count of checks resulting in warnings.",
    )
    failures_count: int = Field(
        default=0,
        ge=0,
        description="Count of checks that failed.",
    )
    issues: List[ConsistencyIssue] = Field(
        default_factory=list,
        description="List of all evaluated check issues and passes.",
    )
    summary: str = Field(
        ...,
        description="Executive summary of the consistency validation results.",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if input had zero or insufficient specialist data.",
    )

    @field_validator("ticker", "summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned


# ===========================================================================
# SPECIALIST STATUS & PAYLOAD CONTAINERS (11.1.1 & 11.1.2)
# ===========================================================================


class SpecialistEntry(BaseModel):
    """Container representing a specialist's availability, status, and payload.

    Provides explicit distinction between:
    - AVAILABLE: Output is present and valid.
    - MISSING: Specialist was not run or not provided.
    - FAILED: Execution failed with an error message.
    - EMPTY: Specialist returned empty/insufficient content.
    """

    model_config = ConfigDict(extra="ignore")

    specialist: SpecialistType = Field(
        ...,
        description="Specialist identifier.",
    )
    status: SpecialistStatus = Field(
        ...,
        description="Explicit availability status (available, missing, failed, empty).",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error description if status is FAILED or EMPTY.",
    )
    raw_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw dictionary representation of specialist output if present.",
    )

    @property
    def is_available(self) -> bool:
        """True if specialist output is successfully available."""
        return self.status == SpecialistStatus.AVAILABLE

    @property
    def is_failed(self) -> bool:
        """True if specialist failed or errored."""
        return self.status == SpecialistStatus.FAILED

    @property
    def is_missing(self) -> bool:
        """True if specialist was omitted or skipped."""
        return self.status == SpecialistStatus.MISSING

    @property
    def is_empty(self) -> bool:
        """True if specialist output was empty or insufficient."""
        return self.status == SpecialistStatus.EMPTY


# ===========================================================================
# INVESTOR PROFILE SCHEMA (11.1.1)
# ===========================================================================


class AggregatorInvestorProfile(BaseModel):
    """Validated investor preferences and constraints for the Report Aggregator."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    target_company: Optional[str] = Field(
        default=None,
        description="Target company name from user context.",
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Associated uppercase ticker symbol if known.",
    )
    investment_goal: Optional[str] = Field(
        default=None,
        description="Primary goal (e.g. 'growth', 'capital preservation', 'income').",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Investment horizon (e.g. 'short term', '1 year', '5+ years').",
    )
    capital_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Planned investment capital amount.",
    )
    risk_tolerance: Optional[str] = Field(
        default=None,
        description="Risk tolerance (e.g. 'conservative', 'moderate', 'aggressive').",
    )
    profile_complete: Optional[bool] = Field(
        default=None,
        description="Whether investor profile was marked complete by clarification.",
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip().upper()
        return cleaned if cleaned else None

    @classmethod
    def from_investor_profile(
        cls, profile: Optional[Union[InvestorProfile, Dict[str, Any]]]
    ) -> Optional["AggregatorInvestorProfile"]:
        """Safely convert a GraphState InvestorProfile TypedDict or dict to model."""
        if not profile or not isinstance(profile, dict):
            return None

        capital = profile.get("capital_amount")
        if capital is not None:
            try:
                capital = float(capital)
                if capital < 0.0 or math.isnan(capital) or math.isinf(capital):
                    capital = None
            except (ValueError, TypeError):
                capital = None

        return cls(
            target_company=profile.get("target_company"),
            ticker=profile.get("ticker"),
            investment_goal=profile.get("investment_goal"),
            time_horizon=profile.get("time_horizon"),
            capital_amount=capital,
            risk_tolerance=profile.get("risk_tolerance"),
            profile_complete=profile.get("profile_complete"),
        )


# ===========================================================================
# REPORT AGGREGATOR INPUT CONTRACT (Goals 1-6, 11.1.1)
# ===========================================================================


class ReportAggregatorInput(BaseModel):
    """Structured input contract for the Report Aggregator (Phase 11.1).

    Fulfills Phase 11.1 requirements:
    - Receives specialist outputs independently (technical, fundamental, news, etc.).
    - Supports partial specialist availability (all, some, failed, or missing).
    - Preserves specialist attribution and evidence/provenance references.
    - Explicitly represents availability/failure status for each specialist.
    - Prevents accidental fabrication or defaulting of financial data.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        description="Normalized uppercase target company ticker symbol.",
    )
    target_company: Optional[str] = Field(
        default=None,
        description="Company name if available.",
    )
    investor_profile: Optional[AggregatorInvestorProfile] = Field(
        default=None,
        description="Clarified investor constraints and preferences.",
    )

    # Independent Specialist Output Containers
    technical: Optional[TechnicalAnalysisOutput] = Field(
        default=None,
        description="Validated structured output from Technical Analyst (Phase 7).",
    )
    fundamental: Optional[FundamentalAnalysisOutput] = Field(
        default=None,
        description="Validated structured output from Fundamental Analyst (Phase 6).",
    )
    news: Optional[NewsAnalysisOutput] = Field(
        default=None,
        description="Validated structured output from News Analyst (Phase 8).",
    )
    research: Optional[ResearchAnalysisOutput] = Field(
        default=None,
        description="Validated structured output from Research Analyst (Phase 9).",
    )
    risk: Optional[RiskAnalysisOutput] = Field(
        default=None,
        description="Validated structured output from Risk Analyst (Phase 10).",
    )

    # Explicit Specialist Status Tracking
    specialist_statuses: Dict[SpecialistType, SpecialistStatus] = Field(
        default_factory=dict,
        description="Explicit availability status for each specialist.",
    )
    specialist_errors: Dict[SpecialistType, str] = Field(
        default_factory=dict,
        description="Error messages for failed or rejected specialists.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return cleaned

    @model_validator(mode="after")
    def sync_specialist_statuses(self) -> "ReportAggregatorInput":
        """Synchronize specialist_statuses dictionary with provided outputs.

        Ensures every known specialist has an explicit status:
        - If model is present and not explicitly marked failed/empty -> AVAILABLE.
        - If not present and not in statuses -> MISSING.
        - If ticker mismatches between specialist and input -> raises validation error.
        """
        specialist_mapping: Dict[SpecialistType, Optional[Any]] = {
            "technical": self.technical,
            "fundamental": self.fundamental,
            "news": self.news,
            "research": self.research,
            "risk": self.risk,
        }

        for spec, output in specialist_mapping.items():
            if spec not in self.specialist_statuses:
                if output is not None:
                    self.specialist_statuses[spec] = SpecialistStatus.AVAILABLE
                else:
                    self.specialist_statuses[spec] = SpecialistStatus.MISSING
            elif self.specialist_statuses[spec] == SpecialistStatus.AVAILABLE:
                if output is None:
                    # Contradiction: marked available but no output provided
                    self.specialist_statuses[spec] = SpecialistStatus.MISSING

            # Cross-validate ticker consistency when output has a ticker
            if output is not None and hasattr(output, "ticker"):
                spec_ticker = getattr(output, "ticker")
                if spec_ticker and spec_ticker.strip().upper() != self.ticker:
                    raise ReportAggregatorValidationError(
                        f"Specialist '{spec}' ticker '{spec_ticker}' does not match "
                        f"aggregator input ticker '{self.ticker}'."
                    )

        return self

    # -----------------------------------------------------------------------
    # INTROSPECTION & AVAILABILITY PROPERTIES
    # -----------------------------------------------------------------------

    @property
    def available_specialists(self) -> List[SpecialistType]:
        """List of specialists whose outputs are successfully available."""
        return [
            spec
            for spec, status in self.specialist_statuses.items()
            if status == SpecialistStatus.AVAILABLE
        ]

    @property
    def missing_specialists(self) -> List[SpecialistType]:
        """List of specialists that are missing/omitted."""
        return [
            spec
            for spec, status in self.specialist_statuses.items()
            if status == SpecialistStatus.MISSING
        ]

    @property
    def failed_specialists(self) -> List[SpecialistType]:
        """List of specialists that failed execution."""
        return [
            spec
            for spec, status in self.specialist_statuses.items()
            if status == SpecialistStatus.FAILED
        ]

    @property
    def empty_specialists(self) -> List[SpecialistType]:
        """List of specialists that returned empty/insufficient results."""
        return [
            spec
            for spec, status in self.specialist_statuses.items()
            if status == SpecialistStatus.EMPTY
        ]

    @property
    def has_technical(self) -> bool:
        return (
            self.technical is not None
            and self.specialist_statuses.get("technical") == SpecialistStatus.AVAILABLE
        )

    @property
    def has_fundamental(self) -> bool:
        return (
            self.fundamental is not None
            and self.specialist_statuses.get("fundamental")
            == SpecialistStatus.AVAILABLE
        )

    @property
    def has_news(self) -> bool:
        return (
            self.news is not None
            and self.specialist_statuses.get("news") == SpecialistStatus.AVAILABLE
        )

    @property
    def has_research(self) -> bool:
        return (
            self.research is not None
            and self.specialist_statuses.get("research") == SpecialistStatus.AVAILABLE
        )

    @property
    def has_risk(self) -> bool:
        return (
            self.risk is not None
            and self.specialist_statuses.get("risk") == SpecialistStatus.AVAILABLE
        )

    @property
    def is_empty(self) -> bool:
        """True if NO specialist outputs are available."""
        return len(self.available_specialists) == 0

    @property
    def core_completeness_ratio(self) -> float:
        """Ratio of core specialists available (technical, fundamental, news, risk)."""
        present = sum(
            1
            for s in CORE_SPECIALISTS
            if self.specialist_statuses.get(s) == SpecialistStatus.AVAILABLE
        )
        return present / len(CORE_SPECIALISTS)

    @property
    def total_completeness_ratio(self) -> float:
        """Ratio of all specialists available (5 total)."""
        present = sum(
            1
            for s in ALL_SPECIALISTS
            if self.specialist_statuses.get(s) == SpecialistStatus.AVAILABLE
        )
        return present / len(ALL_SPECIALISTS)

    # -----------------------------------------------------------------------
    # EVIDENCE EXTRACTION WITH PRESERVED ATTRIBUTION (11.1.2)
    # -----------------------------------------------------------------------

    def extract_attributed_evidence(self) -> List[AggregatedEvidenceItem]:
        """Extract evidence items from specialists with strict attribution.

        Preserves specialist origin, identifiers, and underlying citation metadata.
        """
        evidence_items: List[AggregatedEvidenceItem] = []

        # 1. Technical Analyst Evidence
        if self.has_technical and self.technical:
            for idx, ev_str in enumerate(self.technical.evidence):
                try:
                    evidence_items.append(
                        AggregatedEvidenceItem.from_technical_evidence(ev_str, idx)
                    )
                except ValueError:
                    continue

        # 2. Fundamental Analyst Evidence
        if self.has_fundamental and self.fundamental:
            assessments = [
                ("financial_health", self.fundamental.financial_health),
                ("growth", self.fundamental.growth_assessment),
                ("profitability", self.fundamental.profitability_assessment),
                ("valuation", self.fundamental.valuation_assessment),
                ("leverage", self.fundamental.leverage_assessment),
                ("cash_flow", self.fundamental.cash_flow_assessment),
            ]
            for dim_name, assess in assessments:
                for metric_name in assess.supporting_metrics:
                    expl = f"{dim_name} ({assess.rating}): {assess.explanation}"
                    evidence_items.append(
                        AggregatedEvidenceItem.from_fundamental_metric(
                            metric_name=metric_name,
                            explanation=expl,
                        )
                    )

        # 3. News Analyst Evidence
        if self.has_news and self.news:
            for item in self.news.recent_news:
                evidence_items.append(
                    AggregatedEvidenceItem.from_news_recent_item(
                        article_id=item.article_id,
                        headline=item.headline,
                        source=item.source,
                    )
                )

        # 4. Research Analyst Evidence
        if self.has_research and self.research:
            for ref in self.research.evidence:
                evidence_items.append(
                    AggregatedEvidenceItem.from_research_evidence(ref)
                )

        # 5. Risk Analyst Evidence
        if self.has_risk and self.risk:
            all_risk_factors = (
                self.risk.market_risks
                + self.risk.company_risks
                + self.risk.sector_risks
                + self.risk.financial_risks
                + self.risk.volatility_risks
                + self.risk.investor_specific_risks
            )
            for rf in all_risk_factors:
                for ev_ref in rf.evidence:
                    evidence_items.append(
                        AggregatedEvidenceItem.from_risk_evidence(ev_ref)
                    )

        return evidence_items

    # -----------------------------------------------------------------------
    # FACTORY CONSTRUCTORS (Goal 3 & 11.1.2)
    # -----------------------------------------------------------------------

    @classmethod
    def from_graph_state(
        cls,
        state: Union[GraphState, Dict[str, Any]],
        ticker: Optional[str] = None,
    ) -> "ReportAggregatorInput":
        """Construct ReportAggregatorInput from LangGraph GraphState.

        Gracefully handles:
        - None or missing specialist outputs -> SpecialistStatus.MISSING
        - AgentResult with success=False -> SpecialistStatus.FAILED
        - AgentResult with empty payload -> SpecialistStatus.EMPTY
        - Schema parsing errors -> SpecialistStatus.FAILED with descriptive error
        - Investor profile conversion
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary or GraphState instance.")

        resolved_ticker = ticker
        if not resolved_ticker:
            prof = state.get("investor_profile")
            if isinstance(prof, dict) and prof.get("ticker"):
                resolved_ticker = prof["ticker"]

        if not resolved_ticker:
            cio = state.get("cio_decision")
            if isinstance(cio, dict) and cio.get("ticker"):
                resolved_ticker = cio["ticker"]

        statuses: Dict[SpecialistType, SpecialistStatus] = {}
        errors: Dict[SpecialistType, str] = {}

        # Helper to parse specialist output
        def _parse_specialist(
            spec: SpecialistType,
            raw_field: Optional[Any],
            model_cls: Any,
        ) -> Optional[Any]:
            if raw_field is None:
                statuses[spec] = SpecialistStatus.MISSING
                return None

            # Handle AgentResult instances
            if hasattr(raw_field, "success"):
                if not raw_field.success:
                    statuses[spec] = SpecialistStatus.FAILED
                    errors[spec] = raw_field.error or "Specialist execution failed."
                    return None
                payload = raw_field.data
            elif (
                isinstance(raw_field, dict)
                and "success" in raw_field
                and "data" in raw_field
            ):
                if not raw_field.get("success", False):
                    statuses[spec] = SpecialistStatus.FAILED
                    errors[spec] = (
                        raw_field.get("error") or "Specialist execution failed."
                    )
                    return None
                payload = raw_field.get("data")
            else:
                payload = raw_field

            if payload is None:
                statuses[spec] = SpecialistStatus.EMPTY
                errors[spec] = "Specialist output payload was empty."
                return None

            if isinstance(payload, model_cls):
                statuses[spec] = SpecialistStatus.AVAILABLE
                return payload

            if isinstance(payload, dict):
                # Check for empty dict
                if not payload:
                    statuses[spec] = SpecialistStatus.EMPTY
                    errors[spec] = "Specialist output dictionary was empty."
                    return None
                try:
                    parsed = model_cls.model_validate(payload)
                    statuses[spec] = SpecialistStatus.AVAILABLE
                    return parsed
                except Exception as exc:
                    statuses[spec] = SpecialistStatus.FAILED
                    errors[spec] = f"Validation failed for {spec}: {exc}"
                    return None

            statuses[spec] = SpecialistStatus.FAILED
            errors[spec] = f"Unsupported data type for {spec}: {type(payload).__name__}"
            return None

        tech = _parse_specialist(
            "technical", state.get("technical_result"), TechnicalAnalysisOutput
        )
        fund = _parse_specialist(
            "fundamental", state.get("fundamental_result"), FundamentalAnalysisOutput
        )
        news = _parse_specialist("news", state.get("news_result"), NewsAnalysisOutput)
        res = _parse_specialist(
            "research", state.get("research_result"), ResearchAnalysisOutput
        )
        risk = _parse_specialist("risk", state.get("risk_result"), RiskAnalysisOutput)

        # Fallback ticker from specialists if not yet found
        if not resolved_ticker:
            for item in [tech, fund, news, risk]:
                if item and hasattr(item, "ticker") and getattr(item, "ticker"):
                    resolved_ticker = getattr(item, "ticker")
                    break

        if not resolved_ticker:
            # Check user query as last resort or raise error
            raise ReportAggregatorValidationError(
                "Unable to determine target ticker from GraphState. "
                "Please supply ticker."
            )

        inv_profile = AggregatorInvestorProfile.from_investor_profile(
            state.get("investor_profile")
        )

        return cls(
            ticker=resolved_ticker,
            target_company=(inv_profile.target_company if inv_profile else None),
            investor_profile=inv_profile,
            technical=tech,
            fundamental=fund,
            news=news,
            research=res,
            risk=risk,
            specialist_statuses=statuses,
            specialist_errors=errors,
        )

    @classmethod
    def from_partial_specialists(
        cls,
        ticker: str,
        target_company: Optional[str] = None,
        investor_profile: Optional[
            Union[AggregatorInvestorProfile, Dict[str, Any]]
        ] = None,
        technical: Optional[Any] = None,
        fundamental: Optional[Any] = None,
        news: Optional[Any] = None,
        research: Optional[Any] = None,
        risk: Optional[Any] = None,
        specialist_statuses: Optional[Dict[SpecialistType, SpecialistStatus]] = None,
        specialist_errors: Optional[Dict[SpecialistType, str]] = None,
    ) -> "ReportAggregatorInput":
        """Explicit constructor for partial specialist inputs."""
        inv_prof = (
            investor_profile
            if isinstance(investor_profile, AggregatorInvestorProfile)
            else AggregatorInvestorProfile.from_investor_profile(investor_profile)
        )

        # Build mock state for uniform parsing
        state_dict: Dict[str, Any] = {
            "technical_result": technical,
            "fundamental_result": fundamental,
            "news_result": news,
            "research_result": research,
            "risk_result": risk,
        }

        agg_input = cls.from_graph_state(state=state_dict, ticker=ticker)
        if target_company:
            agg_input.target_company = target_company
        if inv_prof:
            agg_input.investor_profile = inv_prof

        # Override with explicit statuses or errors if passed
        if specialist_statuses:
            agg_input.specialist_statuses.update(specialist_statuses)
        if specialist_errors:
            agg_input.specialist_errors.update(specialist_errors)

        return agg_input


# ===========================================================================
# UNIFIED ANALYSIS OUTPUT SCHEMA (Phase 11.1.1 & 11.1.2)
# ===========================================================================


class UnifiedSpecialistAnalysis(BaseModel):
    """The unified analysis schema combining all specialist outputs + investor profile.

    Fulfills Phase 11.1.1:
    - Unified analysis schema combining all specialist outputs + investor profile.
    - Preserves per-specialist evidence/source references (Phase 11.1.2).
    - Tracks explicit per-specialist availability and completeness.
    - Strictly separates raw specialist results from the aggregated structure.
    - Adheres to safety rules (no buy/sell advice, price targets, guaranteed returns).
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        description="Normalized uppercase ticker symbol.",
    )
    target_company: Optional[str] = Field(
        default=None,
        description="Target company name.",
    )
    investor_profile: Optional[AggregatorInvestorProfile] = Field(
        default=None,
        description="Associated investor constraints and context.",
    )

    # Specialist Availability & Completeness Tracking
    specialist_statuses: Dict[SpecialistType, SpecialistStatus] = Field(
        default_factory=dict,
        description="Explicit availability status for each specialist.",
    )
    data_completeness_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of available core specialists.",
    )

    # Structured Specialist Assessment Blocks (Preserved independently)
    technical_assessment: Optional[TechnicalAnalysisOutput] = Field(
        default=None,
        description="Technical analysis output if available.",
    )
    fundamental_assessment: Optional[FundamentalAnalysisOutput] = Field(
        default=None,
        description="Fundamental analysis output if available.",
    )
    news_assessment: Optional[NewsAnalysisOutput] = Field(
        default=None,
        description="News analysis output if available.",
    )
    research_assessment: Optional[ResearchAnalysisOutput] = Field(
        default=None,
        description="Research document analysis output if available.",
    )
    risk_assessment: Optional[RiskAnalysisOutput] = Field(
        default=None,
        description="Risk assessment output if available.",
    )

    # Attributed Evidence Collection (11.1.2)
    aggregated_evidence: List[AggregatedEvidenceItem] = Field(
        default_factory=list,
        description="All extracted specialist evidence items preserving provenance.",
    )

    # Cross-Specialist Synthesis & Alignment (Phase 11.2)
    areas_of_agreement: List[SynthesisFinding] = Field(
        default_factory=list,
        description="Key findings corroborated across multiple specialists.",
    )
    signal_conflicts: List[SignalConflict] = Field(
        default_factory=list,
        description="Identified contradictions or tensions between specialists.",
    )
    cross_specialist_observations: List[CrossSpecialistObservation] = Field(
        default_factory=list,
        description="Factual cross-domain insights synthesized from specialist data.",
    )
    overall_synthesis: str = Field(
        default="",
        description=(
            "Unified analytical synthesis grounded strictly in supplied evidence."
        ),
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Overall synthesis confidence reflecting data completeness and "
            "consistency."
        ),
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if specialist evidence was insufficient for synthesis.",
    )
    insufficient_evidence_reason: Optional[str] = Field(
        default=None,
        description="Detailed explanation if insufficient_evidence is True.",
    )

    # Metadata & Error Records
    missing_specialists: List[SpecialistType] = Field(
        default_factory=list,
        description="List of specialists that were missing.",
    )
    failed_specialists: List[SpecialistType] = Field(
        default_factory=list,
        description="List of specialists that failed.",
    )
    specialist_errors: Dict[SpecialistType, str] = Field(
        default_factory=dict,
        description="Error messages for failed specialists.",
    )

    # Consistency Validation (Phase 11.3)
    consistency_report: Optional[AggregationConsistencyReport] = Field(
        default=None,
        description="Structured consistency validation report from Phase 11.3.",
    )

    @property
    def available_specialists(self) -> List[SpecialistType]:
        """List of specialists whose outputs are successfully available."""
        return [
            spec
            for spec, status in self.specialist_statuses.items()
            if status == SpecialistStatus.AVAILABLE
        ]

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return cleaned

    @field_validator("overall_synthesis")
    @classmethod
    def validate_safety_boundaries(cls, v: str) -> str:
        """Enforce strict prohibition on buy/sell advice and price targets."""
        for pattern in PROHIBITED_ADVICE_PATTERNS:
            if pattern.search(v):
                raise ReportAggregatorValidationError(
                    "Unified synthesis contains prohibited advisory phrase matching "
                    f"'{pattern.pattern}'. FinPilot provides decision support only."
                )
        return v

    @classmethod
    def from_input(
        cls,
        agg_input: ReportAggregatorInput,
    ) -> "UnifiedSpecialistAnalysis":
        """Initialize UnifiedSpecialistAnalysis directly from ReportAggregatorInput.

        Populates specialist blocks, status tracking, completeness, and evidence.
        """
        evidence_items = agg_input.extract_attributed_evidence()

        return cls(
            ticker=agg_input.ticker,
            target_company=agg_input.target_company,
            investor_profile=agg_input.investor_profile,
            specialist_statuses=dict(agg_input.specialist_statuses),
            data_completeness_ratio=agg_input.core_completeness_ratio,
            technical_assessment=(
                agg_input.technical if agg_input.has_technical else None
            ),
            fundamental_assessment=(
                agg_input.fundamental if agg_input.has_fundamental else None
            ),
            news_assessment=agg_input.news if agg_input.has_news else None,
            research_assessment=agg_input.research if agg_input.has_research else None,
            risk_assessment=agg_input.risk if agg_input.has_risk else None,
            aggregated_evidence=evidence_items,
            missing_specialists=agg_input.missing_specialists,
            failed_specialists=agg_input.failed_specialists,
            specialist_errors=dict(agg_input.specialist_errors),
        )


class AggregatorSynthesisOutput(BaseModel):
    """Structured LLM synthesis payload for Report Aggregator (Phase 11.2)."""

    model_config = ConfigDict(extra="ignore")

    areas_of_agreement: List[SynthesisFinding] = Field(
        default_factory=list,
        description="Key findings corroborated across multiple specialists.",
    )
    signal_conflicts: List[SignalConflict] = Field(
        default_factory=list,
        description="Identified contradictions or tensions between specialists.",
    )
    cross_specialist_observations: List[CrossSpecialistObservation] = Field(
        default_factory=list,
        description="Factual cross-domain insights connecting specialist findings.",
    )
    overall_synthesis: str = Field(
        ...,
        description=(
            "Unified analytical synthesis grounded strictly in supplied evidence."
        ),
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence reflecting signal completeness and consistency.",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if specialist evidence was insufficient for synthesis.",
    )
    insufficient_evidence_reason: Optional[str] = Field(
        default=None,
        description="Detailed reason if insufficient_evidence is True.",
    )

    @field_validator("overall_synthesis")
    @classmethod
    def validate_safety(cls, v: str) -> str:
        for pattern in PROHIBITED_ADVICE_PATTERNS:
            if pattern.search(v):
                raise ReportAggregatorValidationError(
                    "Synthesis contains prohibited advisory phrase matching "
                    f"'{pattern.pattern}'."
                )
        return v
