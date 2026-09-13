"""Pydantic schemas and input aggregation interfaces for the Risk Analyst (Phase 10.1).

Phase 10.1 defines:
- Upstream output signals feeding the Risk Analyst (Technical, Fundamental, News,
  optional Research Vault findings, and Investor Profile).
- Lightweight input aggregation interface (`RiskAnalystInput`) that consumes partial
  results when some specialists were skipped or failed.
- Normalized domain risk models: `RiskCategory`, `RiskSeverity`, `RiskProbability`,
  `RiskEvidenceRef`, and `RiskFactor`.
- Strongly typed `RiskInvestorProfile` model separate from generic company risk.
- Foundation `RiskAnalysisOutput` with strict grounding and safety validation.
- Typed exception `RiskAnalysisValidationError`.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.state import GraphState, InvestorProfile

# ===========================================================================
# ENUMS & CONSTANTS
# ===========================================================================


class RiskCategory(str, Enum):
    """Categorical classification of risk factors in FinPilot."""

    MARKET = "market"
    COMPANY = "company"
    SECTOR = "sector"
    FINANCIAL = "financial"
    VOLATILITY = "volatility"
    INVESTOR_SPECIFIC = "investor_specific"


class RiskSeverity(str, Enum):
    """Severity / impact level of an identified risk factor."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskProbability(str, Enum):
    """Estimated probability / likelihood of risk occurrence."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNCERTAIN = "uncertain"


# Source specialist identifiers
RiskSourceType = Literal[
    "technical",
    "fundamental",
    "news",
    "research",
    "investor_profile",
    "market_data",
]

# Prohibited advisory patterns strictly disallowed in risk outputs
PROHIBITED_ADVICE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:buy|sell|strong\s+buy|strong\s+sell)\s+recommendation\b", re.I),
    re.compile(r"\b(?:target\s+price|price\s+target)\b", re.I),
    re.compile(r"\bguaranteed\s+(?:return|profit|gain)s?\b", re.I),
    re.compile(r"\brisk[- ]free\b", re.I),
    re.compile(r"\b(?:investors?\s+must|you\s+should)\s+(?:buy|sell)\b", re.I),
]


# ===========================================================================
# TYPED EXCEPTIONS
# ===========================================================================


class RiskAnalystError(Exception):
    """Base exception for Risk Analyst Agent errors."""

    def __init__(self, message: str, code: str = "RISK_ANALYST_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class RiskAnalysisValidationError(RiskAnalystError):
    """Raised when risk analysis inputs or outputs violate schema/safety rules."""

    def __init__(self, message: str, code: str = "RISK_VALIDATION_ERROR") -> None:
        super().__init__(message=message, code=code)


# ===========================================================================
# EVIDENCE & RISK FACTOR SCHEMAS
# ===========================================================================


class RiskEvidenceRef(BaseModel):
    """Traceable provenance link mapping a risk finding to upstream specialist data."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    source_type: RiskSourceType = Field(
        ...,
        description="Upstream specialist source supplying this evidence.",
    )
    reference_id: str = Field(
        ...,
        description="Specific metric name, article ID, chunk ID, or profile field.",
    )
    detail: str = Field(
        ...,
        description="Factual observation, quantitative metric, or excerpt.",
    )

    @field_validator("reference_id", "detail")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned


class RiskFactor(BaseModel):
    """Structured risk factor backed by upstream factual evidence."""

    model_config = ConfigDict(extra="ignore")

    category: RiskCategory = Field(
        ...,
        description="Risk category classification.",
    )
    name: str = Field(
        ...,
        description="Concise descriptive title of the risk factor.",
    )
    description: str = Field(
        ...,
        description="Detailed explanation grounded strictly in supplied data.",
    )
    severity: RiskSeverity = Field(
        ...,
        description="Impact severity level: low, moderate, high, or critical.",
    )
    probability: Optional[RiskProbability] = Field(
        default=None,
        description="Likelihood of occurrence if assessable from data.",
    )
    evidence: List[RiskEvidenceRef] = Field(
        default_factory=list,
        description="Upstream evidence items supporting this risk factor.",
    )
    insufficient_data: bool = Field(
        default=False,
        description="True if category cannot be evaluated due to missing input.",
    )

    @field_validator("name", "description")
    @classmethod
    def validate_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Text fields cannot be empty or whitespace.")
        return cleaned


# ===========================================================================
# INVESTOR PROFILE SCHEMA (10.1.1)
# ===========================================================================


class RiskInvestorProfile(BaseModel):
    """Investor profile capturing user investment preferences and constraints."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    time_horizon: Optional[str] = Field(
        default=None,
        description="Investment horizon (e.g. 'short term', '1 year', '5+ years').",
    )
    risk_tolerance: Optional[str] = Field(
        default=None,
        description="Risk tolerance (e.g. 'conservative', 'moderate', 'aggressive').",
    )
    investment_goal: Optional[str] = Field(
        default=None,
        description="Primary goal (e.g. 'capital preservation', 'growth', 'income').",
    )
    capital_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Planned investment capital amount, if provided.",
    )
    target_company: Optional[str] = Field(
        default=None,
        description="Target company name from user context.",
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Associated ticker symbol if known.",
    )
    profile_complete: Optional[bool] = Field(
        default=None,
        description="Whether the profile information is complete.",
    )

    @classmethod
    def from_investor_profile(
        cls, profile: Optional[Union[InvestorProfile, Dict[str, Any]]]
    ) -> Optional["RiskInvestorProfile"]:
        """Safely convert a GraphState InvestorProfile TypedDict or dict to model."""
        if not profile or not isinstance(profile, dict):
            return None

        # Extract values cleanly
        capital = profile.get("capital_amount")
        if capital is not None:
            try:
                capital = float(capital)
            except (ValueError, TypeError):
                capital = None

        return cls(
            time_horizon=profile.get("time_horizon"),
            risk_tolerance=profile.get("risk_tolerance"),
            investment_goal=profile.get("investment_goal"),
            capital_amount=capital,
            target_company=profile.get("target_company"),
            ticker=profile.get("ticker"),
            profile_complete=profile.get("profile_complete"),
        )


# ===========================================================================
# UPSTREAM SIGNAL WRAPPERS (10.1.1)
# ===========================================================================


class TechnicalRiskSignals(BaseModel):
    """Normalized risk signals extracted from Technical Analyst output."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Target ticker symbol.")
    trend: Optional[str] = Field(default=None, description="Trend direction.")
    rsi: Optional[float] = Field(default=None, description="Wilder RSI value.")
    technical_score: Optional[float] = Field(
        default=None, description="Composite score."
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Technical risk factors identified by Technical Analyst.",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Factual indicator observations.",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Upstream confidence score."
    )

    @classmethod
    def from_output(cls, data: Any) -> Optional["TechnicalRiskSignals"]:
        """Safely parse TechnicalAnalysisOutput, dict, or AgentResult."""
        payload = _extract_payload(data)
        if not payload or not isinstance(payload, dict):
            return None

        ticker = payload.get("ticker")
        if not ticker or not isinstance(ticker, str):
            return None

        rsi_val = None
        ind_sum = payload.get("indicators_summary")
        if isinstance(ind_sum, dict):
            rsi_obj = ind_sum.get("rsi")
            if isinstance(rsi_obj, dict):
                rsi_val = rsi_obj.get("value")
            elif hasattr(rsi_obj, "value"):
                rsi_val = rsi_obj.value
            elif isinstance(rsi_obj, (int, float)):
                rsi_val = float(rsi_obj)
        elif hasattr(ind_sum, "rsi"):
            rsi_obj = getattr(ind_sum, "rsi")
            if hasattr(rsi_obj, "value"):
                rsi_val = rsi_obj.value
            elif isinstance(rsi_obj, (int, float)):
                rsi_val = float(rsi_obj)

        if rsi_val is None and "rsi" in payload:
            raw_rsi = payload["rsi"]
            if isinstance(raw_rsi, (int, float)):
                rsi_val = float(raw_rsi)
            elif isinstance(raw_rsi, dict):
                rsi_val = raw_rsi.get("value")
            elif hasattr(raw_rsi, "value"):
                rsi_val = raw_rsi.value

        return cls(
            ticker=ticker.strip().upper(),
            trend=str(payload.get("trend")) if payload.get("trend") else None,
            rsi=rsi_val,
            technical_score=payload.get("technical_score"),
            risks=list(payload.get("risks", [])),
            evidence=list(payload.get("evidence", [])),
            confidence=float(payload.get("confidence", 1.0)),
        )


class FundamentalRiskSignals(BaseModel):
    """Normalized risk signals extracted from Fundamental Analyst output."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Target ticker symbol.")
    financial_health_rating: Optional[str] = Field(
        default=None, description="Balance sheet health rating."
    )
    leverage_rating: Optional[str] = Field(default=None, description="Leverage rating.")
    cash_flow_rating: Optional[str] = Field(
        default=None, description="Cash flow rating."
    )
    key_weaknesses: List[str] = Field(
        default_factory=list,
        description="Fundamental weaknesses identified by Fundamental Analyst.",
    )
    notable_flags: List[str] = Field(
        default_factory=list,
        description="Notable risk/event flags.",
    )
    overall_assessment: Optional[str] = Field(
        default=None, description="Overall fundamental assessment."
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Upstream confidence score."
    )

    @classmethod
    def from_output(cls, data: Any) -> Optional["FundamentalRiskSignals"]:
        """Safely parse FundamentalAnalysisOutput, dict, or AgentResult."""
        payload = _extract_payload(data)
        if not payload or not isinstance(payload, dict):
            return None

        ticker = payload.get("ticker")
        if not ticker or not isinstance(ticker, str):
            return None

        def _get_rating(dim_key: str) -> Optional[str]:
            dim = payload.get(dim_key)
            if isinstance(dim, dict):
                return dim.get("rating")
            if hasattr(dim, "rating"):
                return getattr(dim, "rating")
            return None

        return cls(
            ticker=ticker.strip().upper(),
            financial_health_rating=_get_rating("financial_health"),
            leverage_rating=_get_rating("leverage_assessment"),
            cash_flow_rating=_get_rating("cash_flow_assessment"),
            key_weaknesses=list(payload.get("key_weaknesses", [])),
            notable_flags=list(payload.get("notable_flags", [])),
            overall_assessment=payload.get("overall_assessment"),
            confidence=float(payload.get("confidence", 1.0)),
        )


class NewsRiskSignals(BaseModel):
    """Normalized risk signals extracted from News Analyst output."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Target ticker symbol.")
    overall_sentiment: Optional[str] = Field(
        default=None, description="Synthesized news sentiment."
    )
    negative_factors: List[str] = Field(
        default_factory=list,
        description="Negative narrative factors from news articles.",
    )
    important_events: List[str] = Field(
        default_factory=list,
        description="Identified material financial events.",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Upstream confidence score."
    )

    @classmethod
    def from_output(cls, data: Any) -> Optional["NewsRiskSignals"]:
        """Safely parse NewsAnalysisOutput, dict, or AgentResult."""
        payload = _extract_payload(data)
        if not payload or not isinstance(payload, dict):
            return None

        ticker = payload.get("ticker")
        if not ticker or not isinstance(ticker, str):
            return None

        neg_factors: List[str] = []
        raw_neg = payload.get("negative_factors", [])
        for item in raw_neg:
            if isinstance(item, dict) and "text" in item:
                neg_factors.append(str(item["text"]))
            elif hasattr(item, "text"):
                neg_factors.append(str(item.text))
            elif isinstance(item, str):
                neg_factors.append(item)

        events: List[str] = []
        raw_events = payload.get("important_events", [])
        for item in raw_events:
            if isinstance(item, dict) and "description" in item:
                events.append(str(item["description"]))
            elif hasattr(item, "description"):
                events.append(str(item.description))
            elif isinstance(item, str):
                events.append(item)

        return cls(
            ticker=ticker.strip().upper(),
            overall_sentiment=payload.get("overall_sentiment"),
            negative_factors=neg_factors,
            important_events=events,
            confidence=float(payload.get("confidence", 1.0)),
        )


class ResearchRiskSignals(BaseModel):
    """Normalized risk signals extracted from Research Analyst output (optional)."""

    model_config = ConfigDict(extra="ignore")

    findings: List[str] = Field(
        default_factory=list,
        description="Document-grounded factual findings.",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Research confidence score."
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="Whether research retrieval lacked sufficient evidence.",
    )

    @classmethod
    def from_output(cls, data: Any) -> Optional["ResearchRiskSignals"]:
        """Safely parse ResearchAnalysisOutput, dict, or AgentResult."""
        payload = _extract_payload(data)
        if not payload or not isinstance(payload, dict):
            return None

        findings: List[str] = []
        raw_findings = payload.get("key_findings", [])
        for item in raw_findings:
            if isinstance(item, dict) and "claim" in item:
                findings.append(str(item["claim"]))
            elif hasattr(item, "claim"):
                findings.append(str(item.claim))
            elif isinstance(item, str):
                findings.append(item)

        return cls(
            findings=findings,
            confidence=float(payload.get("confidence", 1.0)),
            insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
        )


# ===========================================================================
# RISK INPUT AGGREGATION INTERFACE (10.1.1 & 10.1.2)
# ===========================================================================


class RiskAnalystInput(BaseModel):
    """Primary aggregated input payload for the Risk Analyst Agent.

    Fulfills Phase 10.1 requirements:
    - Defines upstream outputs that feed the Risk Analyst (10.1.1).
    - Provides a lightweight interface to consume partial results when
      specialists were skipped, timed out, or failed (10.1.2).
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        description="Normalized uppercase company ticker symbol.",
    )
    investor_profile: Optional[RiskInvestorProfile] = Field(
        default=None,
        description="Investor constraints (time horizon, risk tolerance, etc.).",
    )
    technical_signals: Optional[TechnicalRiskSignals] = Field(
        default=None,
        description="Signals from Technical Analyst (Phase 7).",
    )
    fundamental_signals: Optional[FundamentalRiskSignals] = Field(
        default=None,
        description="Signals from Fundamental Analyst (Phase 6).",
    )
    news_signals: Optional[NewsRiskSignals] = Field(
        default=None,
        description="Signals from News Analyst (Phase 8).",
    )
    research_signals: Optional[ResearchRiskSignals] = Field(
        default=None,
        description="Signals from Research Analyst / Research Vault (Phase 9).",
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Optional guidance passed from CIO routing.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty or whitespace.")
        return cleaned

    # -----------------------------------------------------------------------
    # INTROSPECTION & PARTIAL RESULT METADATA (10.1.2)
    # -----------------------------------------------------------------------

    @property
    def available_sources(self) -> List[str]:
        """Return list of specialist sources that provided valid data."""
        sources: List[str] = []
        if self.technical_signals is not None:
            sources.append("technical")
        if self.fundamental_signals is not None:
            sources.append("fundamental")
        if self.news_signals is not None:
            sources.append("news")
        if self.research_signals is not None:
            sources.append("research")
        if self.investor_profile is not None:
            sources.append("investor_profile")
        return sources

    @property
    def missing_sources(self) -> List[str]:
        """Return list of primary specialist sources that are absent."""
        primary = ["technical", "fundamental", "news"]
        return [src for src in primary if src not in self.available_sources]

    @property
    def has_technical_signals(self) -> bool:
        """Whether technical signals are present."""
        return self.technical_signals is not None

    @property
    def has_fundamental_signals(self) -> bool:
        """Whether fundamental signals are present."""
        return self.fundamental_signals is not None

    @property
    def has_news_signals(self) -> bool:
        """Whether news signals are present."""
        return self.news_signals is not None

    @property
    def has_research_signals(self) -> bool:
        """Whether research signals are present."""
        return self.research_signals is not None

    @property
    def has_investor_profile(self) -> bool:
        """Whether an investor profile is present."""
        return self.investor_profile is not None

    @property
    def is_empty(self) -> bool:
        """True if no upstream specialist signals are available."""
        return (
            self.technical_signals is None
            and self.fundamental_signals is None
            and self.news_signals is None
            and self.research_signals is None
        )

    @property
    def data_completeness_ratio(self) -> float:
        """Ratio of available primary sources (technical, fundamental, news)."""
        primary = ["technical", "fundamental", "news"]
        present = sum(1 for src in primary if src in self.available_sources)
        return present / len(primary)

    # -----------------------------------------------------------------------
    # FACTORY CONSTRUCTORS (10.1.2)
    # -----------------------------------------------------------------------

    @classmethod
    def from_graph_state(
        cls,
        state: Union[GraphState, Dict[str, Any]],
        ticker: Optional[str] = None,
    ) -> "RiskAnalystInput":
        """Construct RiskAnalystInput directly from LangGraph GraphState.

        Gracefully consumes partial results:
        - If a specialist output is None (skipped), it is recorded as None.
        - If a specialist execution failed (success=False), it is safely skipped.
        - Ticker is inferred from state (clarified_request, investor_profile,
          or specialist outputs) if not explicitly passed.
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary or GraphState instance.")

        # Resolve ticker
        resolved_ticker = ticker
        if not resolved_ticker:
            # Try investor_profile
            prof = state.get("investor_profile")
            if isinstance(prof, dict) and prof.get("ticker"):
                resolved_ticker = prof["ticker"]

        if not resolved_ticker:
            # Try cio_decision
            cio = state.get("cio_decision")
            if isinstance(cio, dict) and cio.get("ticker"):
                resolved_ticker = cio["ticker"]

        # Extract specialist outputs
        tech_sig = TechnicalRiskSignals.from_output(state.get("technical_result"))
        fund_sig = FundamentalRiskSignals.from_output(state.get("fundamental_result"))
        news_sig = NewsRiskSignals.from_output(state.get("news_result"))
        res_sig = ResearchRiskSignals.from_output(state.get("research_result"))

        # Fallback ticker from specialist signals if still None
        if not resolved_ticker:
            if tech_sig:
                resolved_ticker = tech_sig.ticker
            elif fund_sig:
                resolved_ticker = fund_sig.ticker
            elif news_sig:
                resolved_ticker = news_sig.ticker

        if not resolved_ticker:
            raise ValueError(
                "Unable to resolve ticker from GraphState. Please provide ticker."
            )

        inv_profile = RiskInvestorProfile.from_investor_profile(
            state.get("investor_profile")
        )

        return cls(
            ticker=resolved_ticker,
            investor_profile=inv_profile,
            technical_signals=tech_sig,
            fundamental_signals=fund_sig,
            news_signals=news_sig,
            research_signals=res_sig,
        )

    @classmethod
    def from_partial_results(
        cls,
        ticker: str,
        investor_profile: Optional[Union[RiskInvestorProfile, Dict[str, Any]]] = None,
        technical_output: Optional[Any] = None,
        fundamental_output: Optional[Any] = None,
        news_output: Optional[Any] = None,
        research_output: Optional[Any] = None,
        task_description: Optional[str] = None,
    ) -> "RiskAnalystInput":
        """Constructor accepting raw outputs from any subset of specialists."""
        inv_prof = (
            investor_profile
            if isinstance(investor_profile, RiskInvestorProfile)
            else RiskInvestorProfile.from_investor_profile(investor_profile)
        )

        return cls(
            ticker=ticker,
            investor_profile=inv_prof,
            technical_signals=TechnicalRiskSignals.from_output(technical_output),
            fundamental_signals=FundamentalRiskSignals.from_output(fundamental_output),
            news_signals=NewsRiskSignals.from_output(news_output),
            research_signals=ResearchRiskSignals.from_output(research_output),
            task_description=task_description,
        )


# ===========================================================================
# RISK ANALYST OUTPUT FOUNDATION SCHEMA
# ===========================================================================


class RiskAnalysisOutput(BaseModel):
    """Structured risk assessment produced by Risk Analyst Agent.

    Provides domain structure across all standard risk categories,
    preserves data completeness tracking, and enforces safety boundaries.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        description="Normalized uppercase target ticker symbol.",
    )
    overall_risk_level: Optional[RiskSeverity] = Field(
        default=None,
        description="Composite risk level: low, moderate, high, or critical.",
    )
    market_risks: List[RiskFactor] = Field(
        default_factory=list,
        description="Market-wide risks (macro, rates, equity volatility).",
    )
    company_risks: List[RiskFactor] = Field(
        default_factory=list,
        description="Company-specific operational, management, or competitive risks.",
    )
    sector_risks: List[RiskFactor] = Field(
        default_factory=list,
        description="Industry or sector-wide headwinds.",
    )
    financial_risks: List[RiskFactor] = Field(
        default_factory=list,
        description="Solvency, leverage, liquidity, or debt maturity risks.",
    )
    volatility_risks: List[RiskFactor] = Field(
        default_factory=list,
        description="Price swings, momentum exhaustion, or technical volatility risks.",
    )
    investor_specific_risks: List[RiskFactor] = Field(
        default_factory=list,
        description="Profile mismatch risks (e.g. short horizon vs volatile asset).",
    )
    data_completeness: Dict[str, bool] = Field(
        default_factory=dict,
        description="Specialist data availability map (technical, fundamental, news).",
    )
    insufficient_data: bool = Field(
        default=False,
        description="True if upstream data was insufficient for reliable analysis.",
    )
    insufficient_data_reason: Optional[str] = Field(
        default=None,
        description="Detailed explanation if analysis could not proceed.",
    )
    summary: str = Field(
        default="",
        description="Grounded qualitative narrative synthesizing risk findings.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Analytical confidence score reflecting data completeness.",
    )
    deterministic_risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pre-computed quantitative risk score (0.0=lowest, 1.0=critical).",
    )
    deterministic_risk_level: Optional[RiskSeverity] = Field(
        default=None,
        description="Deterministic risk severity mapped from quantitative score.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty or whitespace.")
        return cleaned

    @field_validator("summary")
    @classmethod
    def validate_summary_safety(cls, v: str) -> str:
        """Enforce strict prohibition on buy/sell advice and price targets."""
        for pattern in PROHIBITED_ADVICE_PATTERNS:
            if pattern.search(v):
                raise RiskAnalysisValidationError(
                    "Risk analysis contains prohibited advisory phrase matching "
                    f"'{pattern.pattern}'. FinPilot provides decision support only."
                )
        return v


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================


def _extract_payload(data: Any) -> Optional[Dict[str, Any]]:
    """Extract dictionary payload from Pydantic model, AgentResult, or dict."""
    if data is None:
        return None

    # Handle AgentResult instances
    if hasattr(data, "success") and hasattr(data, "data"):
        if not data.success:
            return None  # Failed specialist execution
        data = data.data

    # Handle Pydantic models
    if hasattr(data, "model_dump"):
        return data.model_dump()

    # Handle dicts
    if isinstance(data, dict):
        # Check if dict represents an AgentResult model_dump
        if "success" in data and "data" in data:
            if not data.get("success", False):
                return None
            inner = data.get("data")
            if isinstance(inner, dict):
                return inner
            if hasattr(inner, "model_dump"):
                return inner.model_dump()
            return None
        return data

    return None
