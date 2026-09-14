"""Report Generator schemas and input/output contracts for FinPilot (Phase 12.1).

Phase 12.1 establishes:
- 12.1.1: Final report schema combining company, investor profile, horizon, capital,
  technical, fundamental, news, research, risk, overall assessment, recommendation,
  key reasons, important risks, evidence/sources, and disclaimer.

Key Architecture:
- `RecommendationStance`: Decision-support posture enum (favorable, cautious, neutral,
  unfavorable, insufficient_evidence).
- `ReportCompanyInfo`: Company identification (ticker, name, currency, sector).
- `ReportCapitalInfo`: Planned investment capital and currency details.
- Specialist section models: `TechnicalReportSection`, `FundamentalReportSection`,
  `NewsReportSection`, `ResearchReportSection`, `RiskReportSection`.
- `OverallAssessmentSection`: Holistic analytical synthesis combining specialist
  consensus, conflicts, cross-domain observations, and completeness ratio.
- `ReportRecommendation`: Decision-support recommendation with stance, rationale,
  profile alignment, and monitoring points (non-advisory).
- `FinalReport` (alias `InvestmentReport`): The complete structured final report schema
  fulfilling plan.md 12.1.1.
- `ReportGeneratorInput`: Input contract for Report Generator consuming Phase 11
  UnifiedSpecialistAnalysis.
- `ReportGeneratorError`, `ReportValidationError`: Typed exceptions.
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.aggregator_schema import (
    PROHIBITED_ADVICE_PATTERNS,
    AggregatedEvidenceItem,
    AggregatorInvestorProfile,
    CrossSpecialistObservation,
    SignalConflict,
    SpecialistStatus,
    SpecialistType,
    SynthesisFinding,
    UnifiedSpecialistAnalysis,
)
from app.agents.state import GraphState

# ===========================================================================
# CONSTANTS & STANDARD REGULATORY DISCLAIMER (12.1.1 & 12.2.3)
# ===========================================================================

STANDARD_DISCLAIMER: str = (
    "FinPilot provides automated financial research and decision support for "
    "informational purposes only. FinPilot is not a registered investment advisor, "
    "broker-dealer, or financial planner. This report does not constitute "
    "personalized investment advice, an endorsement, or a recommendation to buy, "
    "sell, or hold any security. Past performance does not guarantee future results. "
    "All investments carry risk of loss. Investors must conduct independent "
    "research and consult licensed financial professionals before making "
    "investment decisions."
)


# ===========================================================================
# ENUMS & STATUS CODES
# ===========================================================================


class RecommendationStance(str, Enum):
    """Decision-support posture for the final report recommendation.

    Represents analytical alignment relative to the investor profile without
    violating regulatory non-advisory boundaries.
    """

    FAVORABLE = "favorable"
    CAUTIOUS = "cautious"
    NEUTRAL = "neutral"
    UNFAVORABLE = "unfavorable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ===========================================================================
# TYPED EXCEPTIONS
# ===========================================================================


class ReportGeneratorError(Exception):
    """Base exception for Report Generator errors."""

    def __init__(self, message: str, code: str = "REPORT_GENERATOR_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class ReportValidationError(ReportGeneratorError, ValueError):
    """Raised when report schemas violate validation or safety rules."""

    def __init__(self, message: str, code: str = "REPORT_VALIDATION_ERROR") -> None:
        super().__init__(message=message, code=code)


# ===========================================================================
# METADATA & CONTEXT MODELS (12.1.1)
# ===========================================================================


class ReportCompanyInfo(BaseModel):
    """Company identification and classification context."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        description="Normalized uppercase ticker symbol.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Full company name if available.",
    )
    currency: Optional[str] = Field(
        default="USD",
        description="Primary reporting currency (e.g. USD, INR, EUR).",
    )
    sector: Optional[str] = Field(
        default=None,
        description="Broad industry sector if available.",
    )
    industry: Optional[str] = Field(
        default=None,
        description="Specific industry group if available.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return cleaned


class ReportCapitalInfo(BaseModel):
    """Investment capital details and formatting."""

    model_config = ConfigDict(extra="ignore")

    amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Numerical capital amount planned for investment.",
    )
    currency: str = Field(
        default="USD",
        description="Capital currency symbol or code.",
    )
    formatted: Optional[str] = Field(
        default=None,
        description="Human-readable formatted capital representation (e.g. '$50,000').",
    )

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (math.isnan(v) or math.isinf(v) or v < 0.0):
            raise ValueError("Capital amount must be a finite non-negative number.")
        return v


# ===========================================================================
# SPECIALIST REPORT SECTIONS (12.1.1)
# ===========================================================================


class TechnicalReportSection(BaseModel):
    """Technical analysis findings and key levels for the final report."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        ...,
        description="Synthesized summary of technical posture and price action.",
    )
    trend: Optional[str] = Field(
        default=None,
        description="Identified price trend (e.g. 'uptrend', 'downtrend', 'sideways').",
    )
    momentum: Optional[str] = Field(
        default=None,
        description="Momentum assessment (e.g. RSI interpretation, MACD status).",
    )
    support_levels: List[float] = Field(
        default_factory=list,
        description="Key support price levels identified from technical data.",
    )
    resistance_levels: List[float] = Field(
        default_factory=list,
        description="Key resistance price barriers identified from technical data.",
    )
    technical_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Normalized technical strength score (0-100) if computed.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for technical findings.",
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Attributed evidence references supporting technical claims.",
    )

    @field_validator("summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Section summary cannot be empty.")
        return cleaned


class FundamentalReportSection(BaseModel):
    """Fundamental analysis findings and metric dimensions for the final report."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        ...,
        description="Synthesized summary of fundamental health and valuation.",
    )
    overall_assessment: Optional[str] = Field(
        default=None,
        description="Overall fundamental rating (favorable, neutral, unfavorable).",
    )
    financial_health: Optional[str] = Field(
        default=None,
        description="Balance sheet strength and liquidity assessment.",
    )
    profitability: Optional[str] = Field(
        default=None,
        description="Margin profile and operational profitability assessment.",
    )
    valuation: Optional[str] = Field(
        default=None,
        description="Valuation multiples relative to historical and peer averages.",
    )
    growth: Optional[str] = Field(
        default=None,
        description="Revenue and earnings trajectory assessment.",
    )
    cash_flow: Optional[str] = Field(
        default=None,
        description="Free cash flow generation and conversion efficiency.",
    )
    key_strengths: List[str] = Field(
        default_factory=list,
        description="Highlighted fundamental competitive strengths.",
    )
    key_weaknesses: List[str] = Field(
        default_factory=list,
        description="Highlighted fundamental constraints or operational risks.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for fundamental findings.",
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Attributed evidence references supporting fundamental claims.",
    )

    @field_validator("summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Section summary cannot be empty.")
        return cleaned


class NewsReportSection(BaseModel):
    """News sentiment and market narrative context for the final report."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        ...,
        description="Synthesized summary of market narrative and recent news flow.",
    )
    overall_sentiment: Optional[str] = Field(
        default=None,
        description="Aggregate news sentiment (positive, neutral, negative).",
    )
    sentiment_score: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Quantified sentiment polarity score (-1.0 to 1.0).",
    )
    key_themes: List[str] = Field(
        default_factory=list,
        description="Primary news themes and narrative drivers.",
    )
    recent_headlines: List[str] = Field(
        default_factory=list,
        description="Representative headlines cited from source news items.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for news analysis.",
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Attributed evidence references supporting news observations.",
    )

    @field_validator("summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Section summary cannot be empty.")
        return cleaned


class ResearchReportSection(BaseModel):
    """Document research and SEC regulatory filing insights for the final report."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        ...,
        description="Synthesized summary of proprietary document research findings.",
    )
    filing_type: Optional[str] = Field(
        default=None,
        description="Source filing classification (e.g. '10-K', '10-Q', 'Uploaded').",
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Core qualitative insights extracted from document disclosures.",
    )
    document_citations: List[str] = Field(
        default_factory=list,
        description="Formal citations referencing document sections and page numbers.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for research findings.",
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Attributed evidence references supporting research claims.",
    )

    @field_validator("summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Section summary cannot be empty.")
        return cleaned


class RiskReportSection(BaseModel):
    """Risk evaluation and factor breakdown for the final report."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        ...,
        description="Synthesized summary of overall risk posture and sensitivities.",
    )
    overall_risk_level: Optional[str] = Field(
        default=None,
        description="Categorical risk rating (low, moderate, high, critical).",
    )
    quantitative_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Baseline quantitative risk score (0-100) if computable.",
    )
    top_risk_factors: List[str] = Field(
        default_factory=list,
        description="Primary risk factors across market and company domains.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for risk findings.",
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Attributed evidence references supporting risk claims.",
    )

    @field_validator("summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Section summary cannot be empty.")
        return cleaned


# ===========================================================================
# OVERALL ASSESSMENT & RECOMMENDATION MODELS (12.1.1)
# ===========================================================================


class OverallAssessmentSection(BaseModel):
    """Holistic analytical synthesis combining all specialist domains."""

    model_config = ConfigDict(extra="ignore")

    synthesis: str = Field(
        ...,
        description="Comprehensive narrative synthesis linking signals across domains.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall synthesis confidence reflecting data completeness.",
    )
    data_completeness_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of core specialists successfully incorporated.",
    )
    areas_of_agreement: List[SynthesisFinding] = Field(
        default_factory=list,
        description="Corroborated findings supported by multiple specialists.",
    )
    signal_conflicts: List[SignalConflict] = Field(
        default_factory=list,
        description="Unresolved contradictions or tensions between specialist signals.",
    )
    cross_specialist_observations: List[CrossSpecialistObservation] = Field(
        default_factory=list,
        description="Cross-domain connections derived from specialist data.",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if specialist inputs were insufficient for synthesis.",
    )
    insufficient_evidence_reason: Optional[str] = Field(
        default=None,
        description="Explanation if insufficient_evidence is True.",
    )

    @field_validator("synthesis")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Overall assessment synthesis cannot be empty.")
        return cleaned


class ReportRecommendation(BaseModel):
    """Decision-support recommendation aligned with investor constraints.

    Provides explainable decision support without single-word verdicts or
    prohibited advisory directives (plan.md 12.1.1, 12.2.2, 12.3.2).
    """

    model_config = ConfigDict(extra="ignore")

    stance: RecommendationStance = Field(
        ...,
        description="Analytical posture (favorable, cautious, neutral, etc.).",
    )
    rationale: str = Field(
        ...,
        description="Justification grounded strictly in aggregated evidence.",
    )
    profile_alignment: Optional[str] = Field(
        default=None,
        description=(
            "Explanation of how this posture aligns with the investor's time "
            "horizon, risk tolerance, and capital constraints."
        ),
    )
    monitoring_points: List[str] = Field(
        default_factory=list,
        description=(
            "Key operational, valuation, or technical milestones for the investor "
            "to monitor (e.g. upcoming earnings, margin thresholds, support levels)."
        ),
    )
    time_horizon_suitability: Optional[str] = Field(
        default=None,
        description="Evaluation of alignment with the requested holding period.",
    )
    risk_tolerance_suitability: Optional[str] = Field(
        default=None,
        description="Evaluation of alignment with stated risk tolerance.",
    )

    @field_validator("rationale")
    @classmethod
    def validate_non_empty_and_safe(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Recommendation rationale cannot be empty.")
        for pattern in PROHIBITED_ADVICE_PATTERNS:
            if pattern.search(cleaned):
                raise ReportValidationError(
                    "Recommendation rationale contains prohibited advisory phrase "
                    f"matching '{pattern.pattern}'."
                )
        return cleaned


# ===========================================================================
# FINAL REPORT SCHEMA (Phase 12.1.1)
# ===========================================================================


class FinalReport(BaseModel):
    """Final explainable investment research and decision-support report.

    Fulfills Phase 12.1 requirements (plan.md 12.1.1):
    1. company: Target company identification.
    2. investor_profile: Clarified investor profile and constraints.
    3. horizon: Stated investment time horizon.
    4. capital: Stated investment capital.
    5. technical: Technical analysis section.
    6. fundamental: Fundamental analysis section.
    7. news: News and sentiment section.
    8. research: Document research section.
    9. risk: Risk assessment section.
    10. overall_assessment: Unified overall assessment synthesis.
    11. recommendation: Decision-support recommendation with rationale.
    12. key_reasons: Core supporting reasons tied to evidence.
    13. important_risks: Top critical risks the investor must consider.
    14. evidence_sources: Full list of attributed evidence items with provenance.
    15. disclaimer: Mandatory non-advisory regulatory disclaimer.
    """

    model_config = ConfigDict(extra="ignore")

    # 1. Company Information
    company: ReportCompanyInfo = Field(
        ...,
        description="Target company profile and ticker symbol.",
    )

    # 2. Investor Profile
    investor_profile: Optional[AggregatorInvestorProfile] = Field(
        default=None,
        description="Associated investor constraints, goals, and risk profile.",
    )

    # 3. Horizon
    horizon: Optional[str] = Field(
        default=None,
        description="Investment time horizon (e.g. 'long-term', '3-5 years').",
    )

    # 4. Capital
    capital: Optional[ReportCapitalInfo] = Field(
        default=None,
        description="Planned investment capital and currency details.",
    )

    # 5-9. Specialist Analysis Sections
    technical: Optional[TechnicalReportSection] = Field(
        default=None,
        description="Technical analysis section if available.",
    )
    fundamental: Optional[FundamentalReportSection] = Field(
        default=None,
        description="Fundamental analysis section if available.",
    )
    news: Optional[NewsReportSection] = Field(
        default=None,
        description="News and sentiment section if available.",
    )
    research: Optional[ResearchReportSection] = Field(
        default=None,
        description="Document research section if available.",
    )
    risk: Optional[RiskReportSection] = Field(
        default=None,
        description="Risk assessment section if available.",
    )

    # 10. Overall Assessment
    overall_assessment: OverallAssessmentSection = Field(
        ...,
        description="Synthesized overall assessment across specialist domains.",
    )

    # 11. Recommendation
    recommendation: Optional[ReportRecommendation] = Field(
        default=None,
        description=(
            "Decision-support recommendation and profile alignment. "
            "Populated by Report Generator in Phase 12.2 or supplied from upstream."
        ),
    )

    # 12. Key Reasons
    key_reasons: List[str] = Field(
        default_factory=list,
        description=(
            "Specific analytical reasons directly supporting assessment. "
            "Populated by Report Generator in Phase 12.2 or supplied from upstream."
        ),
    )

    # 13. Important Risks
    important_risks: List[str] = Field(
        default_factory=list,
        description=(
            "Top critical risk factors the investor must monitor. "
            "Populated by Report Generator in Phase 12.2 or supplied from upstream."
        ),
    )

    # 14. Evidence Sources
    evidence_sources: List[AggregatedEvidenceItem] = Field(
        default_factory=list,
        description="Attributed evidence items preserving specialist provenance.",
    )

    # 15. Regulatory Disclaimer
    disclaimer: str = Field(
        default=STANDARD_DISCLAIMER,
        description="Mandatory non-advisory regulatory disclosure.",
    )

    # Report Metadata & Status Tracking
    report_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for report persistence and retrieval.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall report confidence score.",
    )
    specialist_statuses: Dict[SpecialistType, SpecialistStatus] = Field(
        default_factory=dict,
        description="Availability status for each specialist domain.",
    )
    missing_specialists: List[SpecialistType] = Field(
        default_factory=list,
        description="List of specialists that were missing or omitted.",
    )
    failed_specialists: List[SpecialistType] = Field(
        default_factory=list,
        description="List of specialists that failed execution.",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if report was generated under insufficient evidence.",
    )

    # -----------------------------------------------------------------------
    # CONVENIENCE ALIAS PROPERTIES
    # -----------------------------------------------------------------------

    @property
    def ticker(self) -> str:
        """Convenience accessor for target company ticker symbol."""
        return self.company.ticker

    @property
    def evidence(self) -> List[AggregatedEvidenceItem]:
        """Convenience alias for evidence_sources."""
        return self.evidence_sources

    @property
    def sources(self) -> List[AggregatedEvidenceItem]:
        """Convenience alias for evidence_sources."""
        return self.evidence_sources

    @property
    def is_complete(self) -> bool:
        """True if all report sections including recommendation are populated."""
        return (
            self.recommendation is not None
            and len(self.key_reasons) > 0
            and len(self.important_risks) > 0
        )

    # -----------------------------------------------------------------------
    # VALIDATORS
    # -----------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def coerce_company_and_capital(cls, data: Any) -> Any:
        """Allow passing string or dict for company and capital fields."""
        if not isinstance(data, dict):
            return data

        # Allow passing company as plain ticker string
        comp = data.get("company")
        if isinstance(comp, str):
            data["company"] = ReportCompanyInfo(ticker=comp)
        elif isinstance(comp, dict) and "ticker" in comp:
            data["company"] = ReportCompanyInfo.model_validate(comp)

        # Allow passing capital as plain number
        cap = data.get("capital")
        if isinstance(cap, (int, float)):
            data["capital"] = ReportCapitalInfo(amount=float(cap))
        elif isinstance(cap, dict):
            data["capital"] = ReportCapitalInfo.model_validate(cap)

        return data

    @model_validator(mode="after")
    def validate_safety_and_disclaimer(self) -> "FinalReport":
        """Verify regulatory disclaimer presence and enforce non-advisory rules."""
        # 1. Verify disclaimer presence
        if not self.disclaimer or not self.disclaimer.strip():
            raise ReportValidationError(
                "Final report must include a non-empty disclaimer."
            )
        disc_lower = self.disclaimer.lower()
        if not any(
            k in disc_lower
            for k in (
                "informational",
                "not a registered",
                "not constitute",
                "risk of loss",
            )
        ):
            raise ReportValidationError(
                "Disclaimer must contain standard risk and non-advisory disclosures."
            )

        # 2. Check for prohibited advisory phrases across text fields
        texts_to_scan: List[str] = [
            self.overall_assessment.synthesis,
        ]
        if self.recommendation:
            if self.recommendation.rationale:
                texts_to_scan.append(self.recommendation.rationale)
            if self.recommendation.profile_alignment:
                texts_to_scan.append(self.recommendation.profile_alignment)
        texts_to_scan.extend(self.key_reasons)
        texts_to_scan.extend(self.important_risks)

        if self.technical:
            texts_to_scan.append(self.technical.summary)
        if self.fundamental:
            texts_to_scan.append(self.fundamental.summary)
        if self.news:
            texts_to_scan.append(self.news.summary)
        if self.research:
            texts_to_scan.append(self.research.summary)
        if self.risk:
            texts_to_scan.append(self.risk.summary)

        for text in texts_to_scan:
            if not text:
                continue
            for pattern in PROHIBITED_ADVICE_PATTERNS:
                if pattern.search(text):
                    raise ReportValidationError(
                        "Report contains prohibited advisory phrase matching "
                        f"'{pattern.pattern}'. FinPilot provides research and "
                        "decision support only."
                    )

        return self

    # -----------------------------------------------------------------------
    # FACTORY CONSTRUCTOR FROM PHASE 11 AGGREGATOR
    # -----------------------------------------------------------------------

    @classmethod
    def from_unified_analysis(
        cls,
        analysis: UnifiedSpecialistAnalysis,
        recommendation: Optional[ReportRecommendation] = None,
        key_reasons: Optional[List[str]] = None,
        important_risks: Optional[List[str]] = None,
        disclaimer: str = STANDARD_DISCLAIMER,
        report_id: Optional[str] = None,
    ) -> "FinalReport":
        """Construct a FinalReport structurally from UnifiedSpecialistAnalysis.

        PERFORMS PURE STRUCTURAL MAPPING ONLY (No Decision-Making or Synthesis):
        - Faithfully copies upstream specialist sections and overall_assessment.
        - Preserves evidence/provenance and specialist statuses exactly.
        - Does NOT invent or infer a recommendation stance or rationale.
        - Does NOT convert signal conflicts into CAUTIOUS or any recommendation.
        - Does NOT convert low completeness into INSUFFICIENT_EVIDENCE.
        - Leaves recommendation, key_reasons, and risks unset unless supplied.
        - All analytical decision-making is reserved for Phase 12.2.
        """
        # 1. Company
        company = ReportCompanyInfo(
            ticker=analysis.ticker,
            name=analysis.target_company,
        )

        # 2. Investor profile context
        inv_prof = analysis.investor_profile
        horizon = inv_prof.time_horizon if inv_prof else None
        capital = (
            ReportCapitalInfo(amount=inv_prof.capital_amount)
            if inv_prof and inv_prof.capital_amount is not None
            else None
        )

        # 3. Technical section
        tech_sec: Optional[TechnicalReportSection] = None
        if analysis.technical_assessment:
            t = analysis.technical_assessment
            tech_sec = TechnicalReportSection(
                summary=(
                    t.interpretation.overall_summary
                    if t.interpretation
                    else "Technical data evaluated."
                ),
                trend=t.trend,
                momentum=(
                    t.interpretation.momentum_analysis if t.interpretation else None
                ),
                support_levels=(
                    list(t.support_resistance.support_levels)
                    if t.support_resistance
                    else []
                ),
                resistance_levels=(
                    list(t.support_resistance.resistance_levels)
                    if t.support_resistance
                    else []
                ),
                technical_score=t.technical_score,
                confidence=t.confidence,
                evidence_refs=[
                    e.reference_id
                    for e in analysis.aggregated_evidence
                    if e.specialist == "technical"
                ],
            )

        # 4. Fundamental section
        fund_sec: Optional[FundamentalReportSection] = None
        if analysis.fundamental_assessment:
            f = analysis.fundamental_assessment
            fund_sec = FundamentalReportSection(
                summary=(
                    f.overall_summary
                    if f.overall_summary
                    else "Fundamental data evaluated."
                ),
                overall_assessment=f.overall_assessment,
                financial_health=(
                    f.financial_health.rating if f.financial_health else None
                ),
                profitability=(
                    f.profitability_assessment.rating
                    if f.profitability_assessment
                    else None
                ),
                valuation=(
                    f.valuation_assessment.rating if f.valuation_assessment else None
                ),
                growth=f.growth_assessment.rating if f.growth_assessment else None,
                cash_flow=(
                    f.cash_flow_assessment.rating if f.cash_flow_assessment else None
                ),
                key_strengths=list(f.key_strengths),
                key_weaknesses=list(f.key_weaknesses),
                confidence=f.confidence,
                evidence_refs=[
                    e.reference_id
                    for e in analysis.aggregated_evidence
                    if e.specialist == "fundamental"
                ],
            )

        # 5. News section
        news_sec: Optional[NewsReportSection] = None
        if analysis.news_assessment:
            n = analysis.news_assessment
            news_sec = NewsReportSection(
                summary=(
                    getattr(n, "summary", None)
                    or getattr(n, "overall_summary", "News flow evaluated.")
                ),
                overall_sentiment=getattr(n, "overall_sentiment", None),
                sentiment_score=getattr(n, "sentiment_score", None),
                key_themes=[
                    item.headline
                    for item in getattr(n, "recent_news", [])[:3]
                    if getattr(item, "headline", None)
                ],
                recent_headlines=[
                    item.headline
                    for item in getattr(n, "recent_news", [])
                    if getattr(item, "headline", None)
                ],
                confidence=getattr(n, "confidence", 0.0),
                evidence_refs=[
                    e.reference_id
                    for e in analysis.aggregated_evidence
                    if e.specialist == "news"
                ],
            )

        # 6. Research section
        res_sec: Optional[ResearchReportSection] = None
        if analysis.research_assessment:
            r = analysis.research_assessment
            findings_list: List[str] = []
            if hasattr(r, "key_findings") and r.key_findings:
                findings_list = [
                    f.claim if hasattr(f, "claim") else str(f) for f in r.key_findings
                ]
            elif hasattr(r, "findings") and r.findings:
                findings_list = [str(f) for f in r.findings]

            summary_text = getattr(r, "answer", None) or getattr(
                r, "summary", "Document disclosures evaluated."
            )

            res_sec = ResearchReportSection(
                summary=summary_text,
                filing_type=getattr(r, "filing_type", None),
                key_findings=findings_list,
                document_citations=[
                    getattr(ref, "source_document", str(ref))
                    for ref in getattr(r, "evidence", [])
                ],
                confidence=getattr(r, "confidence", 0.0),
                evidence_refs=[
                    e.reference_id
                    for e in analysis.aggregated_evidence
                    if e.specialist == "research"
                ],
            )

        # 7. Risk section
        risk_sec: Optional[RiskReportSection] = None
        if analysis.risk_assessment:
            rk = analysis.risk_assessment
            all_factors = (
                getattr(rk, "market_risks", [])
                + getattr(rk, "company_risks", [])
                + getattr(rk, "sector_risks", [])
                + getattr(rk, "financial_risks", [])
                + getattr(rk, "volatility_risks", [])
                + getattr(rk, "investor_specific_risks", [])
            )
            risk_summary = getattr(rk, "summary", "") or getattr(
                rk, "overall_summary", ""
            )
            if not risk_summary.strip():
                risk_summary = "Risk assessment evaluated."

            risk_level_str = None
            if getattr(rk, "overall_risk_level", None):
                risk_level_str = (
                    rk.overall_risk_level.value
                    if hasattr(rk.overall_risk_level, "value")
                    else str(rk.overall_risk_level)
                )

            quant_score = getattr(rk, "deterministic_risk_score", None)
            if quant_score is None and getattr(rk, "deterministic_score", None):
                quant_score = getattr(rk.deterministic_score, "score", None)

            risk_sec = RiskReportSection(
                summary=risk_summary,
                overall_risk_level=risk_level_str,
                quantitative_score=quant_score,
                top_risk_factors=[
                    rf.name for rf in all_factors[:5] if getattr(rf, "name", None)
                ],
                confidence=getattr(rk, "confidence", 0.0),
                evidence_refs=[
                    e.reference_id
                    for e in analysis.aggregated_evidence
                    if e.specialist == "risk"
                ],
            )

        # 8. Overall assessment section (faithful structural copy from Phase 11)
        overall_sec = OverallAssessmentSection(
            synthesis=analysis.overall_synthesis,
            confidence=analysis.confidence,
            data_completeness_ratio=analysis.data_completeness_ratio,
            areas_of_agreement=list(analysis.areas_of_agreement),
            signal_conflicts=list(analysis.signal_conflicts),
            cross_specialist_observations=list(analysis.cross_specialist_observations),
            insufficient_evidence=analysis.insufficient_evidence,
            insufficient_evidence_reason=analysis.insufficient_evidence_reason,
        )

        # 9. Recommendation, Key Reasons, Important Risks:
        # PURE STRUCTURAL PRESERVATION: do NOT invent analytical conclusions.
        # If passed by caller/upstream, preserve them; otherwise leave unset.
        return cls(
            company=company,
            investor_profile=inv_prof,
            horizon=horizon,
            capital=capital,
            technical=tech_sec,
            fundamental=fund_sec,
            news=news_sec,
            research=res_sec,
            risk=risk_sec,
            overall_assessment=overall_sec,
            recommendation=recommendation,
            key_reasons=list(key_reasons) if key_reasons is not None else [],
            important_risks=(
                list(important_risks) if important_risks is not None else []
            ),
            evidence_sources=list(analysis.aggregated_evidence),
            disclaimer=disclaimer,
            report_id=report_id,
            confidence=analysis.confidence,
            specialist_statuses=dict(analysis.specialist_statuses),
            missing_specialists=list(analysis.missing_specialists),
            failed_specialists=list(analysis.failed_specialists),
            insufficient_evidence=analysis.insufficient_evidence,
        )


# Alias for flexible naming
InvestmentReport = FinalReport


# ===========================================================================
# REPORT GENERATOR INPUT CONTRACT (Phase 12.1)
# ===========================================================================


class ReportGeneratorInput(BaseModel):
    """Input contract for the Report Generator agent (Phase 12.1).

    Receives the unified specialist analysis and context from Phase 11,
    providing structured access to all upstream evidence and investor parameters.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        description="Target ticker symbol.",
    )
    target_company: Optional[str] = Field(
        default=None,
        description="Target company name if available.",
    )
    investor_profile: Optional[AggregatorInvestorProfile] = Field(
        default=None,
        description="Clarified investor profile context.",
    )
    aggregated_analysis: UnifiedSpecialistAnalysis = Field(
        ...,
        description="Unified specialist analysis output from Phase 11.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return cleaned

    @classmethod
    def from_unified_analysis(
        cls,
        analysis: UnifiedSpecialistAnalysis,
    ) -> "ReportGeneratorInput":
        """Construct ReportGeneratorInput directly from UnifiedSpecialistAnalysis."""
        return cls(
            ticker=analysis.ticker,
            target_company=analysis.target_company,
            investor_profile=analysis.investor_profile,
            aggregated_analysis=analysis,
        )

    @classmethod
    def from_graph_state(
        cls,
        state: Union[GraphState, Dict[str, Any]],
        ticker: Optional[str] = None,
    ) -> "ReportGeneratorInput":
        """Construct ReportGeneratorInput from LangGraph GraphState.

        Consumes state['aggregated_result'], converting dict or model to
        UnifiedSpecialistAnalysis.
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary or GraphState instance.")

        agg_raw = state.get("aggregated_result")
        if not agg_raw:
            raise ReportValidationError(
                "GraphState does not contain 'aggregated_result'. "
                "Report Aggregator (Phase 11) must run before Report Generator."
            )

        if isinstance(agg_raw, UnifiedSpecialistAnalysis):
            analysis = agg_raw
        elif isinstance(agg_raw, dict):
            # Check if wrapped in AgentResult format: {"success": True, "data": {...}}
            if "data" in agg_raw and isinstance(
                agg_raw["data"], (dict, UnifiedSpecialistAnalysis)
            ):
                payload = agg_raw["data"]
                if isinstance(payload, UnifiedSpecialistAnalysis):
                    analysis = payload
                else:
                    analysis = UnifiedSpecialistAnalysis.model_validate(payload)
            else:
                analysis = UnifiedSpecialistAnalysis.model_validate(agg_raw)
        else:
            raise TypeError(
                f"Unsupported aggregated_result type: {type(agg_raw)}. "
                "Expected UnifiedSpecialistAnalysis or dict."
            )

        resolved_ticker = ticker or analysis.ticker
        inv_prof = (
            AggregatorInvestorProfile.from_investor_profile(
                state.get("investor_profile")
            )
            or analysis.investor_profile
        )

        return cls(
            ticker=resolved_ticker,
            target_company=analysis.target_company,
            investor_profile=inv_prof,
            aggregated_analysis=analysis,
        )
