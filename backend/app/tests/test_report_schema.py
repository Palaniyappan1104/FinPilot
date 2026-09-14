"""Unit tests for Phase 12.1 — Report Generation Schema.

Tests cover:
- Valid construction of FinalReport with all 15 required fields from plan.md 12.1.1.
- Type coercion (company string, capital float/dict).
- Convenience properties (ticker, evidence, sources).
- Construction directly from Phase 11 UnifiedSpecialistAnalysis.
- Recommendation postures and profile suitability.
- Missing / failed specialist representation.
- Insufficient evidence handling.
- Prohibited advisory language rejection (buy/sell, price targets, guaranteed returns).
- Disclaimer presence and content validation.
- Serialization and deserialization (JSON round-trip).
- ReportGeneratorInput validation and GraphState integration.
"""

import json
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from app.agents.aggregator_schema import (
    AggregatedEvidenceItem,
    AggregatorInvestorProfile,
    CrossSpecialistObservation,
    SignalConflict,
    SpecialistStatus,
    SynthesisFinding,
    UnifiedSpecialistAnalysis,
)
from app.agents.fundamental_schema import (
    DimensionAssessment,
    FundamentalAnalysisOutput,
)
from app.agents.news_schema import (
    NewsAnalysisOutput,
    RecentNewsItem,
)
from app.agents.report_schema import (
    STANDARD_DISCLAIMER,
    FinalReport,
    FundamentalReportSection,
    InvestmentReport,
    NewsReportSection,
    OverallAssessmentSection,
    RecommendationStance,
    ReportCapitalInfo,
    ReportCompanyInfo,
    ReportGeneratorInput,
    ReportRecommendation,
    ReportValidationError,
    ResearchReportSection,
    RiskReportSection,
    TechnicalReportSection,
)
from app.agents.research_schema import (
    ResearchAnalysisOutput,
    ResearchEvidenceRef,
    ResearchFinding,
)
from app.agents.risk_schema import (
    RiskAnalysisOutput,
    RiskCategory,
    RiskEvidenceRef,
    RiskFactor,
    RiskSeverity,
)
from app.agents.state import GraphState
from app.agents.technical_schema import (
    SupportResistanceSummary,
    TechnicalAnalysisOutput,
    TechnicalIndicatorsSummary,
    TechnicalInterpretation,
)
from app.models.technical_metrics import (
    MACDMetrics,
    MovingAverageMetrics,
    RSIMetrics,
    VolumeMetrics,
)

# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture
def sample_company() -> ReportCompanyInfo:
    return ReportCompanyInfo(
        ticker="AAPL",
        name="Apple Inc.",
        currency="USD",
        sector="Technology",
        industry="Consumer Electronics",
    )


@pytest.fixture
def sample_investor_profile() -> AggregatorInvestorProfile:
    return AggregatorInvestorProfile(
        target_company="Apple Inc.",
        ticker="AAPL",
        investment_goal="long-term capital appreciation",
        time_horizon="3-5 years",
        capital_amount=50000.0,
        risk_tolerance="moderate",
        profile_complete=True,
    )


@pytest.fixture
def sample_technical_section() -> TechnicalReportSection:
    return TechnicalReportSection(
        summary="Bullish continuation with strong momentum above key moving averages.",
        trend="uptrend",
        momentum="RSI at 62.0 confirms healthy buying pressure.",
        support_levels=[180.0, 175.0],
        resistance_levels=[192.0, 198.0],
        technical_score=75.0,
        confidence=0.88,
        evidence_refs=["tech_ev_0", "tech_ev_1"],
    )


@pytest.fixture
def sample_fundamental_section() -> FundamentalReportSection:
    return FundamentalReportSection(
        summary=(
            "High operating margin and pristine balance sheet with disciplined "
            "capital return."
        ),
        overall_assessment="favorable",
        financial_health="strong",
        profitability="strong",
        valuation="neutral",
        growth="moderate",
        cash_flow="strong",
        key_strengths=["High operating margin", "Exceptional free cash flow"],
        key_weaknesses=["Hardware growth saturation"],
        confidence=0.92,
        evidence_refs=["pe_ratio", "gross_margin"],
    )


@pytest.fixture
def sample_news_section() -> NewsReportSection:
    return NewsReportSection(
        summary=(
            "Broadly positive sentiment driven by Services growth and AI "
            "announcements."
        ),
        overall_sentiment="positive",
        sentiment_score=0.65,
        key_themes=["Services expansion", "WWDC product roadmap"],
        recent_headlines=[
            "Apple expands AI capabilities",
            "Services revenue hits record",
        ],
        confidence=0.82,
        evidence_refs=["art_001"],
    )


@pytest.fixture
def sample_research_section() -> ResearchReportSection:
    return ResearchReportSection(
        summary=(
            "SEC 10-K confirms disciplined R&D spending and robust supply chain "
            "commitments."
        ),
        filing_type="10-K",
        key_findings=["R&D spending up 12% YoY", "Gross margin guidance maintained"],
        document_citations=["Apple 2024 Form 10-K (Item 7, p. 34)"],
        confidence=0.85,
        evidence_refs=["chunk_aapl_10k_01"],
    )


@pytest.fixture
def sample_risk_section() -> RiskReportSection:
    return RiskReportSection(
        summary=(
            "Overall risk is low-to-moderate, dominated by regulatory and "
            "antitrust scrutiny."
        ),
        overall_risk_level="low",
        quantitative_score=24.5,
        top_risk_factors=[
            "Antitrust regulatory litigation",
            "Consumer spending softness",
        ],
        confidence=0.89,
        evidence_refs=["risk_comp_01"],
    )


@pytest.fixture
def sample_overall_assessment() -> OverallAssessmentSection:
    return OverallAssessmentSection(
        synthesis=(
            "Apple demonstrates robust alignment across technical uptrend and strong "
            "cash flow fundamentals, moderated by high valuation multiples."
        ),
        confidence=0.87,
        data_completeness_ratio=1.0,
        areas_of_agreement=[
            SynthesisFinding(
                topic="Operational Cash Generation",
                summary="Strong cash flow provides financial resilience.",
                supporting_specialists=["fundamental", "risk"],
            )
        ],
        signal_conflicts=[],
        cross_specialist_observations=[
            CrossSpecialistObservation(
                observation="Services growth buffers consumer cycle softness.",
                connected_specialists=["fundamental", "news"],
            )
        ],
    )


@pytest.fixture
def sample_recommendation() -> ReportRecommendation:
    return ReportRecommendation(
        stance=RecommendationStance.FAVORABLE,
        rationale=(
            "Apple exhibits strong alignment with a 3-5 year investment horizon "
            "supported by industry-leading cash conversion and robust technical trends."
        ),
        profile_alignment=(
            "Well-suited for moderate risk tolerance and growth objectives."
        ),
        monitoring_points=[
            "Services segment gross margin stability",
            "Sustained price support above $180.0",
        ],
        time_horizon_suitability="Strong for multi-year holding period.",
        risk_tolerance_suitability="Appropriate for moderate risk tolerance.",
    )


@pytest.fixture
def sample_evidence_sources() -> list[AggregatedEvidenceItem]:
    return [
        AggregatedEvidenceItem(
            specialist="technical",
            reference_id="tech_ev_0",
            detail="Price 185.50 is above 50-day SMA 178.0",
        ),
        AggregatedEvidenceItem(
            specialist="fundamental",
            reference_id="gross_margin",
            detail="Gross margin expands to 44.0%",
        ),
    ]


@pytest.fixture
def sample_final_report(
    sample_company: ReportCompanyInfo,
    sample_investor_profile: AggregatorInvestorProfile,
    sample_technical_section: TechnicalReportSection,
    sample_fundamental_section: FundamentalReportSection,
    sample_news_section: NewsReportSection,
    sample_research_section: ResearchReportSection,
    sample_risk_section: RiskReportSection,
    sample_overall_assessment: OverallAssessmentSection,
    sample_recommendation: ReportRecommendation,
    sample_evidence_sources: list[AggregatedEvidenceItem],
) -> FinalReport:
    return FinalReport(
        company=sample_company,
        investor_profile=sample_investor_profile,
        horizon="3-5 years",
        capital=ReportCapitalInfo(amount=50000.0, currency="USD", formatted="$50,000"),
        technical=sample_technical_section,
        fundamental=sample_fundamental_section,
        news=sample_news_section,
        research=sample_research_section,
        risk=sample_risk_section,
        overall_assessment=sample_overall_assessment,
        recommendation=sample_recommendation,
        key_reasons=[
            "Pristine balance sheet with over $35B in free cash flow",
            "Technical trend maintains bullish momentum above ascending SMAs",
            "Services revenue expansion offsets hardware seasonality",
        ],
        important_risks=[
            "Antitrust regulatory challenges in App Store ecosystem",
            "Extended valuation multiples limit immediate price margin of safety",
        ],
        evidence_sources=sample_evidence_sources,
        disclaimer=STANDARD_DISCLAIMER,
        confidence=0.87,
        specialist_statuses={
            "technical": SpecialistStatus.AVAILABLE,
            "fundamental": SpecialistStatus.AVAILABLE,
            "news": SpecialistStatus.AVAILABLE,
            "research": SpecialistStatus.AVAILABLE,
            "risk": SpecialistStatus.AVAILABLE,
        },
    )


# ===========================================================================
# 1. VALID CONSTRUCTION & 15 REQUIRED FIELDS (12.1.1)
# ===========================================================================


class TestFinalReportSchemaCompleteness:
    """Validate all 15 required fields from plan.md 12.1.1 are present."""

    def test_all_15_fields_present_and_valid(
        self, sample_final_report: FinalReport
    ) -> None:
        r = sample_final_report

        # 1. company
        assert r.company.ticker == "AAPL"
        assert r.company.name == "Apple Inc."
        assert r.ticker == "AAPL"

        # 2. investor_profile
        assert r.investor_profile is not None
        assert r.investor_profile.risk_tolerance == "moderate"

        # 3. horizon
        assert r.horizon == "3-5 years"

        # 4. capital
        assert r.capital is not None
        assert r.capital.amount == 50000.0

        # 5. technical
        assert r.technical is not None
        assert r.technical.trend == "uptrend"

        # 6. fundamental
        assert r.fundamental is not None
        assert r.fundamental.overall_assessment == "favorable"

        # 7. news
        assert r.news is not None
        assert r.news.overall_sentiment == "positive"

        # 8. research
        assert r.research is not None
        assert r.research.filing_type == "10-K"

        # 9. risk
        assert r.risk is not None
        assert r.risk.overall_risk_level == "low"

        # 10. overall_assessment
        assert "Apple demonstrates" in r.overall_assessment.synthesis
        assert r.overall_assessment.data_completeness_ratio == 1.0

        # 11. recommendation
        assert r.recommendation.stance == RecommendationStance.FAVORABLE
        assert "Apple exhibits strong alignment" in r.recommendation.rationale

        # 12. key_reasons
        assert len(r.key_reasons) >= 1
        assert "free cash flow" in r.key_reasons[0]

        # 13. important_risks
        assert len(r.important_risks) >= 1
        assert "Antitrust" in r.important_risks[0]

        # 14. evidence/sources
        assert len(r.evidence_sources) == 2
        assert len(r.evidence) == 2
        assert len(r.sources) == 2

        # 15. disclaimer
        assert STANDARD_DISCLAIMER in r.disclaimer
        assert "informational purposes only" in r.disclaimer.lower()

    def test_alias_investment_report_is_identical(
        self, sample_final_report: FinalReport
    ) -> None:
        assert isinstance(sample_final_report, InvestmentReport)


# ===========================================================================
# 2. COERCION AND FLEXIBLE INPUTS
# ===========================================================================


class TestFieldCoercion:
    """Validate flexible input coercion for company and capital."""

    def test_company_string_coerced(
        self,
        sample_overall_assessment: OverallAssessmentSection,
        sample_recommendation: ReportRecommendation,
    ) -> None:
        report = FinalReport(
            company="MSFT",  # plain string
            capital=100000.0,  # plain float
            overall_assessment=sample_overall_assessment,
            recommendation=sample_recommendation,
            key_reasons=["Reason 1"],
            important_risks=["Risk 1"],
            disclaimer=STANDARD_DISCLAIMER,
        )
        assert report.company.ticker == "MSFT"
        assert report.ticker == "MSFT"
        assert report.capital is not None
        assert report.capital.amount == 100000.0

    def test_company_dict_coerced(
        self,
        sample_overall_assessment: OverallAssessmentSection,
        sample_recommendation: ReportRecommendation,
    ) -> None:
        report = FinalReport(
            company={"ticker": "nvda", "name": "NVIDIA Corp"},
            overall_assessment=sample_overall_assessment,
            recommendation=sample_recommendation,
            key_reasons=["Reason 1"],
            important_risks=["Risk 1"],
            disclaimer=STANDARD_DISCLAIMER,
        )
        assert report.company.ticker == "NVDA"
        assert report.company.name == "NVIDIA Corp"


# ===========================================================================
# 3. CONSTRUCTION FROM PHASE 11 AGGREGATOR
# ===========================================================================


class TestFactoryFromUnifiedAnalysis:
    """Validate direct construction from Phase 11 UnifiedSpecialistAnalysis."""

    @pytest.fixture
    def mock_unified_analysis(
        self,
        sample_investor_profile: AggregatorInvestorProfile,
    ) -> UnifiedSpecialistAnalysis:
        tech = TechnicalAnalysisOutput(
            ticker="AAPL",
            trend="uptrend",
            indicators_summary=TechnicalIndicatorsSummary(
                latest_close=185.50,
                moving_averages=MovingAverageMetrics(
                    sma_20=182.0,
                    sma_50=178.0,
                    sma_200=170.0,
                ),
                rsi=RSIMetrics(value=62.0),
                macd=MACDMetrics(macd_line=1.5, signal_line=1.0, histogram=0.5),
                volume=VolumeMetrics(
                    latest_volume=55000000.0, average_volume_20d=48000000.0
                ),
            ),
            support_resistance=SupportResistanceSummary(
                primary_support=180.0,
                primary_resistance=192.0,
                support_levels=[180.0, 175.0],
                resistance_levels=[192.0, 198.0],
            ),
            technical_score=75.0,
            interpretation=TechnicalInterpretation(
                overall_summary="Bullish continuation with strong momentum.",
                trend_analysis="Price holds firmly above ascending SMAs.",
                moving_averages_analysis="Bullish stacking order SMA 20 > 50 > 200.",
                momentum_analysis="RSI at 62.0 reflects robust buying pressure.",
                volume_analysis="Volume confirms breakout over previous consolidation.",
                support_resistance_analysis=(
                    "Primary floor at 180.0, key barrier at 192.0."
                ),
            ),
            evidence=["Price 185.50 is above 50-day SMA 178.0"],
            risks=["Approaching resistance at 192.0"],
            confidence=0.88,
        )
        fund = FundamentalAnalysisOutput(
            ticker="AAPL",
            company_name="Apple Inc.",
            financial_health=DimensionAssessment(
                rating="strong",
                explanation="Healthy cash reserves.",
                supporting_metrics=["current_ratio"],
            ),
            growth_assessment=DimensionAssessment(
                rating="moderate",
                explanation="Steady growth.",
                supporting_metrics=["revenue_growth"],
            ),
            profitability_assessment=DimensionAssessment(
                rating="strong",
                explanation="High gross margin.",
                supporting_metrics=["gross_margin"],
            ),
            valuation_assessment=DimensionAssessment(
                rating="neutral",
                explanation="Multiple at fair value.",
                supporting_metrics=["pe_ratio"],
            ),
            leverage_assessment=DimensionAssessment(
                rating="strong",
                explanation="Debt is controlled.",
                supporting_metrics=["debt_to_ebitda"],
            ),
            cash_flow_assessment=DimensionAssessment(
                rating="strong",
                explanation="Exceptional free cash flow.",
                supporting_metrics=["free_cash_flow"],
            ),
            key_strengths=["High operating margin", "Pristine balance sheet"],
            key_weaknesses=["Hardware saturation"],
            overall_assessment="favorable",
            overall_summary=(
                "Apple exhibits exceptional cash generation and financial health."
            ),
            confidence=0.92,
        )
        news = NewsAnalysisOutput(
            ticker="AAPL",
            company_name="Apple Inc.",
            overall_sentiment="positive",
            sentiment_score=0.6,
            summary="Positive news coverage regarding Services expansion.",
            overall_summary="Services expansion drives positive market sentiment.",
            recent_news=[
                RecentNewsItem(
                    article_id="art_1",
                    headline="Apple launches new Services features",
                    source="Reuters",
                    published_at="2024-01-14",
                    sentiment="positive",
                    sentiment_score=0.7,
                )
            ],
            confidence=0.85,
        )
        res_ev = ResearchEvidenceRef(
            document_id="doc_01",
            chunk_id="chunk_01",
            source_document="AAPL_10K.pdf",
            page_numbers=[12],
        )
        res = ResearchAnalysisOutput(
            query="Analyze Apple 10-K disclosures regarding Services segment.",
            answer=(
                "Continued growth in high-margin Services with strong gross margins."
            ),
            key_findings=[
                ResearchFinding(
                    claim="Continued growth in high-margin Services.",
                    evidence=[res_ev],
                )
            ],
            evidence=[res_ev],
            confidence=0.80,
        )
        risk = RiskAnalysisOutput(
            ticker="AAPL",
            overall_risk_level="low",
            company_risks=[
                RiskFactor(
                    category=RiskCategory.COMPANY,
                    name="Antitrust Litigation",
                    description="Scrutiny over App Store developer policies.",
                    severity=RiskSeverity.LOW,
                    evidence=[
                        RiskEvidenceRef(
                            source_type="news",
                            reference_id="art_1",
                            detail="Scrutiny cited in news reports",
                        )
                    ],
                )
            ],
            confidence=0.89,
        )

        return UnifiedSpecialistAnalysis(
            ticker="AAPL",
            target_company="Apple Inc.",
            investor_profile=sample_investor_profile,
            specialist_statuses={
                "technical": SpecialistStatus.AVAILABLE,
                "fundamental": SpecialistStatus.AVAILABLE,
                "news": SpecialistStatus.AVAILABLE,
                "research": SpecialistStatus.AVAILABLE,
                "risk": SpecialistStatus.AVAILABLE,
            },
            data_completeness_ratio=1.0,
            technical_assessment=tech,
            fundamental_assessment=fund,
            news_assessment=news,
            research_assessment=res,
            risk_assessment=risk,
            aggregated_evidence=[
                AggregatedEvidenceItem(
                    specialist="technical",
                    reference_id="tech_ev_0",
                    detail="Price 185.50 above 50-day SMA",
                ),
                AggregatedEvidenceItem(
                    specialist="fundamental",
                    reference_id="gross_margin",
                    detail="High gross margin",
                ),
            ],
            areas_of_agreement=[
                SynthesisFinding(
                    topic="Financial Strength",
                    summary="Exceptional cash flow and balance sheet quality.",
                    supporting_specialists=["fundamental", "risk"],
                )
            ],
            signal_conflicts=[],
            cross_specialist_observations=[],
            overall_synthesis=(
                "Apple exhibits strong analytical alignment across technical momentum "
                "and robust fundamental cash flow."
            ),
            confidence=0.89,
        )

    def test_construction_from_unified_analysis_does_not_invent_recommendation(
        self, mock_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Verify schema mapping copies data without inventing a recommendation."""
        report = FinalReport.from_unified_analysis(mock_unified_analysis)

        assert report.ticker == "AAPL"
        assert report.company.name == "Apple Inc."
        assert report.horizon == "3-5 years"
        assert report.capital is not None and report.capital.amount == 50000.0

        # Sections are faithfully preserved
        assert report.technical is not None
        assert report.technical.trend == "uptrend"
        assert report.fundamental is not None
        assert report.fundamental.overall_assessment == "favorable"
        assert report.news is not None
        assert report.news.overall_sentiment == "positive"
        assert report.research is not None
        assert report.risk is not None
        assert report.risk.overall_risk_level == "low"

        # Pure structural mapping: NO invented recommendation or synthesized reasons
        assert report.recommendation is None
        assert report.key_reasons == []
        assert report.important_risks == []
        assert report.is_complete is False

        # Evidence sources and disclaimer preserved
        assert len(report.evidence_sources) == 2
        assert STANDARD_DISCLAIMER in report.disclaimer

    def test_conflicts_preserved_without_converting_to_recommendation(
        self, mock_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Signal conflicts must be preserved, NOT converted to CAUTIOUS."""
        conflicted = mock_unified_analysis.model_copy(
            update={
                "signal_conflicts": [
                    SignalConflict(
                        topic="Technical Bullish vs Fundamental Weakness",
                        description=("Trend is upward but margins are deteriorating."),
                        specialist_positions={
                            "technical": "bullish",
                            "fundamental": "weak",
                        },
                        involved_specialists=["technical", "fundamental"],
                    )
                ]
            }
        )
        report = FinalReport.from_unified_analysis(conflicted)

        # Must NOT convert conflict to a recommendation stance
        assert report.recommendation is None
        assert report.key_reasons == []
        assert report.important_risks == []

        # Conflict is faithfully preserved in overall_assessment
        assert len(report.overall_assessment.signal_conflicts) == 1
        conflict = report.overall_assessment.signal_conflicts[0]
        assert conflict.topic == "Technical Bullish vs Fundamental Weakness"
        assert conflict.specialist_positions["technical"] == "bullish"

    def test_insufficient_evidence_represented_structurally_without_inference(
        self, mock_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Insufficient evidence preserved structurally without hidden inference."""
        insufficient = mock_unified_analysis.model_copy(
            update={
                "insufficient_evidence": True,
                "insufficient_evidence_reason": "Zero core specialists provided.",
                "data_completeness_ratio": 0.0,
                "confidence": 0.0,
            }
        )
        report = FinalReport.from_unified_analysis(insufficient)

        # Schema must NOT generate an INSUFFICIENT_EVIDENCE recommendation object
        assert report.recommendation is None
        assert report.key_reasons == []
        assert report.important_risks == []

        # Upstream status flags are faithfully preserved
        assert report.insufficient_evidence is True
        assert report.overall_assessment.insufficient_evidence is True
        assert (
            report.overall_assessment.insufficient_evidence_reason
            == "Zero core specialists provided."
        )
        assert report.overall_assessment.data_completeness_ratio == 0.0

    def test_upstream_recommendation_preserved_exactly_when_passed(
        self, mock_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """When recommendation is explicitly supplied, it is preserved exactly."""
        rec = ReportRecommendation(
            stance=RecommendationStance.FAVORABLE,
            rationale="Aligned signals across fundamentals and technical momentum.",
            profile_alignment="Supports 3-5 year growth horizon.",
            monitoring_points=["Upcoming earnings report", "50-day SMA support"],
        )
        reasons = ["Solid revenue expansion", "Robust liquidity cushion"]
        risks = ["Hardware market saturation", "Regulatory antitrust reviews"]

        report = FinalReport.from_unified_analysis(
            mock_unified_analysis,
            recommendation=rec,
            key_reasons=reasons,
            important_risks=risks,
        )

        assert report.recommendation is not None
        assert report.recommendation.stance == RecommendationStance.FAVORABLE
        assert report.recommendation.rationale == rec.rationale
        assert report.recommendation.profile_alignment == rec.profile_alignment
        assert report.key_reasons == reasons
        assert report.important_risks == risks
        assert report.is_complete is True

    def test_evidence_and_provenance_preserved_faithfully(
        self, mock_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Evidence provenance and specialist availability statuses are intact."""
        report = FinalReport.from_unified_analysis(mock_unified_analysis)

        assert len(report.evidence_sources) == len(
            mock_unified_analysis.aggregated_evidence
        )
        assert report.evidence_sources[0].reference_id == "tech_ev_0"
        assert report.evidence_sources[0].specialist == "technical"
        assert report.evidence_sources[1].reference_id == "gross_margin"
        assert report.evidence_sources[1].specialist == "fundamental"

        assert report.specialist_statuses["technical"] == SpecialistStatus.AVAILABLE
        assert report.specialist_statuses["fundamental"] == SpecialistStatus.AVAILABLE
        assert report.confidence == mock_unified_analysis.confidence


# ===========================================================================
# 4. REQUIRED FIELDS AND MALFORMED VALUES
# ===========================================================================


class TestRequiredFieldsAndMalformedValues:
    """Validate mandatory fields and schema constraints."""

    def test_missing_ticker_fails(
        self,
        sample_overall_assessment: OverallAssessmentSection,
        sample_recommendation: ReportRecommendation,
    ) -> None:
        with pytest.raises(ValidationError):
            FinalReport(
                company={"ticker": "  "},  # empty ticker
                overall_assessment=sample_overall_assessment,
                recommendation=sample_recommendation,
                key_reasons=["Reason 1"],
                important_risks=["Risk 1"],
                disclaimer=STANDARD_DISCLAIMER,
            )

    def test_missing_overall_assessment_fails(
        self,
        sample_company: ReportCompanyInfo,
    ) -> None:
        """FinalReport requires overall_assessment."""
        with pytest.raises(ValidationError):
            FinalReport(
                company=sample_company,
                disclaimer=STANDARD_DISCLAIMER,
            )

    def test_placeholder_recommendation_and_reasons_allowed(
        self,
        sample_company: ReportCompanyInfo,
        sample_overall_assessment: OverallAssessmentSection,
    ) -> None:
        """FinalReport allows recommendation to be unset for Phase 12.2 generator."""
        report = FinalReport(
            company=sample_company,
            overall_assessment=sample_overall_assessment,
            disclaimer=STANDARD_DISCLAIMER,
        )
        assert report.recommendation is None
        assert report.key_reasons == []
        assert report.important_risks == []
        assert report.is_complete is False

    def test_negative_capital_fails(self) -> None:
        with pytest.raises(ValidationError):
            ReportCapitalInfo(amount=-500.0)


# ===========================================================================
# 5. SAFETY & PROHIBITED ADVICE VALIDATION
# ===========================================================================


class TestSafetyAndProhibitedAdvice:
    """Verify strict prohibition on investment advice, price targets, and guarantees."""

    def test_prohibited_buy_recommendation_in_rationale_fails(
        self,
        sample_company: ReportCompanyInfo,
        sample_overall_assessment: OverallAssessmentSection,
    ) -> None:
        with pytest.raises((ReportValidationError, ValidationError)) as exc:
            ReportRecommendation(
                stance=RecommendationStance.FAVORABLE,
                rationale="We issue a strong buy recommendation for this stock.",
            )
        assert "prohibited" in str(exc.value).lower()

    def test_prohibited_price_target_in_synthesis_fails(
        self,
        sample_company: ReportCompanyInfo,
        sample_recommendation: ReportRecommendation,
    ) -> None:
        unsafe_overall = OverallAssessmentSection(
            synthesis="We establish a 12-month target price of $250.0 for Apple.",
        )
        with pytest.raises((ReportValidationError, ValidationError)) as exc:
            FinalReport(
                company=sample_company,
                overall_assessment=unsafe_overall,
                recommendation=sample_recommendation,
                key_reasons=["Reason 1"],
                important_risks=["Risk 1"],
                disclaimer=STANDARD_DISCLAIMER,
            )
        assert (
            "target price" in str(exc.value).lower()
            or "prohibited" in str(exc.value).lower()
        )

    def test_prohibited_guaranteed_return_in_key_reasons_fails(
        self,
        sample_company: ReportCompanyInfo,
        sample_overall_assessment: OverallAssessmentSection,
        sample_recommendation: ReportRecommendation,
    ) -> None:
        with pytest.raises((ReportValidationError, ValidationError)) as exc:
            FinalReport(
                company=sample_company,
                overall_assessment=sample_overall_assessment,
                recommendation=sample_recommendation,
                key_reasons=["Guaranteed returns of 25% expected over 1 year."],
                important_risks=["Risk 1"],
                disclaimer=STANDARD_DISCLAIMER,
            )
        assert (
            "guaranteed" in str(exc.value).lower()
            or "prohibited" in str(exc.value).lower()
        )

    def test_missing_or_blank_disclaimer_fails(
        self,
        sample_company: ReportCompanyInfo,
        sample_overall_assessment: OverallAssessmentSection,
        sample_recommendation: ReportRecommendation,
    ) -> None:
        with pytest.raises((ReportValidationError, ValidationError)) as exc:
            FinalReport(
                company=sample_company,
                overall_assessment=sample_overall_assessment,
                recommendation=sample_recommendation,
                key_reasons=["Reason 1"],
                important_risks=["Risk 1"],
                disclaimer="   ",  # blank
            )
        assert "disclaimer" in str(exc.value).lower()

    def test_invalid_disclaimer_missing_risk_disclosure_fails(
        self,
        sample_company: ReportCompanyInfo,
        sample_overall_assessment: OverallAssessmentSection,
        sample_recommendation: ReportRecommendation,
    ) -> None:
        with pytest.raises((ReportValidationError, ValidationError)) as exc:
            FinalReport(
                company=sample_company,
                overall_assessment=sample_overall_assessment,
                recommendation=sample_recommendation,
                key_reasons=["Reason 1"],
                important_risks=["Risk 1"],
                disclaimer="This report was produced by automated AI.",
            )
        assert "disclaimer" in str(exc.value).lower()


# ===========================================================================
# 6. SERIALIZATION & DESERIALIZATION
# ===========================================================================


class TestSerializationAndDeserialization:
    """Validate JSON round-trip compatibility."""

    def test_final_report_json_round_trip(
        self, sample_final_report: FinalReport
    ) -> None:
        # Dump to JSON
        json_str = sample_final_report.model_dump_json()
        assert isinstance(json_str, str)

        # Ensure valid JSON
        parsed_dict = json.loads(json_str)
        assert parsed_dict["company"]["ticker"] == "AAPL"
        assert parsed_dict["recommendation"]["stance"] == "favorable"
        assert len(parsed_dict["key_reasons"]) == 3

        # Validate back into model
        restored = FinalReport.model_validate_json(json_str)
        assert restored.ticker == sample_final_report.ticker
        assert restored.confidence == sample_final_report.confidence
        assert (
            restored.recommendation.stance == sample_final_report.recommendation.stance
        )
        assert len(restored.evidence_sources) == len(
            sample_final_report.evidence_sources
        )


# ===========================================================================
# 7. REPORT GENERATOR INPUT CONTRACT & GRAPH STATE INTEGRATION
# ===========================================================================


class TestReportGeneratorInput:
    """Validate ReportGeneratorInput contract and GraphState constructor."""

    def test_input_from_unified_analysis(
        self, sample_final_report: FinalReport
    ) -> None:
        # Construct a dummy UnifiedSpecialistAnalysis
        analysis = UnifiedSpecialistAnalysis(
            ticker="AAPL",
            target_company="Apple Inc.",
            data_completeness_ratio=1.0,
            overall_synthesis="Consistent analytical alignment across specialists.",
        )
        gen_input = ReportGeneratorInput.from_unified_analysis(analysis)
        assert gen_input.ticker == "AAPL"
        assert gen_input.target_company == "Apple Inc."
        assert gen_input.aggregated_analysis.ticker == "AAPL"

    def test_input_from_graph_state_dict(self) -> None:
        analysis_dict: Dict[str, Any] = {
            "ticker": "MSFT",
            "target_company": "Microsoft Corp",
            "data_completeness_ratio": 0.8,
            "overall_synthesis": "Strong cloud momentum and cash generation.",
        }
        state: GraphState = {
            "user_query": "Analyze MSFT",
            "aggregated_result": analysis_dict,
            "investor_profile": {
                "ticker": "MSFT",
                "time_horizon": "5 years",
                "capital_amount": 25000.0,
            },
        }
        gen_input = ReportGeneratorInput.from_graph_state(state)
        assert gen_input.ticker == "MSFT"
        assert gen_input.target_company == "Microsoft Corp"
        assert gen_input.investor_profile is not None
        assert gen_input.investor_profile.capital_amount == 25000.0

    def test_input_from_graph_state_missing_aggregated_result_fails(self) -> None:
        state: GraphState = {"user_query": "Analyze MSFT"}
        with pytest.raises(ReportValidationError) as exc:
            ReportGeneratorInput.from_graph_state(state)
        assert "aggregated_result" in str(exc.value).lower()
