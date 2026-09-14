"""Focused unit and integration tests for Report Generator Agent (Phase 12.2).

Covers all 15 Phase 12.2 requirements:
1. Complete aggregated input -> valid report generation.
2. Partial specialist availability.
3. Missing specialist.
4. Failed specialist.
5. Insufficient evidence.
6. Recommendation grounding.
7. Preservation of evidence/source references.
8. No fabricated financial values.
9. No fabricated source references.
10. Prohibited guarantee/certainty language.
11. Malformed/invalid LLM output.
12. LLM/provider failure handling.
13. Investor profile preservation.
14. Horizon/capital preservation.
15. Report completeness/status handling.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.agents.aggregator_schema import (
    AggregatedEvidenceItem,
    AggregatorInvestorProfile,
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
    FactorItem,
    NewsAnalysisEvent,
    NewsAnalysisOutput,
    RecentNewsItem,
)
from app.agents.report_generator import (
    GeneratedReportSynthesis,
    ReportGeneratorAgent,
    generate_deterministic_report,
    report_generator_node,
    validate_report_grounding,
)
from app.agents.report_prompt import format_report_generator_prompt
from app.agents.report_schema import (
    STANDARD_DISCLAIMER,
    FinalReport,
    RecommendationStance,
    ReportValidationError,
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
from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import LLMError, LLMStructuredOutputError
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
def sample_technical() -> TechnicalAnalysisOutput:
    return TechnicalAnalysisOutput(
        ticker="AAPL",
        trend="uptrend",
        indicators_summary=TechnicalIndicatorsSummary(
            latest_close=185.50,
            moving_averages=MovingAverageMetrics(
                sma_20=182.0,
                sma_50=178.0,
                sma_200=165.0,
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
            support_resistance_analysis="Support at 180.0, resistance at 192.0.",
        ),
        evidence=["Price 185.50 is above 50-day SMA 178.0"],
        risks=["Approaching resistance at 192.0"],
        confidence=0.88,
    )


@pytest.fixture
def sample_fundamental() -> FundamentalAnalysisOutput:
    return FundamentalAnalysisOutput(
        ticker="AAPL",
        company_name="Apple Inc.",
        currency="USD",
        overall_assessment="favorable",
        overall_summary=(
            "Apple exhibits strong cash generation and operating efficiency."
        ),
        financial_health=DimensionAssessment(
            rating="strong",
            score=0.90,
            explanation="Pristine balance sheet with robust net cash position.",
            supporting_metrics=["net_cash", "interest_coverage"],
        ),
        profitability_assessment=DimensionAssessment(
            rating="strong",
            score=0.92,
            explanation="Gross margin remains at 45.0% with strong operating leverage.",
            supporting_metrics=["gross_margin", "operating_margin"],
        ),
        valuation_assessment=DimensionAssessment(
            rating="neutral",
            score=0.60,
            explanation="Forward P/E of 28.5 is in line with historical premium.",
            supporting_metrics=["pe_ratio"],
        ),
        growth_assessment=DimensionAssessment(
            rating="moderate",
            score=0.70,
            explanation="Services revenue expands while hardware remains steady.",
            supporting_metrics=["services_growth"],
        ),
        leverage_assessment=DimensionAssessment(
            rating="strong",
            score=0.95,
            explanation="Low net debt and high interest coverage.",
            supporting_metrics=["net_debt", "debt_to_equity"],
        ),
        cash_flow_assessment=DimensionAssessment(
            rating="strong",
            score=0.95,
            explanation="Free cash flow exceeding 100 billion dollars annually.",
            supporting_metrics=["free_cash_flow"],
        ),
        key_strengths=["High operating margin", "Robust balance sheet"],
        key_weaknesses=["Hardware saturation"],
        confidence=0.92,
    )


@pytest.fixture
def sample_news() -> NewsAnalysisOutput:
    return NewsAnalysisOutput(
        ticker="AAPL",
        overall_sentiment="positive",
        sentiment_distribution={"positive": 8, "neutral": 2, "negative": 0},
        recent_news=[
            RecentNewsItem(
                article_id="art_1",
                headline="Apple Expands Services Ecosystem with AI Integration",
                summary="Paid services subscriptions saw significant expansion.",
                source="Reuters",
                url="https://example.com/art1",
                published_at=datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc),
                sentiment="positive",
            )
        ],
        important_events=[
            NewsAnalysisEvent(
                event_type="earnings",
                description="Quarterly financial results release.",
                article_ids=["art_1"],
            )
        ],
        positive_factors=[
            FactorItem(
                text="Services Growth",
                article_ids=["art_1"],
            )
        ],
        negative_factors=[],
        summary="Positive news coverage regarding ecosystem expansion.",
        confidence=0.85,
    )


@pytest.fixture
def sample_research() -> ResearchAnalysisOutput:
    res_ev = ResearchEvidenceRef(
        document_id="doc_01",
        chunk_id="chunk_01",
        source_document="AAPL_10K.pdf",
        page_numbers=[12],
    )
    return ResearchAnalysisOutput(
        query="Analyze Apple 10-K disclosures regarding Services segment.",
        answer="Continued growth in high-margin Services with strong gross margins.",
        key_findings=[
            ResearchFinding(
                claim="Continued growth in high-margin Services.",
                evidence=[res_ev],
            )
        ],
        evidence=[res_ev],
        confidence=0.80,
    )


@pytest.fixture
def sample_risk() -> RiskAnalysisOutput:
    return RiskAnalysisOutput(
        ticker="AAPL",
        overall_risk_level=RiskSeverity.LOW,
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
        market_risks=[
            RiskFactor(
                category=RiskCategory.MARKET,
                name="Macro Volatility",
                description="Interest rate sensitivity and FX currency fluctuations.",
                severity=RiskSeverity.LOW,
            )
        ],
        confidence=0.89,
    )


@pytest.fixture
def sample_unified_analysis(
    sample_investor_profile: AggregatorInvestorProfile,
    sample_technical: TechnicalAnalysisOutput,
    sample_fundamental: FundamentalAnalysisOutput,
    sample_news: NewsAnalysisOutput,
    sample_research: ResearchAnalysisOutput,
    sample_risk: RiskAnalysisOutput,
) -> UnifiedSpecialistAnalysis:
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
        technical_assessment=sample_technical,
        fundamental_assessment=sample_fundamental,
        news_assessment=sample_news,
        research_assessment=sample_research,
        risk_assessment=sample_risk,
        aggregated_evidence=[
            AggregatedEvidenceItem(
                specialist="technical",
                reference_id="tech_ev_0",
                detail="Price 185.50 above 50-day SMA",
            ),
            AggregatedEvidenceItem(
                specialist="fundamental",
                reference_id="gross_margin",
                detail="Gross margin at 45.0%",
            ),
            AggregatedEvidenceItem(
                specialist="research",
                reference_id="chunk_01",
                detail="Services segment expansion from AAPL_10K.pdf",
                document_id="doc_01",
                chunk_id="chunk_01",
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


# ===========================================================================
# 1. FULL AND PARTIAL SPECIALIST REPORT GENERATION
# ===========================================================================


class TestReportGeneratorFullAndPartialExecution:
    """Tests 1-5: Full, partial, missing, failed, and insufficient specialist flows."""

    def test_complete_aggregated_input_generates_valid_report(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 1: Complete aggregated input produces a complete FinalReport."""
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(sample_unified_analysis)

        assert result.success is True
        report: FinalReport = result.data

        # All 15 required sections verified
        assert report.company.ticker == "AAPL"
        assert report.company.name == "Apple Inc."
        assert report.investor_profile is not None
        assert report.horizon == "3-5 years"
        assert report.capital is not None and report.capital.amount == 50000.0
        assert report.technical is not None
        assert report.fundamental is not None
        assert report.news is not None
        assert report.research is not None
        assert report.risk is not None
        assert report.overall_assessment is not None
        assert report.recommendation is not None
        assert len(report.key_reasons) >= 1
        assert len(report.important_risks) >= 1
        assert len(report.evidence_sources) == 3
        assert STANDARD_DISCLAIMER in report.disclaimer
        assert report.is_complete is True

    def test_partial_specialist_availability(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 2: Handles partial specialist availability cleanly."""
        partial = sample_unified_analysis.model_copy(
            update={
                "research_assessment": None,
                "specialist_statuses": {
                    "technical": SpecialistStatus.AVAILABLE,
                    "fundamental": SpecialistStatus.AVAILABLE,
                    "news": SpecialistStatus.AVAILABLE,
                    "research": SpecialistStatus.MISSING,
                    "risk": SpecialistStatus.AVAILABLE,
                },
                "missing_specialists": ["research"],
                "data_completeness_ratio": 0.8,
            }
        )
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(partial)

        assert result.success is True
        report: FinalReport = result.data
        assert report.research is None
        assert "research" in report.missing_specialists
        assert report.is_complete is True
        assert report.recommendation is not None

    def test_missing_specialist_explicitly_noted(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 3: Missing core specialist tracked without fabrication."""
        missing_tech = sample_unified_analysis.model_copy(
            update={
                "technical_assessment": None,
                "specialist_statuses": {
                    "technical": SpecialistStatus.MISSING,
                    "fundamental": SpecialistStatus.AVAILABLE,
                    "news": SpecialistStatus.AVAILABLE,
                    "research": SpecialistStatus.AVAILABLE,
                    "risk": SpecialistStatus.AVAILABLE,
                },
                "missing_specialists": ["technical"],
                "data_completeness_ratio": 0.75,
            }
        )
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(missing_tech)

        assert result.success is True
        report: FinalReport = result.data
        assert report.technical is None
        assert "technical" in report.missing_specialists
        assert report.specialist_statuses["technical"] == SpecialistStatus.MISSING

    def test_failed_specialist_handling(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 4: Failed specialist records error and preserves status."""
        failed_news = sample_unified_analysis.model_copy(
            update={
                "news_assessment": None,
                "specialist_statuses": {
                    "technical": SpecialistStatus.AVAILABLE,
                    "fundamental": SpecialistStatus.AVAILABLE,
                    "news": SpecialistStatus.FAILED,
                    "research": SpecialistStatus.AVAILABLE,
                    "risk": SpecialistStatus.AVAILABLE,
                },
                "failed_specialists": ["news"],
                "specialist_errors": {"news": "API quota rate limited"},
                "data_completeness_ratio": 0.8,
            }
        )
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(failed_news)

        assert result.success is True
        report: FinalReport = result.data
        assert report.news is None
        assert "news" in report.failed_specialists
        assert report.specialist_statuses["news"] == SpecialistStatus.FAILED

    def test_insufficient_evidence_communicates_limitations(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 5: Communicates data limitations honestly without fabrication."""
        insufficient = sample_unified_analysis.model_copy(
            update={
                "insufficient_evidence": True,
                "insufficient_evidence_reason": "Zero core specialists available.",
                "data_completeness_ratio": 0.0,
                "technical_assessment": None,
                "fundamental_assessment": None,
                "news_assessment": None,
                "confidence": 0.0,
            }
        )
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(insufficient)

        assert result.success is True
        report: FinalReport = result.data
        assert report.insufficient_evidence is True
        assert (
            report.recommendation.stance == RecommendationStance.INSUFFICIENT_EVIDENCE
        )
        assert "insufficient" in report.recommendation.rationale.lower()
        assert report.is_complete is True


# ===========================================================================
# 2. GROUNDING & SAFETY VALIDATION
# ===========================================================================


class TestRecommendationAndFactualGrounding:
    """Tests 6-10: Evidence grounding, numeric provenance, citations, and safety."""

    def test_recommendation_grounding_reflects_conflicts(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 6: Recommendation stance reflects conflicts (CAUTIOUS)."""
        conflicted = sample_unified_analysis.model_copy(
            update={
                "signal_conflicts": [
                    SignalConflict(
                        topic="Technical Bullish vs Fundamental Weakness",
                        description="Price in uptrend but profit margins deteriorate.",
                        specialist_positions={
                            "technical": "bullish",
                            "fundamental": "weak",
                        },
                        involved_specialists=["technical", "fundamental"],
                    )
                ]
            }
        )
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(conflicted)

        assert result.success is True
        report: FinalReport = result.data
        assert report.recommendation.stance == RecommendationStance.CAUTIOUS
        assert "caution" in report.recommendation.rationale.lower()
        assert any("conflict" in r.lower() for r in report.important_risks)

    def test_preservation_of_evidence_and_source_references(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 7: Attributed evidence references must be preserved verbatim."""
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(sample_unified_analysis)

        assert result.success is True
        report: FinalReport = result.data
        ref_ids = [e.reference_id for e in report.evidence_sources]
        assert "tech_ev_0" in ref_ids
        assert "gross_margin" in ref_ids
        assert "chunk_01" in ref_ids

    def test_fabricated_financial_value_rejected(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 8: Rejects invented financial numbers not in source data."""
        det_report = generate_deterministic_report(sample_unified_analysis)

        # Inject fabricated numeric metric
        corrupted_report = det_report.model_copy(
            update={
                "key_reasons": [
                    "Quarterly revenue unexpectedly jumped to $99999.00 billion."
                ]
            }
        )
        with pytest.raises(ReportValidationError) as exc_info:
            validate_report_grounding(corrupted_report, sample_unified_analysis)

        assert "ungrounded numerical value" in str(exc_info.value).lower()

    def test_fabricated_source_reference_rejected(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 9: Rejects non-existent evidence citations."""
        det_report = generate_deterministic_report(sample_unified_analysis)

        # Inject fake citation
        corrupted_rec = det_report.recommendation.model_copy(
            update={
                "rationale": (
                    "Earnings growth supported by [ref: fake_nonexistent_chunk]."
                )
            }
        )
        corrupted_report = det_report.model_copy(
            update={"recommendation": corrupted_rec}
        )

        with pytest.raises(ReportValidationError) as exc_info:
            validate_report_grounding(corrupted_report, sample_unified_analysis)

        assert "ungrounded evidence reference" in str(exc_info.value).lower()

    def test_prohibited_guarantee_certainty_language_rejected(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 10: Rejects prohibited guarantee/certainty claims."""
        det_report = generate_deterministic_report(sample_unified_analysis)

        # Inject prohibited advice/certainty language
        corrupted_rec = det_report.recommendation.model_copy(
            update={
                "rationale": (
                    "This investment delivers guaranteed returns with zero risk."
                )
            }
        )
        corrupted_report = det_report.model_copy(
            update={"recommendation": corrupted_rec}
        )

        with pytest.raises(ReportValidationError) as exc_info:
            validate_report_grounding(corrupted_report, sample_unified_analysis)

        assert "prohibited advisory phrase" in str(exc_info.value).lower()


# ===========================================================================
# 3. LLM SYNTHESIS & ROBUSTNESS
# ===========================================================================


class TestLLMIntegrationAndRobustness:
    """Tests 11-12: LLM generation, malformed response, and failure fallback."""

    def test_llm_structured_generation_success(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """LLM structured generation produces complete, validated report."""
        mock_output = GeneratedReportSynthesis(
            recommendation_stance=RecommendationStance.FAVORABLE,
            recommendation_rationale=(
                "Analytical indicators for AAPL show favorable alignment across "
                "cash flow strength and technical momentum above 180.0 support."
            ),
            profile_alignment=(
                "Favorable stance aligns with investor's 3-5 years growth horizon."
            ),
            monitoring_points=[
                "Next quarterly earnings disclosure",
                "Holding 180.0 primary support level",
            ],
            key_reasons=[
                "Robust operating cash flows and balance sheet strength",
                "Price action continuing in disciplined uptrend",
            ],
            important_risks=[
                "Regulatory antitrust reviews across mobile app ecosystem",
                "Potential macroeconomic interest rate volatility",
            ],
            executive_synthesis=(
                "Apple combines steady high-margin services growth with momentum."
            ),
        )

        mock_provider = MagicMock(spec=LLMProvider)
        agent = ReportGeneratorAgent(provider=mock_provider, deterministic_only=False)

        with patch(
            "app.agents.report_generator.generate_structured", return_value=mock_output
        ):
            result = agent.run(sample_unified_analysis)

        assert result.success is True
        report: FinalReport = result.data
        assert report.recommendation.stance == RecommendationStance.FAVORABLE
        assert "180.0" in report.recommendation.rationale
        assert report.is_complete is True

    def test_malformed_invalid_llm_output_falls_back_gracefully(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 11: Schema/parsing error in LLM output triggers fallback."""
        mock_provider = MagicMock(spec=LLMProvider)
        agent = ReportGeneratorAgent(provider=mock_provider, deterministic_only=False)

        with patch(
            "app.agents.report_generator.generate_structured",
            side_effect=LLMStructuredOutputError(
                "Schema validation failed: missing field"
            ),
        ):
            result = agent.run(sample_unified_analysis)

        assert result.success is True
        report: FinalReport = result.data
        assert report.is_complete is True
        assert report.recommendation is not None
        assert STANDARD_DISCLAIMER in report.disclaimer

    def test_llm_provider_failure_falls_back_gracefully(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 12: Network/API exception triggers deterministic fallback."""
        mock_provider = MagicMock(spec=LLMProvider)
        agent = ReportGeneratorAgent(provider=mock_provider, deterministic_only=False)

        with patch(
            "app.agents.report_generator.generate_structured",
            side_effect=LLMError("Connection reset by peer"),
        ):
            result = agent.run(sample_unified_analysis)

        assert result.success is True
        report: FinalReport = result.data
        assert report.is_complete is True
        assert report.recommendation is not None


# ===========================================================================
# 4. CONTEXT & PROFILE PRESERVATION
# ===========================================================================


class TestContextAndProfilePreservation:
    """Tests 13-15: Investor profile, capital/horizon, and completeness tracking."""

    def test_investor_profile_preservation(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 13: Investor profile constraints are preserved exactly."""
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(sample_unified_analysis)

        assert result.success is True
        report: FinalReport = result.data
        assert report.investor_profile is not None
        assert (
            report.investor_profile.investment_goal == "long-term capital appreciation"
        )
        assert report.investor_profile.risk_tolerance == "moderate"

    def test_horizon_and_capital_preservation(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 14: Time horizon and allocated capital are preserved."""
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(sample_unified_analysis)

        assert result.success is True
        report: FinalReport = result.data
        assert report.horizon == "3-5 years"
        assert report.capital is not None
        assert report.capital.amount == 50000.0

    def test_report_completeness_and_status_handling(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Requirement 15: Final report completeness indicator is True."""
        agent = ReportGeneratorAgent(deterministic_only=True)
        result = agent.run(sample_unified_analysis)

        assert result.success is True
        report: FinalReport = result.data
        assert report.is_complete is True
        assert len(report.key_reasons) >= 1
        assert len(report.important_risks) >= 1
        assert report.recommendation is not None


# ===========================================================================
# 5. PROMPT FORMATTING & LANGGRAPH NODE ADAPTER
# ===========================================================================


class TestPromptAndNodeAdapter:
    """Tests prompt formatting and LangGraph node integration."""

    def test_prompt_formatter_contains_all_context(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """Prompt formatter contains all specialist blocks and instructions."""
        prompt = format_report_generator_prompt(sample_unified_analysis)

        assert "TARGET TICKER: AAPL" in prompt
        assert "TECHNICAL ANALYST FINDINGS" in prompt
        assert "FUNDAMENTAL ANALYST FINDINGS" in prompt
        assert "NEWS & SENTIMENT FINDINGS" in prompt
        assert "DOCUMENT & SEC RESEARCH FINDINGS" in prompt
        assert "RISK ASSESSMENT FINDINGS" in prompt
        assert "PHASE 11 SYNTHESIS" in prompt
        assert "ATTRIBUTED EVIDENCE ITEMS" in prompt
        assert "tech_ev_0" in prompt
        assert "NEVER invent numbers" in prompt

    def test_report_generator_node_success(
        self, sample_unified_analysis: UnifiedSpecialistAnalysis
    ) -> None:
        """LangGraph node extracts aggregated_result and returns update for 'report'."""
        state: GraphState = {
            "user_query": "Generate final report for AAPL",
            "aggregated_result": {
                "success": True,
                "data": sample_unified_analysis.model_dump(),
            },
        }
        agent = ReportGeneratorAgent(deterministic_only=True)
        update = report_generator_node(state, agent=agent)

        assert "report" in update
        assert update["report"]["success"] is True
        report_data = update["report"]["data"]
        assert report_data["company"]["ticker"] == "AAPL"
        assert report_data["recommendation"]["stance"] == "favorable"

    def test_report_generator_node_missing_aggregated_result_fails(self) -> None:
        """LangGraph node returns failure update if aggregated_result is missing."""
        state: GraphState = {
            "user_query": "Generate report for AAPL",
        }
        agent = ReportGeneratorAgent(deterministic_only=True)
        update = report_generator_node(state, agent=agent)

        assert "report" in update
        assert update["report"]["success"] is False
        assert "aggregated_result" in update["report"]["error"]
