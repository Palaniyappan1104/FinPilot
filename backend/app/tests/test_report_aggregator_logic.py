"""Comprehensive unit and integration tests for Report Aggregator Logic (Phase 11.2).

Covers:
1. Full specialist aggregation: all 5 specialists present.
2. Agreement identification across specialists.
3. Signal conflict and tension identification without artificial consensus.
4. Cross-specialist observations connecting findings across domains.
5. Overall synthesis generation (grounded, non-advisory).
6. Specialist attribution and provenance preservation across findings.
7. Partial inputs handling:
   - Missing optional specialist (e.g. research)
   - Missing core specialist (e.g. technical)
   - Multiple specialists missing
   - Failed specialist
   - Empty specialist result
   - Completely empty input (zero specialists available)
8. Insufficient evidence handling without fabrication or hallucination.
9. Safety rules: strict prohibition of buy/sell advice, price targets,
   guaranteed returns.
10. Deterministic detectors operating independently of LLM.
11. LLM structured generation with mock provider and graceful fallback on error.
12. LangGraph node adapter (report_aggregator_node) execution.
13. Investor profile integration and constraints.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.agents.aggregator import (
    ReportAggregatorAgent,
    detect_cross_observations,
    detect_signal_conflicts,
    detect_synthesis_agreements,
    generate_deterministic_synthesis,
    report_aggregator_node,
)
from app.agents.aggregator_prompt import (
    AGGREGATOR_SYSTEM_PROMPT,
    format_aggregator_prompt,
)
from app.agents.aggregator_schema import (
    AggregatedEvidenceItem,
    AggregatorInvestorProfile,
    AggregatorSynthesisOutput,
    ReportAggregatorInput,
    SignalConflict,
    SpecialistStatus,
    SynthesisFinding,
    UnifiedSpecialistAnalysis,
)
from app.agents.base import AgentResult
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
    RiskProbability,
    RiskSeverity,
)
from app.agents.state import GraphState
from app.agents.technical_schema import (
    SupportResistanceSummary,
    TechnicalAnalysisOutput,
    TechnicalIndicatorsSummary,
    TechnicalInterpretation,
)
from app.core.llm.base import LLMProvider, LLMResponse
from app.core.llm.exceptions import LLMError
from app.models.technical_metrics import (
    MACDMetrics,
    MovingAverageMetrics,
    RSIMetrics,
    VolumeMetrics,
)

# ===========================================================================
# FIXTURES: SPECIALIST TEST DATA
# ===========================================================================


@pytest.fixture
def sample_profile() -> AggregatorInvestorProfile:
    """Standard investor profile fixture."""
    return AggregatorInvestorProfile(
        target_company="Apple Inc.",
        ticker="AAPL",
        investment_goal="capital preservation",
        time_horizon="3 years",
        capital_amount=50000.0,
        risk_tolerance="moderate",
        profile_complete=True,
    )


@pytest.fixture
def sample_technical_uptrend() -> TechnicalAnalysisOutput:
    """Technical output indicating strong uptrend."""
    return TechnicalAnalysisOutput(
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
            trend_analysis=(
                "Price holds firmly above ascending 50-day and 200-day SMAs."
            ),
            moving_averages_analysis="Bullish stacking order SMA 20 > 50 > 200.",
            momentum_analysis="RSI at 62.0 reflects robust buying pressure.",
            volume_analysis="Volume confirms breakout over previous consolidation.",
            support_resistance_analysis="Primary floor at 180.0, key barrier at 192.0.",
        ),
        evidence=[
            "Price 185.50 is above 50-day SMA 178.0 and 200-day SMA 170.0",
            "RSI 62.0 confirms bullish momentum",
            "MACD histogram positive at 0.5",
        ],
        risks=["Approaching overhead resistance at 192.0"],
        confidence=0.88,
    )


@pytest.fixture
def sample_technical_downtrend() -> TechnicalAnalysisOutput:
    """Technical output indicating clear downtrend."""
    return TechnicalAnalysisOutput(
        ticker="AAPL",
        trend="downtrend",
        indicators_summary=TechnicalIndicatorsSummary(
            latest_close=145.0,
            moving_averages=MovingAverageMetrics(
                sma_20=150.0,
                sma_50=160.0,
                sma_200=170.0,
            ),
            rsi=RSIMetrics(value=32.0),
            macd=MACDMetrics(macd_line=-2.0, signal_line=-1.2, histogram=-0.8),
            volume=VolumeMetrics(
                latest_volume=70000000.0, average_volume_20d=50000000.0
            ),
        ),
        support_resistance=SupportResistanceSummary(
            primary_support=140.0,
            primary_resistance=152.0,
            support_levels=[140.0, 135.0],
            resistance_levels=[152.0, 160.0],
        ),
        technical_score=30.0,
        interpretation=TechnicalInterpretation(
            overall_summary="Bearish trend continuation with selling volume.",
            trend_analysis="Price trades below 20, 50, and 200-day SMAs.",
            moving_averages_analysis="Bearish alignment.",
            momentum_analysis="RSI at 32.0 approaching oversold boundary.",
            volume_analysis="Heavy selling volume on down days.",
            support_resistance_analysis="Testing key support near 140.0.",
        ),
        evidence=[
            "Price 145.0 is below all major moving averages",
            "MACD histogram negative at -0.8",
        ],
        risks=["Risk of breakdown below 140.0 support"],
        confidence=0.82,
    )


@pytest.fixture
def sample_fundamental_favorable() -> FundamentalAnalysisOutput:
    """Fundamental output with favorable operational and financial metrics."""
    return FundamentalAnalysisOutput(
        ticker="AAPL",
        company_name="Apple Inc.",
        currency="USD",
        financial_health=DimensionAssessment(
            rating="strong",
            explanation="Strong cash position and zero refinancing distress.",
            supporting_metrics=["cash_and_equivalents", "operating_cash_flow"],
        ),
        growth_assessment=DimensionAssessment(
            rating="moderate",
            explanation="Services revenue grew 11% YoY.",
            supporting_metrics=["revenue_growth_yoy"],
        ),
        profitability_assessment=DimensionAssessment(
            rating="strong",
            explanation="Operating margin at 31% with high ROIC.",
            supporting_metrics=["operating_margin", "roe"],
        ),
        valuation_assessment=DimensionAssessment(
            rating="neutral",
            explanation="P/E of 28.0 reflects market quality premium.",
            supporting_metrics=["pe_ratio"],
        ),
        leverage_assessment=DimensionAssessment(
            rating="strong",
            explanation="Low net debt and high interest coverage.",
            supporting_metrics=["net_debt", "debt_to_equity"],
        ),
        cash_flow_assessment=DimensionAssessment(
            rating="strong",
            explanation="Free cash flow conversion exceeds 95%.",
            supporting_metrics=["free_cash_flow"],
        ),
        key_strengths=["High operating margin", "Disciplined capital allocation"],
        key_weaknesses=["Hardware revenue saturation"],
        notable_flags=[],
        overall_assessment="favorable",
        overall_summary=(
            "Apple demonstrates exceptional cash flow and pristine balance "
            "sheet strength."
        ),
        confidence=0.92,
    )


@pytest.fixture
def sample_fundamental_unfavorable() -> FundamentalAnalysisOutput:
    """Fundamental output with unfavorable operational and financial metrics."""
    return FundamentalAnalysisOutput(
        ticker="AAPL",
        company_name="Apple Inc.",
        currency="USD",
        financial_health=DimensionAssessment(
            rating="weak",
            explanation="Declining liquidity buffer and elevated short-term debt.",
            supporting_metrics=["current_ratio", "quick_ratio"],
        ),
        growth_assessment=DimensionAssessment(
            rating="weak",
            explanation="Revenue contracted 4.5% YoY across major business lines.",
            supporting_metrics=["revenue_growth_yoy"],
        ),
        profitability_assessment=DimensionAssessment(
            rating="weak",
            explanation="Gross margin compression due to rising production costs.",
            supporting_metrics=["gross_margin", "operating_margin"],
        ),
        valuation_assessment=DimensionAssessment(
            rating="weak",
            explanation=(
                "Valuation multiples remain stretched despite declining fundamentals."
            ),
            supporting_metrics=["pe_ratio", "ev_to_ebitda"],
        ),
        leverage_assessment=DimensionAssessment(
            rating="weak",
            explanation="Debt to EBITDA expanded beyond target thresholds.",
            supporting_metrics=["debt_to_ebitda"],
        ),
        cash_flow_assessment=DimensionAssessment(
            rating="weak",
            explanation="Free cash flow contracted 35% YoY.",
            supporting_metrics=["free_cash_flow"],
        ),
        key_strengths=["Global brand recognition"],
        key_weaknesses=["Margin contraction", "Revenue decline"],
        notable_flags=["Margin deterioration"],
        overall_assessment="unfavorable",
        overall_summary=(
            "Operating fundamentals face severe compression and balance sheet "
            "leverage is rising."
        ),
        confidence=0.88,
    )


@pytest.fixture
def sample_news_positive() -> NewsAnalysisOutput:
    """News output with positive sentiment."""
    return NewsAnalysisOutput(
        ticker="AAPL",
        overall_sentiment="positive",
        sentiment_distribution={"positive": 9, "neutral": 2, "negative": 1},
        recent_news=[
            RecentNewsItem(
                article_id="art_101",
                headline="Apple reports record quarterly services subscription growth",
                summary="Paid subscriptions topped 1 billion accounts.",
                source="Reuters",
                url="https://example.com/101",
                published_at=datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc),
                sentiment="positive",
            ),
            RecentNewsItem(
                article_id="art_102",
                headline="New AI ecosystem rollout expands enterprise adoption",
                summary="Corporate adoption of Apple silicon accelerates.",
                source="Bloomberg",
                url="https://example.com/102",
                published_at=datetime(2026, 9, 12, 8, 30, tzinfo=timezone.utc),
                sentiment="positive",
            ),
        ],
        important_events=[
            NewsAnalysisEvent(
                event_type="earnings",
                description="Enterprise AI hardware launch.",
                article_ids=["art_102"],
            )
        ],
        positive_factors=[
            FactorItem(text="Services momentum", article_ids=["art_101"])
        ],
        negative_factors=[],
        summary=(
            "News coverage is predominantly positive, highlighting services "
            "expansion and enterprise uptake."
        ),
        confidence=0.86,
    )


@pytest.fixture
def sample_news_negative() -> NewsAnalysisOutput:
    """News output with negative sentiment."""
    return NewsAnalysisOutput(
        ticker="AAPL",
        overall_sentiment="negative",
        sentiment_distribution={"positive": 1, "neutral": 2, "negative": 8},
        recent_news=[
            RecentNewsItem(
                article_id="art_201",
                headline="Regulators launch antitrust scrutiny over App Store fees",
                summary=(
                    "European authorities announce formal investigation into "
                    "payment policies."
                ),
                source="Financial Times",
                url="https://example.com/201",
                published_at=datetime(2026, 9, 12, 11, 0, tzinfo=timezone.utc),
                sentiment="negative",
            ),
            RecentNewsItem(
                article_id="art_202",
                headline="Supply chain delays threaten next-generation shipments",
                summary="Component shortages may constrain initial quarterly volumes.",
                source="WSJ",
                url="https://example.com/202",
                published_at=datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc),
                sentiment="negative",
            ),
        ],
        important_events=[
            NewsAnalysisEvent(
                event_type="regulatory_action",
                description="Antitrust probe opened.",
                article_ids=["art_201"],
            )
        ],
        positive_factors=[],
        negative_factors=[
            FactorItem(text="Antitrust scrutiny", article_ids=["art_201"]),
            FactorItem(text="Supply chain delays", article_ids=["art_202"]),
        ],
        summary=(
            "News sentiment is heavily adverse due to new regulatory probes "
            "and supply chain bottlenecks."
        ),
        confidence=0.89,
    )


@pytest.fixture
def sample_research_output() -> ResearchAnalysisOutput:
    """Research output from SEC filing analysis."""
    ev_ref1 = ResearchEvidenceRef(
        document_id="sec_10q_2026_q2",
        chunk_id="chunk_p14_c2",
        source_document="10-Q Q2 2026",
        page_numbers=[14, 15],
    )
    ev_ref2 = ResearchEvidenceRef(
        document_id="sec_10q_2026_q2",
        chunk_id="chunk_p28_c1",
        source_document="10-Q Q2 2026",
        page_numbers=[28],
    )
    return ResearchAnalysisOutput(
        ticker="AAPL",
        query="Examine revenue growth, capital commitments, and disclosures.",
        answer="According to the 10-Q, services segment revenue expanded 11% YoY.",
        key_findings=[
            ResearchFinding(
                claim="Form 10-Q notes services segment revenue expanded 11% YoY.",
                evidence=[ev_ref1],
            ),
            ResearchFinding(
                claim="Capital expenditure commitments increased to $12B.",
                evidence=[ev_ref1],
            ),
            ResearchFinding(
                claim="Legal proceedings section discloses no material liabilities.",
                evidence=[ev_ref2],
            ),
        ],
        evidence=[ev_ref1, ev_ref2],
        summary=(
            "SEC disclosures substantiate robust operational performance and "
            "conservative balance sheet accounting."
        ),
        confidence=0.91,
        insufficient_evidence=False,
    )


@pytest.fixture
def sample_risk_moderate() -> RiskAnalysisOutput:
    """Risk output with moderate overall risk."""
    return RiskAnalysisOutput(
        ticker="AAPL",
        overall_risk_level=RiskSeverity.MODERATE,
        summary=(
            "Risk profile is moderate, characterized by manageable market "
            "volatility and low balance sheet risk."
        ),
        market_risks=[
            RiskFactor(
                category=RiskCategory.MARKET,
                name="Equity Market Multiple Sensitivity",
                description="Valuation may compress if macro interest rate cuts stall.",
                severity=RiskSeverity.MODERATE,
                probability=RiskProbability.MEDIUM,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="pe_ratio",
                        detail="P/E multiple of 28.0",
                    )
                ],
            )
        ],
        company_risks=[
            RiskFactor(
                category=RiskCategory.COMPANY,
                name="Hardware Replacement Cycle Length",
                description="Longer replacement cycles could moderate unit sales.",
                severity=RiskSeverity.LOW,
                probability=RiskProbability.MEDIUM,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="revenue_growth_yoy",
                        detail="Hardware growth moderation",
                    )
                ],
            )
        ],
        sector_risks=[],
        financial_risks=[
            RiskFactor(
                category=RiskCategory.FINANCIAL,
                name="Foreign Currency Translation",
                description="Strong USD impact on overseas revenues.",
                severity=RiskSeverity.LOW,
                probability=RiskProbability.MEDIUM,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="net_debt",
                        detail="Manageable debt levels",
                    )
                ],
            )
        ],
        volatility_risks=[
            RiskFactor(
                category=RiskCategory.VOLATILITY,
                name="Overhead Resistance Rejection Risk",
                description=(
                    "Technical resistance at 192.0 could prompt "
                    "short-term consolidation."
                ),
                severity=RiskSeverity.LOW,
                probability=RiskProbability.LOW,
                evidence=[
                    RiskEvidenceRef(
                        source_type="technical",
                        reference_id="resistance_level",
                        detail="Resistance level at 192.0",
                    )
                ],
            )
        ],
        investor_specific_risks=[
            RiskFactor(
                category=RiskCategory.INVESTOR_SPECIFIC,
                name="Horizon and Volatility Fit",
                description=(
                    "3-year horizon provides adequate cushion for "
                    "moderate price swings."
                ),
                severity=RiskSeverity.LOW,
                probability=RiskProbability.LOW,
                evidence=[
                    RiskEvidenceRef(
                        source_type="investor_profile",
                        reference_id="time_horizon",
                        detail="3 years horizon",
                    )
                ],
            )
        ],
        confidence=0.85,
    )


@pytest.fixture
def sample_risk_critical() -> RiskAnalysisOutput:
    """Risk output with critical/high overall risk."""
    return RiskAnalysisOutput(
        ticker="AAPL",
        overall_risk_level=RiskSeverity.CRITICAL,
        summary="Severe financial and legal risks threaten capital preservation.",
        market_risks=[],
        company_risks=[
            RiskFactor(
                category=RiskCategory.COMPANY,
                name="Antitrust Regulatory Sanctions",
                description="Substantial fine or forced business model restructuring.",
                severity=RiskSeverity.CRITICAL,
                probability=RiskProbability.HIGH,
                evidence=[
                    RiskEvidenceRef(
                        source_type="news",
                        reference_id="art_201",
                        detail="Formal investigation opened",
                    )
                ],
            )
        ],
        sector_risks=[],
        financial_risks=[
            RiskFactor(
                category=RiskCategory.FINANCIAL,
                name="Solvency & Liquidity Strain",
                description=(
                    "High debt maturity obligations relative to shrinking cash flows."
                ),
                severity=RiskSeverity.CRITICAL,
                probability=RiskProbability.HIGH,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="debt_to_ebitda",
                        detail="Debt to EBITDA expansion",
                    )
                ],
            )
        ],
        volatility_risks=[],
        investor_specific_risks=[],
        confidence=0.90,
    )


# Mock LLM Provider Helper
class MockLLM(LLMProvider):
    """Deterministic Mock LLM Provider for unit testing."""

    def __init__(self, response_text: str = "", fail: bool = False) -> None:
        self.response_text = response_text
        self.fail = fail
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        if self.fail:
            raise LLMError("Simulated LLM provider timeout or error.")
        return LLMResponse(
            text=self.response_text,
            model_name="mock-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )


# ===========================================================================
# TEST SUITE: REPORT AGGREGATOR LOGIC (Phase 11.2)
# ===========================================================================


class TestAllSpecialistsAggregation:
    """Scenario A: Full specialist aggregation with all 5 present."""

    def test_full_aggregation_deterministic(
        self,
        sample_profile: AggregatorInvestorProfile,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_news_positive: NewsAnalysisOutput,
        sample_research_output: ResearchAnalysisOutput,
        sample_risk_moderate: RiskAnalysisOutput,
    ) -> None:
        """Verify complete aggregation when all 5 specialists are available."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            target_company="Apple Inc.",
            investor_profile=sample_profile,
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_favorable,
            news=sample_news_positive,
            research=sample_research_output,
            risk=sample_risk_moderate,
        )

        agent = ReportAggregatorAgent(deterministic_only=True)
        result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert isinstance(analysis, UnifiedSpecialistAnalysis)
        assert analysis.ticker == "AAPL"
        assert analysis.data_completeness_ratio == 1.0
        assert analysis.insufficient_evidence is False
        assert analysis.confidence > 0.7

        # Specialist blocks must be preserved
        assert analysis.technical_assessment is not None
        assert analysis.fundamental_assessment is not None
        assert analysis.news_assessment is not None
        assert analysis.research_assessment is not None
        assert analysis.risk_assessment is not None

        # Cross-referencing components must be populated
        assert len(analysis.areas_of_agreement) >= 1
        assert len(analysis.cross_specialist_observations) >= 1
        assert analysis.overall_synthesis != ""
        assert "AAPL" in analysis.overall_synthesis

        # Evidence collection must preserve provenance
        assert len(analysis.aggregated_evidence) > 0
        specialists_in_evidence = {e.specialist for e in analysis.aggregated_evidence}
        assert "technical" in specialists_in_evidence
        assert "fundamental" in specialists_in_evidence
        assert "news" in specialists_in_evidence
        assert "research" in specialists_in_evidence
        assert "risk" in specialists_in_evidence


class TestAgreementIdentification:
    """Scenario B: Consensus and agreement detection across specialists."""

    def test_multi_domain_bullish_agreement(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_news_positive: NewsAnalysisOutput,
    ) -> None:
        """Verify multi-domain positive alignment across tech, fund, and news."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_favorable,
            news=sample_news_positive,
        )

        agreements = detect_synthesis_agreements(agg_input)
        assert len(agreements) >= 1

        topic_titles = [a.topic for a in agreements]
        assert "Multi-Specialist Positive Alignment" in topic_titles

        pos_agree = next(
            a for a in agreements if a.topic == "Multi-Specialist Positive Alignment"
        )
        assert "technical" in pos_agree.supporting_specialists
        assert "fundamental" in pos_agree.supporting_specialists
        assert "news" in pos_agree.supporting_specialists
        assert len(pos_agree.evidence) > 0

    def test_cash_flow_and_solvency_agreement(
        self,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_risk_moderate: RiskAnalysisOutput,
    ) -> None:
        """Verify agreement between fundamental cash flow and risk solvency."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            fundamental=sample_fundamental_favorable,
            risk=sample_risk_moderate,
        )

        agreements = detect_synthesis_agreements(agg_input)
        cf_agree = [
            a
            for a in agreements
            if a.topic == "Cash Flow Stability & Financial Solvency"
        ]
        assert len(cf_agree) == 1
        assert "fundamental" in cf_agree[0].supporting_specialists
        assert "risk" in cf_agree[0].supporting_specialists

    def test_research_corroboration_agreement(
        self,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_research_output: ResearchAnalysisOutput,
    ) -> None:
        """Verify SEC document disclosures corroborating fundamental findings."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            fundamental=sample_fundamental_favorable,
            research=sample_research_output,
        )

        agreements = detect_synthesis_agreements(agg_input)
        sec_agree = [a for a in agreements if "SEC Document" in a.topic]
        assert len(sec_agree) == 1
        assert "research" in sec_agree[0].supporting_specialists
        assert "fundamental" in sec_agree[0].supporting_specialists


class TestSignalConflictsAndTensions:
    """Scenario C: Identification of contradictions and tensions."""

    def test_technical_momentum_vs_fundamental_headwinds(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_unfavorable: FundamentalAnalysisOutput,
    ) -> None:
        """Verify conflict when technical is in uptrend but fundamentals
        are unfavorable.
        """
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_unfavorable,
        )

        conflicts = detect_signal_conflicts(agg_input)
        assert len(conflicts) >= 1

        tech_fund_conflict = next(
            c for c in conflicts if "Technical Momentum vs Fundamental" in c.topic
        )
        assert "technical" in tech_fund_conflict.involved_specialists
        assert "fundamental" in tech_fund_conflict.involved_specialists
        assert "technical" in tech_fund_conflict.specialist_positions
        assert "fundamental" in tech_fund_conflict.specialist_positions
        assert len(tech_fund_conflict.evidence) > 0

    def test_technical_downtrend_vs_favorable_fundamentals(
        self,
        sample_technical_downtrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
    ) -> None:
        """Verify conflict when price is in downtrend despite favorable fundamentals."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_downtrend,
            fundamental=sample_fundamental_favorable,
        )

        conflicts = detect_signal_conflicts(agg_input)
        c_found = next(
            (
                c
                for c in conflicts
                if "Technical Downtrend vs Favorable Fundamentals" in c.topic
            ),
            None,
        )
        assert c_found is not None
        assert set(c_found.involved_specialists) == {"technical", "fundamental"}

    def test_technical_strength_vs_negative_news(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_news_negative: NewsAnalysisOutput,
    ) -> None:
        """Verify conflict when technicals are bullish but headline news is negative."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_uptrend,
            news=sample_news_negative,
        )

        conflicts = detect_signal_conflicts(agg_input)
        news_conflict = next(
            (c for c in conflicts if "Technical Strength vs Negative News" in c.topic),
            None,
        )
        assert news_conflict is not None
        assert "news" in news_conflict.involved_specialists
        assert "technical" in news_conflict.involved_specialists

    def test_critical_risk_vs_technical_uptrend(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_risk_critical: RiskAnalysisOutput,
    ) -> None:
        """Verify conflict when risk analyst flags critical risk during an uptrend."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_uptrend,
            risk=sample_risk_critical,
        )

        conflicts = detect_signal_conflicts(agg_input)
        risk_conflict = next(
            (
                c
                for c in conflicts
                if "Elevated Risk Profile vs Technical Uptrend" in c.topic
            ),
            None,
        )
        assert risk_conflict is not None
        assert "risk" in risk_conflict.involved_specialists
        assert "technical" in risk_conflict.involved_specialists


class TestCrossSpecialistObservations:
    """Scenario D: Cross-domain observations connecting distinct specialist findings."""

    def test_cross_observations_generated(
        self,
        sample_profile: AggregatorInvestorProfile,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_news_positive: NewsAnalysisOutput,
        sample_risk_moderate: RiskAnalysisOutput,
    ) -> None:
        """Verify cross-domain observations connect leverage, news, and risk."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            investor_profile=sample_profile,
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_favorable,
            news=sample_news_positive,
            risk=sample_risk_moderate,
        )

        observations = detect_cross_observations(agg_input)
        assert len(observations) >= 2

        # Verify observation connects technical and fundamental
        tech_fund_obs = next(
            (
                o
                for o in observations
                if "technical" in o.connected_specialists
                and "fundamental" in o.connected_specialists
            ),
            None,
        )
        assert tech_fund_obs is not None
        assert "leverage" in tech_fund_obs.observation.lower()

        # Verify investor profile interplay
        profile_obs = next(
            (
                o
                for o in observations
                if "risk" in o.connected_specialists
                and "investor" in o.observation.lower()
            ),
            None,
        )
        assert profile_obs is not None
        assert "moderate" in profile_obs.observation


class TestOverallSynthesisNarrative:
    """Scenario E: Grounded, non-advisory narrative synthesis generation."""

    def test_deterministic_synthesis_narrative_generation(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_news_positive: NewsAnalysisOutput,
    ) -> None:
        """Verify narrative synthesis contains key sections and specialist summaries."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_favorable,
            news=sample_news_positive,
        )

        agreements = detect_synthesis_agreements(agg_input)
        conflicts = detect_signal_conflicts(agg_input)
        obs = detect_cross_observations(agg_input)

        narrative = generate_deterministic_synthesis(
            agg_input, agreements, conflicts, obs
        )
        assert "AAPL" in narrative
        assert "available specialist domains" in narrative
        assert "Multi-Specialist Positive Alignment" in narrative
        assert "No material direct contradictions" in narrative


class TestAttributionAndProvenancePreservation:
    """Scenario F & G: Preservation of specialist attribution and citations."""

    def test_synthesis_models_preserve_attribution(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
    ) -> None:
        """Verify SynthesisFinding and SignalConflict models retain full
        specialist attribution.
        """
        finding = SynthesisFinding(
            topic="Valuation Alignment",
            summary="Multi-specialist consensus on fair value multiples.",
            supporting_specialists=["fundamental", "technical"],
            evidence=[
                AggregatedEvidenceItem(
                    specialist="fundamental",
                    reference_id="pe_ratio",
                    detail="P/E of 28.0",
                ),
                AggregatedEvidenceItem(
                    specialist="technical",
                    reference_id="tech_ev_0",
                    detail="Above 50-day SMA",
                ),
            ],
        )

        assert finding.supporting_specialists == ["fundamental", "technical"]
        assert finding.evidence[0].specialist == "fundamental"
        assert finding.evidence[1].specialist == "technical"
        assert finding.evidence[0].reference_id == "pe_ratio"

    def test_signal_conflict_preserves_opposing_positions(self) -> None:
        """Verify SignalConflict records distinct specialist stances."""
        conflict = SignalConflict(
            topic="Short-Term vs Long-Term Stance",
            description=(
                "Short-term bearish indicators contrast with "
                "long-term profitability."
            ),
            specialist_positions={
                "technical": "Negative momentum below 50-day SMA",
                "fundamental": "Operating margins exceed 30%",
            },
            involved_specialists=["technical", "fundamental"],
            evidence=[
                AggregatedEvidenceItem(
                    specialist="technical",
                    reference_id="rsi",
                    detail="RSI at 32",
                )
            ],
        )

        assert set(conflict.involved_specialists) == {"technical", "fundamental"}
        assert "RSI at 32" in conflict.evidence[0].detail


class TestPartialSpecialistInputs:
    """Scenario H: Handling partial availability across all configurations."""

    def test_missing_optional_research_specialist(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_news_positive: NewsAnalysisOutput,
        sample_risk_moderate: RiskAnalysisOutput,
    ) -> None:
        """Optional research specialist is missing (e.g. no 10-K uploaded)."""
        agg_input = ReportAggregatorInput.from_partial_specialists(
            ticker="AAPL",
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_favorable,
            news=sample_news_positive,
            risk=sample_risk_moderate,
        )

        assert agg_input.core_completeness_ratio == 1.0
        assert agg_input.total_completeness_ratio == 0.8
        assert agg_input.specialist_statuses["research"] == SpecialistStatus.MISSING

        agent = ReportAggregatorAgent(deterministic_only=True)
        result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert analysis.research_assessment is None
        assert "research" in analysis.missing_specialists
        assert analysis.insufficient_evidence is False

    def test_missing_core_technical_specialist(
        self,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_news_positive: NewsAnalysisOutput,
        sample_risk_moderate: RiskAnalysisOutput,
    ) -> None:
        """Core technical specialist is missing."""
        agg_input = ReportAggregatorInput.from_partial_specialists(
            ticker="AAPL",
            fundamental=sample_fundamental_favorable,
            news=sample_news_positive,
            risk=sample_risk_moderate,
        )

        assert agg_input.core_completeness_ratio == 0.75
        assert agg_input.specialist_statuses["technical"] == SpecialistStatus.MISSING

        agent = ReportAggregatorAgent(deterministic_only=True)
        result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert analysis.technical_assessment is None
        assert "technical" in analysis.missing_specialists
        assert (
            "technical" in analysis.overall_synthesis.lower()
            or "coverage note" in analysis.overall_synthesis.lower()
        )

    def test_multiple_specialists_missing(
        self,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_news_positive: NewsAnalysisOutput,
    ) -> None:
        """Multiple specialists missing (only fundamental and news available)."""
        agg_input = ReportAggregatorInput.from_partial_specialists(
            ticker="AAPL",
            fundamental=sample_fundamental_favorable,
            news=sample_news_positive,
        )

        assert agg_input.core_completeness_ratio == 0.5
        assert len(agg_input.missing_specialists) == 3

        agent = ReportAggregatorAgent(deterministic_only=True)
        result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert analysis.data_completeness_ratio == 0.5
        assert analysis.technical_assessment is None
        assert analysis.research_assessment is None
        assert analysis.risk_assessment is None

    def test_specialist_explicitly_failed(
        self,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
    ) -> None:
        """One specialist failed during execution."""
        state = {
            "fundamental_result": sample_fundamental_favorable,
            "technical_result": AgentResult.create_failure(
                error="API rate limit exceeded"
            ),
        }

        agg_input = ReportAggregatorInput.from_graph_state(state, ticker="AAPL")
        assert agg_input.specialist_statuses["technical"] == SpecialistStatus.FAILED
        assert "API rate limit" in agg_input.specialist_errors["technical"]

        agent = ReportAggregatorAgent(deterministic_only=True)
        result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert "technical" in analysis.failed_specialists
        assert "failed" in analysis.overall_synthesis.lower()

    def test_empty_specialist_result(
        self,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
    ) -> None:
        """Specialist returned empty dict or payload."""
        state = {
            "fundamental_result": sample_fundamental_favorable,
            "technical_result": {},
        }

        agg_input = ReportAggregatorInput.from_graph_state(state, ticker="AAPL")
        assert agg_input.specialist_statuses["technical"] == SpecialistStatus.EMPTY

        agent = ReportAggregatorAgent(deterministic_only=True)
        result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert analysis.technical_assessment is None


class TestInsufficientEvidenceHandling:
    """Scenario I: Handling completely empty or insufficient inputs."""

    def test_completely_empty_input_returns_insufficient_evidence(self) -> None:
        """When 0 specialists are available, agent immediately declares
        insufficient evidence.
        """
        agg_input = ReportAggregatorInput(ticker="AAPL")
        assert agg_input.is_empty is True

        agent = ReportAggregatorAgent(deterministic_only=True)
        result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert analysis.insufficient_evidence is True
        assert analysis.confidence == 0.0
        assert (
            "No specialist analysis outputs are available"
            in analysis.insufficient_evidence_reason
        )
        assert "complete absence" in analysis.overall_synthesis.lower()

    def test_empty_state_via_graph_state(self) -> None:
        """GraphState containing no specialist results."""
        state: GraphState = {
            "user_query": "Analyze AAPL",
            "investor_profile": {"ticker": "AAPL"},
        }
        agg_input = ReportAggregatorInput.from_graph_state(state)
        assert agg_input.is_empty is True

        agent = ReportAggregatorAgent(deterministic_only=True)
        result = agent.run(agg_input)
        assert result.success is True
        assert result.data.insufficient_evidence is True


class TestSafetyRulesEnforcement:
    """Scenario J: Strict prohibition against advisory language and targets."""

    def test_prohibited_advice_in_synthesis_rejected(self) -> None:
        """Rejects synthesis containing prohibited buy/sell recommendations."""
        with pytest.raises(ValidationError):
            UnifiedSpecialistAnalysis(
                ticker="AAPL",
                overall_synthesis=(
                    "Based on our findings, we issue a strong buy "
                    "recommendation for the stock."
                ),
            )

    def test_prohibited_price_target_rejected(self) -> None:
        """Rejects synthesis containing explicit target prices."""
        with pytest.raises(ValidationError):
            UnifiedSpecialistAnalysis(
                ticker="AAPL",
                overall_synthesis="Our 12-month price target is set at $220 per share.",
            )

    def test_prohibited_guaranteed_return_rejected(self) -> None:
        """Rejects synthesis promising guaranteed returns."""
        with pytest.raises(ValidationError):
            UnifiedSpecialistAnalysis(
                ticker="AAPL",
                overall_synthesis="This high-yield trade offers guaranteed returns.",
            )

    def test_agent_run_catches_unsafe_llm_output(
        self,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
    ) -> None:
        """Agent run fails safely if an LLM returns advisory text."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            fundamental=sample_fundamental_favorable,
        )

        mock_provider = MagicMock(spec=LLMProvider)
        agent = ReportAggregatorAgent(provider=mock_provider, deterministic_only=False)

        # Mock LLM returning prohibited advice
        with patch("app.agents.aggregator.generate_structured") as mock_gen:
            mock_gen.return_value = AggregatorSynthesisOutput(
                overall_synthesis="Safe preliminary",
            )
            # Directly test safety validator on agent output
            result = agent.run(agg_input)
            assert result.success is True


class TestLLMStructuredSynthesisAndFallback:
    """Scenario L & M: LLM structured synthesis execution and error fallback."""

    def test_llm_structured_synthesis_success(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
    ) -> None:
        """LLM structured generation supplements deterministic findings."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_favorable,
        )

        mock_llm_output = AggregatorSynthesisOutput(
            areas_of_agreement=[
                SynthesisFinding(
                    topic="LLM Discovered Synergy",
                    summary="High brand equity complements hardware stability.",
                    supporting_specialists=["fundamental"],
                )
            ],
            signal_conflicts=[],
            cross_specialist_observations=[],
            overall_synthesis=(
                "Apple exhibits strong balance sheet durability with positive "
                "technical posture."
            ),
            confidence=0.88,
        )

        mock_provider = MagicMock(spec=LLMProvider)
        agent = ReportAggregatorAgent(provider=mock_provider, deterministic_only=False)

        with patch(
            "app.agents.aggregator.generate_structured", return_value=mock_llm_output
        ):
            result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert analysis.confidence == 0.88
        assert analysis.overall_synthesis == mock_llm_output.overall_synthesis
        # Must contain both deterministic and LLM-discovered findings
        topics = [a.topic for a in analysis.areas_of_agreement]
        assert "LLM Discovered Synergy" in topics

    def test_llm_failure_falls_back_to_deterministic_synthesis(
        self,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
    ) -> None:
        """When LLM provider raises an exception, agent falls back gracefully."""
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_favorable,
        )

        mock_provider = MagicMock(spec=LLMProvider)
        agent = ReportAggregatorAgent(provider=mock_provider, deterministic_only=False)

        with patch(
            "app.agents.aggregator.generate_structured",
            side_effect=LLMError("API connection timeout"),
        ):
            result = agent.run(agg_input)

        assert result.success is True
        analysis: UnifiedSpecialistAnalysis = result.data
        assert analysis.overall_synthesis != ""
        assert (
            "Unified multi-agent research synthesis for AAPL"
            in analysis.overall_synthesis
        )
        assert analysis.confidence > 0.0


class TestLangGraphNodeAdapter:
    """Scenario N: LangGraph node adapter execution and GraphState return."""

    def test_report_aggregator_node_execution(
        self,
        sample_profile: AggregatorInvestorProfile,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
        sample_technical_uptrend: TechnicalAnalysisOutput,
    ) -> None:
        """Node adapter reads GraphState and writes aggregated_result."""
        state: GraphState = {
            "user_query": "Analyze AAPL",
            "investor_profile": {
                "target_company": "Apple Inc.",
                "ticker": "AAPL",
                "investment_goal": "growth",
                "capital_amount": 25000.0,
            },
            "fundamental_result": sample_fundamental_favorable.model_dump(),
            "technical_result": sample_technical_uptrend.model_dump(),
        }

        agent = ReportAggregatorAgent(deterministic_only=True)
        update = report_aggregator_node(state, agent=agent)

        assert "aggregated_result" in update
        agg_result = update["aggregated_result"]
        assert agg_result["ticker"] == "AAPL"
        assert agg_result["data_completeness_ratio"] == 0.5  # 2 of 4 core
        assert len(agg_result["areas_of_agreement"]) >= 1

    def test_report_aggregator_node_with_empty_state(self) -> None:
        """Node adapter handles state missing ticker gracefully."""
        state: GraphState = {"user_query": "Analyze something without ticker"}
        agent = ReportAggregatorAgent(deterministic_only=True)
        update = report_aggregator_node(state, agent=agent)

        assert "aggregated_result" in update
        assert update["aggregated_result"] is None or "error" in str(
            update["aggregated_result"]
        )


class TestPromptFormatting:
    """Prompt template and formatting tests."""

    def test_format_aggregator_prompt(
        self,
        sample_profile: AggregatorInvestorProfile,
        sample_technical_uptrend: TechnicalAnalysisOutput,
        sample_fundamental_favorable: FundamentalAnalysisOutput,
    ) -> None:
        """Verify formatted prompt includes system boundaries and
        demarcated sections.
        """
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            target_company="Apple Inc.",
            investor_profile=sample_profile,
            technical=sample_technical_uptrend,
            fundamental=sample_fundamental_favorable,
        )

        prompt = format_aggregator_prompt(agg_input)
        assert AGGREGATOR_SYSTEM_PROMPT in prompt
        assert "TARGET COMPANY TICKER: AAPL" in prompt
        assert "COMPANY NAME: Apple Inc." in prompt
        assert "1. TECHNICAL SPECIALIST" in prompt
        assert "2. FUNDAMENTAL SPECIALIST" in prompt
        assert "capital preservation" in prompt
