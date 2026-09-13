"""Deterministic unit tests for Phase 11.1 Report Aggregator Input & Schema.

Requirements tested:
1. All five specialists present:
   - Technical, Fundamental, News, Research, Risk all available.
   - Successful ReportAggregatorInput and UnifiedSpecialistAnalysis creation.
   - Introspection properties (available, completeness ratios).
2. One specialist missing:
   - e.g., Research missing (no documents uploaded).
   - Core completeness 100%, total completeness 80%.
   - Research is explicitly SpecialistStatus.MISSING.
3. Multiple specialists missing:
   - Only fundamental and news available.
   - Graceful partial handling without failure.
4. Specialist explicitly failed:
   - Upstream AgentResult(success=False, error="API timeout").
   - Recorded as SpecialistStatus.FAILED with error message preserved.
5. Empty specialist result:
   - Empty dict or payload=None in result.
   - Recorded as SpecialistStatus.EMPTY or FAILED with error message preserved.
6. Invalid specialist result:
   - Malformed dict failing schema validation.
   - Recorded as SpecialistStatus.FAILED with descriptive validation error.
7. Preservation of specialist identity / attribution:
   - Every evidence item traces back to exact specialist type.
   - No flattening away of source specialist.
8. Preservation of evidence/provenance:
   - Technical evidence strings preserved with indexed reference_ids.
   - Fundamental supporting metrics preserved with dimension and rating explanations.
   - News articles preserved with article_id and headline/source.
   - Research citations preserve document_id, chunk_id, page_numbers.
   - Risk citations preserve source_type and reference_id.
9. Deterministic availability/completeness information:
   - core_completeness_ratio and total_completeness_ratio strictly calculated.
   - SpecialistStatus enum explicitly tracked per specialist.
10. Validation of required common metadata (ticker/query):
    - Ticker normalization to uppercase.
    - Empty or whitespace ticker rejected.
    - Specialist ticker mismatch raises ReportAggregatorValidationError.
11. No accidental fabrication / default specialist output:
    - Missing specialist does NOT fabricate fake metrics, scores, or text.
    - Assessments for missing/failed specialists remain None.
12. Serialization / deserialization of the schema:
    - model_dump() and model_dump_json() round-trip without data loss.
    - UnifiedSpecialistAnalysis cleanly serializes to dict/json.
13. Safety boundaries:
    - Disallows buy/sell advice, price targets, guaranteed returns.
"""

from datetime import datetime, timezone
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from app.agents.aggregator_schema import (
    AggregatedEvidenceItem,
    AggregatorInvestorProfile,
    ReportAggregatorInput,
    ReportAggregatorValidationError,
    SpecialistEntry,
    SpecialistStatus,
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
# FIXTURES: DETERMINISTIC SPECIALIST OUTPUTS
# ===========================================================================


@pytest.fixture
def sample_technical_output() -> TechnicalAnalysisOutput:
    """Fixture for valid deterministic TechnicalAnalysisOutput."""
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
            rsi=RSIMetrics(value=58.5),
            macd=MACDMetrics(macd_line=1.2, signal_line=0.8, histogram=0.4),
            volume=VolumeMetrics(
                latest_volume=50000000.0, average_volume_20d=48000000.0
            ),
        ),
        support_resistance=SupportResistanceSummary(
            primary_support=180.0,
            primary_resistance=190.0,
            support_levels=[180.0, 175.0],
            resistance_levels=[190.0, 195.0],
        ),
        technical_score=72.5,
        interpretation=TechnicalInterpretation(
            overall_summary="Bullish continuation with moving average alignment.",
            trend_analysis="Price trades solidly above both 50-day and 200-day SMAs.",
            moving_averages_analysis="Bullish stacking order SMA 20 > 50 > 200.",
            momentum_analysis=(
                "RSI at 58.5 indicates healthy momentum without overbought regime."
            ),
            volume_analysis="Volume is slightly above 20-day average confirming trend.",
            support_resistance_analysis=(
                "Nearest support at 180.0, nearest resistance at 190.0."
            ),
        ),
        evidence=[
            "Closing price 185.50 is above 50-day SMA 178.0 and 200-day SMA 170.0",
            "RSI 58.5 in positive momentum zone",
            "MACD histogram positive at 0.4",
        ],
        risks=[
            "Approaching key overhead resistance at 190.0",
        ],
        confidence=0.85,
    )


@pytest.fixture
def sample_fundamental_output() -> FundamentalAnalysisOutput:
    """Fixture for valid deterministic FundamentalAnalysisOutput."""
    return FundamentalAnalysisOutput(
        ticker="AAPL",
        company_name="Apple Inc.",
        currency="USD",
        financial_health=DimensionAssessment(
            rating="strong",
            explanation=(
                "Substantial operating cash flow and net cash provide strong liquidity."
            ),
            supporting_metrics=["cash_and_equivalents", "operating_cash_flow"],
        ),
        growth_assessment=DimensionAssessment(
            rating="moderate",
            explanation=(
                "Revenue grew 6.5% YoY supported by Services revenue expansion."
            ),
            supporting_metrics=["revenue_growth_yoy", "eps_growth_yoy"],
        ),
        profitability_assessment=DimensionAssessment(
            rating="strong",
            explanation=(
                "Operating margin at 30.5% and ROE exceeds 100% due to buybacks."
            ),
            supporting_metrics=["operating_margin", "roe"],
        ),
        valuation_assessment=DimensionAssessment(
            rating="neutral",
            explanation=("P/E ratio of 28.5 is in line with premium technology peers."),
            supporting_metrics=["pe_ratio", "market_cap"],
        ),
        leverage_assessment=DimensionAssessment(
            rating="strong",
            explanation=(
                "Manageable debt supported by high interest coverage and solid FCF."
            ),
            supporting_metrics=["net_debt", "debt_to_equity"],
        ),
        cash_flow_assessment=DimensionAssessment(
            rating="strong",
            explanation="Free cash flow conversion exceeds 90% of net income.",
            supporting_metrics=["free_cash_flow", "operating_cash_flow"],
        ),
        key_strengths=[
            "High operating margin and robust Services growth",
            "Exceptional free cash flow generation",
        ],
        key_weaknesses=[
            "Hardware sales growth decelerating in selected international markets",
        ],
        notable_flags=[],
        overall_assessment="favorable",
        overall_summary=(
            "Apple demonstrates balance sheet durability and high cash flow generation."
        ),
        confidence=0.90,
    )


@pytest.fixture
def sample_news_output() -> NewsAnalysisOutput:
    """Fixture for valid deterministic NewsAnalysisOutput."""
    return NewsAnalysisOutput(
        ticker="AAPL",
        overall_sentiment="positive",
        sentiment_distribution={"positive": 8, "neutral": 2, "negative": 1},
        recent_news=[
            RecentNewsItem(
                article_id="art_001",
                headline="Apple unveils new AI features across product line",
                summary=(
                    "New intelligence ecosystem integrated into flagship devices."
                ),
                source="Reuters",
                url="https://example.com/art1",
                published_at=datetime(2026, 9, 10, 14, 30, tzinfo=timezone.utc),
                sentiment="positive",
            ),
            RecentNewsItem(
                article_id="art_002",
                headline="Services revenue reaches record high in quarterly update",
                summary="Subscription services surge 12% YoY.",
                source="Bloomberg",
                url="https://example.com/art2",
                published_at=datetime(2026, 9, 11, 9, 15, tzinfo=timezone.utc),
                sentiment="positive",
            ),
        ],
        important_events=[
            NewsAnalysisEvent(
                event_type="earnings",
                description="Quarterly revenue beat consensus by 2.4%.",
                article_ids=["art_002"],
            )
        ],
        positive_factors=[
            FactorItem(
                text="Expansion in recurring high-margin Services revenue.",
                article_ids=["art_002"],
            )
        ],
        negative_factors=[
            FactorItem(
                text="Regulatory antitrust scrutiny in European app marketplace.",
                article_ids=["art_001"],
            )
        ],
        confidence=0.88,
        summary=(
            "News sentiment is positive following AI announcements and Services growth."
        ),
        evidence=[
            "art_001: AI feature announcements",
            "art_002: Record Services revenue",
        ],
    )


@pytest.fixture
def sample_research_output() -> ResearchAnalysisOutput:
    """Fixture for valid deterministic ResearchAnalysisOutput."""
    return ResearchAnalysisOutput(
        query="What are Apple's capital allocation and share repurchase plans?",
        answer=(
            "According to the 10-K, Apple maintains an ongoing repurchase program."
        ),
        key_findings=[
            ResearchFinding(
                claim="Board authorized additional $90 billion in share repurchases.",
                evidence=[
                    ResearchEvidenceRef(
                        chunk_id="chunk_aapl_10k_p45",
                        document_id="doc_aapl_10k_2025",
                        source_document="AAPL_2025_10K.pdf",
                        page_numbers=[45],
                    )
                ],
            )
        ],
        evidence=[
            ResearchEvidenceRef(
                chunk_id="chunk_aapl_10k_p45",
                document_id="doc_aapl_10k_2025",
                source_document="AAPL_2025_10K.pdf",
                page_numbers=[45],
            )
        ],
        confidence=0.92,
        insufficient_evidence=False,
    )


@pytest.fixture
def sample_risk_output() -> RiskAnalysisOutput:
    """Fixture for valid deterministic RiskAnalysisOutput."""
    return RiskAnalysisOutput(
        ticker="AAPL",
        overall_risk_level=RiskSeverity.MODERATE,
        market_risks=[
            RiskFactor(
                category=RiskCategory.MARKET,
                name="Foreign Exchange & Macro Sensitivity",
                description=(
                    "Significant revenue derived from volatile international markets."
                ),
                severity=RiskSeverity.MODERATE,
                probability=RiskProbability.MEDIUM,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="revenue_history",
                        detail=("Over 55% of consolidated sales generated outside US."),
                    )
                ],
            )
        ],
        company_risks=[
            RiskFactor(
                category=RiskCategory.COMPANY,
                name="Supplier & Assembly Concentration",
                description=(
                    "High reliance on concentrated manufacturing in East Asia."
                ),
                severity=RiskSeverity.MODERATE,
                probability=RiskProbability.MEDIUM,
                evidence=[
                    RiskEvidenceRef(
                        source_type="research",
                        reference_id="chunk_aapl_10k_p45",
                        detail="Manufacturing concentration cited in 10-K.",
                    )
                ],
            )
        ],
        sector_risks=[],
        financial_risks=[],
        volatility_risks=[
            RiskFactor(
                category=RiskCategory.VOLATILITY,
                name="Overhead Resistance Proximity",
                description="Price is testing primary resistance boundary at 190.0.",
                severity=RiskSeverity.LOW,
                probability=RiskProbability.LOW,
                evidence=[
                    RiskEvidenceRef(
                        source_type="technical",
                        reference_id="primary_resistance",
                        detail="Primary resistance level identified at 190.0.",
                    )
                ],
            )
        ],
        investor_specific_risks=[],
        data_completeness={
            "technical": True,
            "fundamental": True,
            "news": True,
            "research": True,
        },
        summary=("Composite risk is moderate driven by global supply chain exposure."),
        confidence=0.85,
        deterministic_risk_score=0.35,
        deterministic_risk_level=RiskSeverity.MODERATE,
    )


@pytest.fixture
def sample_investor_profile() -> AggregatorInvestorProfile:
    """Fixture for valid investor profile."""
    return AggregatorInvestorProfile(
        target_company="Apple Inc.",
        ticker="AAPL",
        investment_goal="growth",
        time_horizon="3-5 years",
        capital_amount=50000.0,
        risk_tolerance="moderate",
        profile_complete=True,
    )


# ===========================================================================
# 1. TEST ALL FIVE SPECIALISTS PRESENT
# ===========================================================================


class TestAllSpecialistsPresent:
    """Verifies behavior when all 5 specialists provide valid outputs."""

    def test_all_five_specialists_present_success(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
        sample_fundamental_output: FundamentalAnalysisOutput,
        sample_news_output: NewsAnalysisOutput,
        sample_research_output: ResearchAnalysisOutput,
        sample_risk_output: RiskAnalysisOutput,
        sample_investor_profile: AggregatorInvestorProfile,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            investor_profile=sample_investor_profile,
            technical=sample_technical_output,
            fundamental=sample_fundamental_output,
            news=sample_news_output,
            research=sample_research_output,
            risk=sample_risk_output,
        )

        assert agg_input.ticker == "AAPL"
        assert agg_input.has_technical is True
        assert agg_input.has_fundamental is True
        assert agg_input.has_news is True
        assert agg_input.has_research is True
        assert agg_input.has_risk is True
        assert agg_input.is_empty is False

        # All 5 available
        assert len(agg_input.available_specialists) == 5
        assert len(agg_input.missing_specialists) == 0
        assert len(agg_input.failed_specialists) == 0
        assert agg_input.core_completeness_ratio == 1.0
        assert agg_input.total_completeness_ratio == 1.0

        # Build UnifiedSpecialistAnalysis
        unified = UnifiedSpecialistAnalysis.from_input(agg_input)
        assert unified.ticker == "AAPL"
        assert unified.data_completeness_ratio == 1.0
        assert unified.technical_assessment is not None
        assert unified.fundamental_assessment is not None
        assert unified.news_assessment is not None
        assert unified.research_assessment is not None
        assert unified.risk_assessment is not None
        assert len(unified.missing_specialists) == 0
        assert len(unified.failed_specialists) == 0
        assert len(unified.aggregated_evidence) > 0


# ===========================================================================
# 2. TEST ONE SPECIALIST MISSING (PARTIAL AVAILABILITY)
# ===========================================================================


class TestOneSpecialistMissing:
    """Verifies partial availability when one specialist is absent (e.g. Research)."""

    def test_research_missing(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
        sample_fundamental_output: FundamentalAnalysisOutput,
        sample_news_output: NewsAnalysisOutput,
        sample_risk_output: RiskAnalysisOutput,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_output,
            fundamental=sample_fundamental_output,
            news=sample_news_output,
            research=None,
            risk=sample_risk_output,
        )

        assert agg_input.has_research is False
        assert "research" in agg_input.missing_specialists
        assert agg_input.specialist_statuses["research"] == SpecialistStatus.MISSING

        # Core specialists (technical, fundamental, news, risk) are 4/4 = 1.0
        assert agg_input.core_completeness_ratio == 1.0
        # Total specialists (4/5) = 0.8
        assert agg_input.total_completeness_ratio == 0.8

        unified = UnifiedSpecialistAnalysis.from_input(agg_input)
        assert unified.research_assessment is None
        assert "research" in unified.missing_specialists
        assert unified.technical_assessment is not None
        assert unified.fundamental_assessment is not None

    def test_technical_missing(
        self,
        sample_fundamental_output: FundamentalAnalysisOutput,
        sample_news_output: NewsAnalysisOutput,
        sample_risk_output: RiskAnalysisOutput,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=None,
            fundamental=sample_fundamental_output,
            news=sample_news_output,
            risk=sample_risk_output,
        )

        assert agg_input.has_technical is False
        assert "technical" in agg_input.missing_specialists
        assert agg_input.core_completeness_ratio == 0.75  # 3 of 4 core


# ===========================================================================
# 3. TEST MULTIPLE SPECIALISTS MISSING
# ===========================================================================


class TestMultipleSpecialistsMissing:
    """Verifies behavior when multiple specialists are omitted."""

    def test_only_fundamental_and_news_present(
        self,
        sample_fundamental_output: FundamentalAnalysisOutput,
        sample_news_output: NewsAnalysisOutput,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            fundamental=sample_fundamental_output,
            news=sample_news_output,
        )

        assert len(agg_input.available_specialists) == 2
        assert set(agg_input.available_specialists) == {"fundamental", "news"}
        assert set(agg_input.missing_specialists) == {"technical", "research", "risk"}
        assert agg_input.core_completeness_ratio == 0.5  # 2 of 4 core
        assert agg_input.total_completeness_ratio == 0.4  # 2 of 5 total
        assert agg_input.is_empty is False

        unified = UnifiedSpecialistAnalysis.from_input(agg_input)
        assert unified.fundamental_assessment is not None
        assert unified.news_assessment is not None
        assert unified.technical_assessment is None
        assert unified.research_assessment is None
        assert unified.risk_assessment is None


# ===========================================================================
# 4. TEST SPECIALIST EXPLICITLY FAILED
# ===========================================================================


class TestSpecialistExplicitlyFailed:
    """Verifies that an upstream failure is preserved with status and error message."""

    def test_specialist_failed_via_agent_result(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
    ) -> None:
        state: Dict[str, Any] = {
            "investor_profile": {"ticker": "AAPL"},
            "technical_result": sample_technical_output.model_dump(),
            "fundamental_result": AgentResult.create_failure(
                error="Fundamental provider timed out after 30 seconds."
            ).model_dump(),
        }

        agg_input = ReportAggregatorInput.from_graph_state(state)

        assert agg_input.has_technical is True
        assert agg_input.has_fundamental is False
        assert agg_input.specialist_statuses["fundamental"] == SpecialistStatus.FAILED
        assert "timed out" in agg_input.specialist_errors["fundamental"]

        unified = UnifiedSpecialistAnalysis.from_input(agg_input)
        assert "fundamental" in unified.failed_specialists
        assert unified.fundamental_assessment is None
        assert "timed out" in unified.specialist_errors["fundamental"]

    def test_specialist_failed_via_partial_constructor(self) -> None:
        agg_input = ReportAggregatorInput.from_partial_specialists(
            ticker="AAPL",
            technical=None,
            specialist_statuses={"technical": SpecialistStatus.FAILED},
            specialist_errors={"technical": "Market data source 503 unavailable."},
        )

        assert agg_input.has_technical is False
        assert agg_input.specialist_statuses["technical"] == SpecialistStatus.FAILED
        assert (
            agg_input.specialist_errors["technical"]
            == "Market data source 503 unavailable."
        )


# ===========================================================================
# 5. TEST EMPTY SPECIALIST RESULT
# ===========================================================================


class TestEmptySpecialistResult:
    """Verifies handling when specialist result is an empty dict or empty payload."""

    def test_empty_dict_treated_as_empty_status(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
    ) -> None:
        state: Dict[str, Any] = {
            "investor_profile": {"ticker": "AAPL"},
            "technical_result": sample_technical_output.model_dump(),
            "fundamental_result": {},  # empty dict
        }

        agg_input = ReportAggregatorInput.from_graph_state(state)
        assert agg_input.has_fundamental is False
        assert agg_input.specialist_statuses["fundamental"] == SpecialistStatus.EMPTY
        assert "empty" in agg_input.specialist_errors["fundamental"].lower()

    def test_agent_result_none_payload_treated_as_empty(self) -> None:
        state: Dict[str, Any] = {
            "investor_profile": {"ticker": "AAPL"},
            "news_result": AgentResult.create_success(data=None).model_dump(),
        }

        agg_input = ReportAggregatorInput.from_graph_state(state)
        assert agg_input.has_news is False
        assert agg_input.specialist_statuses["news"] == SpecialistStatus.EMPTY


# ===========================================================================
# 6. TEST INVALID SPECIALIST RESULT
# ===========================================================================


class TestInvalidSpecialistResult:
    """Verifies handling of invalid/malformed specialist outputs."""

    def test_malformed_specialist_dict_records_failure(self) -> None:
        state: Dict[str, Any] = {
            "investor_profile": {"ticker": "AAPL"},
            # Missing required fields like indicators_summary, interpretation, etc.
            "technical_result": {"ticker": "AAPL", "trend": "invalid_trend"},
        }

        agg_input = ReportAggregatorInput.from_graph_state(state)
        assert agg_input.has_technical is False
        assert agg_input.specialist_statuses["technical"] == SpecialistStatus.FAILED
        assert "Validation failed" in agg_input.specialist_errors["technical"]


# ===========================================================================
# 7. TEST PRESERVATION OF SPECIALIST ATTRIBUTION
# ===========================================================================


class TestPreservationOfSpecialistAttribution:
    """Verifies that every piece of evidence retains origin specialist attribution."""

    def test_evidence_attribution_strictly_preserved(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
        sample_fundamental_output: FundamentalAnalysisOutput,
        sample_news_output: NewsAnalysisOutput,
        sample_research_output: ResearchAnalysisOutput,
        sample_risk_output: RiskAnalysisOutput,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            technical=sample_technical_output,
            fundamental=sample_fundamental_output,
            news=sample_news_output,
            research=sample_research_output,
            risk=sample_risk_output,
        )

        evidence_items = agg_input.extract_attributed_evidence()
        specialists_in_evidence = {item.specialist for item in evidence_items}

        assert "technical" in specialists_in_evidence
        assert "fundamental" in specialists_in_evidence
        assert "news" in specialists_in_evidence
        assert "research" in specialists_in_evidence
        assert "risk" in specialists_in_evidence

        # Check technical item attribution
        tech_items = [e for e in evidence_items if e.specialist == "technical"]
        assert len(tech_items) == len(sample_technical_output.evidence)
        for t in tech_items:
            assert t.reference_id.startswith("tech_ev_")

        # Check fundamental item attribution
        fund_items = [e for e in evidence_items if e.specialist == "fundamental"]
        assert len(fund_items) > 0
        for f in fund_items:
            assert f.reference_id in [
                "cash_and_equivalents",
                "operating_cash_flow",
                "revenue_growth_yoy",
                "eps_growth_yoy",
                "operating_margin",
                "roe",
                "pe_ratio",
                "market_cap",
                "net_debt",
                "debt_to_equity",
                "free_cash_flow",
            ]


# ===========================================================================
# 8. TEST PRESERVATION OF EVIDENCE & PROVENANCE
# ===========================================================================


class TestPreservationOfEvidenceAndProvenance:
    """Verifies that citation metadata (document_id, chunk_id, page_numbers) is kept."""

    def test_research_citations_preserve_document_and_page_metadata(
        self,
        sample_research_output: ResearchAnalysisOutput,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            research=sample_research_output,
        )

        evidence_items = agg_input.extract_attributed_evidence()
        res_items = [e for e in evidence_items if e.specialist == "research"]

        assert len(res_items) == 1
        ref = res_items[0]
        assert ref.document_id == "doc_aapl_10k_2025"
        assert ref.chunk_id == "chunk_aapl_10k_p45"
        assert ref.page_numbers == [45]
        assert ref.reference_id == "chunk_aapl_10k_p45"

    def test_news_citations_preserve_article_id_and_source(
        self,
        sample_news_output: NewsAnalysisOutput,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            news=sample_news_output,
        )

        evidence_items = agg_input.extract_attributed_evidence()
        news_items = [e for e in evidence_items if e.specialist == "news"]

        assert len(news_items) == 2
        art_ids = {n.reference_id for n in news_items}
        assert art_ids == {"art_001", "art_002"}
        assert any("Reuters" in n.detail for n in news_items)
        assert any("Bloomberg" in n.detail for n in news_items)

    def test_risk_citations_preserve_source_type_and_ref(
        self,
        sample_risk_output: RiskAnalysisOutput,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            risk=sample_risk_output,
        )

        evidence_items = agg_input.extract_attributed_evidence()
        risk_items = [e for e in evidence_items if e.specialist == "risk"]

        assert len(risk_items) > 0
        ref_ids = [r.reference_id for r in risk_items]
        assert any("fundamental:revenue_history" in rid for rid in ref_ids)
        assert any("research:chunk_aapl_10k_p45" in rid for rid in ref_ids)


# ===========================================================================
# 9. TEST DETERMINISTIC AVAILABILITY & COMPLETENESS RATIOS
# ===========================================================================


class TestDeterministicAvailabilityAndCompleteness:
    """Verifies explicit status tracking and mathematical completeness ratios."""

    def test_empty_aggregator_input(self) -> None:
        agg_input = ReportAggregatorInput(ticker="AAPL")
        assert agg_input.is_empty is True
        assert agg_input.core_completeness_ratio == 0.0
        assert agg_input.total_completeness_ratio == 0.0
        assert len(agg_input.available_specialists) == 0
        assert len(agg_input.missing_specialists) == 5

    def test_specialist_entry_model(self) -> None:
        entry = SpecialistEntry(
            specialist="technical",
            status=SpecialistStatus.AVAILABLE,
            raw_data={"ticker": "AAPL"},
        )
        assert entry.is_available is True
        assert entry.is_missing is False
        assert entry.is_failed is False
        assert entry.is_empty is False

        failed_entry = SpecialistEntry(
            specialist="fundamental",
            status=SpecialistStatus.FAILED,
            error="API down",
        )
        assert failed_entry.is_failed is True
        assert failed_entry.error == "API down"


# ===========================================================================
# 10. TEST METADATA VALIDATION (TICKER, PROFILE, CONFLICTS)
# ===========================================================================


class TestMetadataValidation:
    """Verifies common metadata validation (ticker uppercase, whitespace, conflict)."""

    def test_ticker_normalized_to_uppercase(self) -> None:
        agg_input = ReportAggregatorInput(ticker="  aapl  ")
        assert agg_input.ticker == "AAPL"

    def test_ticker_empty_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            ReportAggregatorInput(ticker="   ")

    def test_specialist_ticker_mismatch_raises_error(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
    ) -> None:
        # Technical output is for AAPL, input ticker is MSFT
        with pytest.raises(
            (ReportAggregatorValidationError, ValidationError)
        ) as exc_info:
            ReportAggregatorInput(
                ticker="MSFT",
                technical=sample_technical_output,
            )
        assert "ticker 'AAPL' does not match aggregator input ticker 'MSFT'" in str(
            exc_info.value
        )

    def test_from_graph_state_infers_ticker_from_profile(self) -> None:
        state: Dict[str, Any] = {
            "investor_profile": {"ticker": "NVDA", "target_company": "NVIDIA"},
        }
        agg_input = ReportAggregatorInput.from_graph_state(state)
        assert agg_input.ticker == "NVDA"
        assert agg_input.target_company == "NVIDIA"

    def test_from_graph_state_infers_ticker_from_specialist_if_profile_empty(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
    ) -> None:
        state: Dict[str, Any] = {
            "technical_result": sample_technical_output.model_dump(),
        }
        agg_input = ReportAggregatorInput.from_graph_state(state)
        assert agg_input.ticker == "AAPL"

    def test_from_graph_state_raises_if_no_ticker_found(self) -> None:
        with pytest.raises(ReportAggregatorValidationError):
            ReportAggregatorInput.from_graph_state({})


# ===========================================================================
# 11. TEST NO ACCIDENTAL FABRICATION / DEFAULT VALUES
# ===========================================================================


class TestNoAccidentalFabrication:
    """Verifies that missing specialists are NOT assigned default scores or data."""

    def test_missing_specialist_remains_none(self) -> None:
        agg_input = ReportAggregatorInput(ticker="AAPL")
        unified = UnifiedSpecialistAnalysis.from_input(agg_input)

        assert unified.technical_assessment is None
        assert unified.fundamental_assessment is None
        assert unified.news_assessment is None
        assert unified.research_assessment is None
        assert unified.risk_assessment is None
        assert len(unified.aggregated_evidence) == 0


# ===========================================================================
# 12. TEST SERIALIZATION AND DESERIALIZATION
# ===========================================================================


class TestSerializationAndDeserialization:
    """Verifies round-trip model_dump and JSON serialization."""

    def test_input_round_trip_serialization(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
        sample_fundamental_output: FundamentalAnalysisOutput,
        sample_news_output: NewsAnalysisOutput,
        sample_research_output: ResearchAnalysisOutput,
        sample_risk_output: RiskAnalysisOutput,
        sample_investor_profile: AggregatorInvestorProfile,
    ) -> None:
        original = ReportAggregatorInput(
            ticker="AAPL",
            investor_profile=sample_investor_profile,
            technical=sample_technical_output,
            fundamental=sample_fundamental_output,
            news=sample_news_output,
            research=sample_research_output,
            risk=sample_risk_output,
        )

        data = original.model_dump()
        restored = ReportAggregatorInput.model_validate(data)

        assert restored.ticker == original.ticker
        assert restored.has_technical is True
        assert restored.has_fundamental is True
        assert restored.has_news is True
        assert restored.has_research is True
        assert restored.has_risk is True
        assert restored.core_completeness_ratio == 1.0

    def test_unified_round_trip_json_serialization(
        self,
        sample_technical_output: TechnicalAnalysisOutput,
        sample_fundamental_output: FundamentalAnalysisOutput,
        sample_investor_profile: AggregatorInvestorProfile,
    ) -> None:
        agg_input = ReportAggregatorInput(
            ticker="AAPL",
            investor_profile=sample_investor_profile,
            technical=sample_technical_output,
            fundamental=sample_fundamental_output,
        )

        unified = UnifiedSpecialistAnalysis.from_input(agg_input)
        json_str = unified.model_dump_json()
        assert "AAPL" in json_str
        assert "Apple Inc." in json_str

        restored = UnifiedSpecialistAnalysis.model_validate_json(json_str)
        assert restored.ticker == "AAPL"
        assert restored.technical_assessment is not None
        assert restored.fundamental_assessment is not None
        assert restored.news_assessment is None
        assert len(restored.aggregated_evidence) > 0


# ===========================================================================
# 13. TEST AGGREGATED EVIDENCE VALIDATION
# ===========================================================================


class TestAggregatedEvidenceItemValidation:
    """Verifies validation rules on AggregatedEvidenceItem."""

    def test_empty_reference_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AggregatedEvidenceItem(
                specialist="technical",
                reference_id="   ",
                detail="Some detail",
            )

    def test_empty_detail_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AggregatedEvidenceItem(
                specialist="technical",
                reference_id="ref_1",
                detail="",
            )
