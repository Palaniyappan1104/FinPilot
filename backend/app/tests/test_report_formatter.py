"""Comprehensive tests for Report Formatter and Output Formats (Phase 12.4).

Fulfills Phase 12.4 test requirements:
1. Markdown rendering (format_report_markdown, report.to_markdown).
2. Plain text rendering (format_report_text, report.to_text).
3. Structured JSON serialization and round-trip validity (format_report_json).
4. Frontend UI dictionary and JSON formatting (format_report_frontend_dict,
   format_report_frontend_json).
5. Handling of partial reports, missing/failed specialists, and edge cases.
6. Safety and disclaimer integrity across all output formats.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.agents.aggregator_schema import (
    AggregatedEvidenceItem,
    AggregatorInvestorProfile,
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
from app.agents.report_formatter import (
    format_report_frontend_dict,
    format_report_frontend_json,
    format_report_json,
    format_report_markdown,
    format_report_text,
)
from app.agents.report_generator import ReportGeneratorAgent
from app.agents.report_schema import (
    FinalReport,
    OverallAssessmentSection,
    RecommendationStance,
    ReportCompanyInfo,
    ReportRecommendation,
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
from app.agents.technical_schema import (
    MACDMetrics,
    MovingAverageMetrics,
    RSIMetrics,
    SupportResistanceSummary,
    TechnicalAnalysisOutput,
    TechnicalIndicatorsSummary,
    TechnicalInterpretation,
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
        investment_goal="growth",
        risk_tolerance="moderate",
        time_horizon="3-5 years",
        capital_amount=50000.0,
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
            trend_analysis="Uptrend confirmed above 50-day SMA.",
            moving_averages_analysis="Moving averages are stacked bullishly.",
            momentum_analysis="RSI at 62.0 indicates healthy positive momentum.",
            volume_analysis="Volume confirms breakout over resistance.",
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
            explanation="Gross margin remains at 45.0% with operating leverage.",
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
                headline="Apple launches next-gen silicon",
                summary="Services ecosystem expansion and AI hardware launch.",
                source="Reuters",
                published_at=datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc),
                sentiment="positive",
                url="https://example.com/art1",
            )
        ],
        important_events=[
            NewsAnalysisEvent(
                event_type="earnings",
                description="AI Platform Announcement",
                article_ids=["art_1"],
            )
        ],
        positive_factors=[
            FactorItem(
                text="Strong services revenue",
                article_ids=["art_1"],
            )
        ],
        negative_factors=[
            FactorItem(
                text="China smartphone headwinds",
                article_ids=["art_1"],
            )
        ],
        summary="Positive coverage around AI hardware launch and ecosystem services.",
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
                document_id="AAPL_10K.pdf",
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


@pytest.fixture
def sample_valid_report(
    sample_unified_analysis: UnifiedSpecialistAnalysis,
) -> FinalReport:
    agent = ReportGeneratorAgent(deterministic_only=True)
    res = agent.run(sample_unified_analysis)
    assert res.data is not None
    return res.data


# ===========================================================================
# 1. MARKDOWN RENDERING TESTS (12.4.2)
# ===========================================================================


class TestReportMarkdownFormatting:
    """Test suite for Markdown formatting (format_report_markdown)."""

    def test_markdown_contains_header_and_metadata(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Markdown output includes company name, ticker, and metadata header."""
        md = format_report_markdown(sample_valid_report)
        assert "# Investment Research Report: AAPL (Apple Inc.)" in md
        assert "**Generated:**" in md
        assert "**Currency:** USD" in md
        assert "**Data Completeness:** 100%" in md

    def test_markdown_contains_investor_context(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Markdown output includes structured investor context."""
        md = format_report_markdown(sample_valid_report)
        assert "## 1. Investor Context" in md
        assert "- **Target Profile:** growth" in md
        assert "- **Risk Tolerance:** moderate" in md
        assert "- **Investment Horizon:** 3-5 years" in md
        assert "- **Allocated Capital:** $50,000.00" in md

    def test_markdown_contains_recommendation_and_rationale(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Markdown output highlights recommendation stance, badge, and rationale."""
        md = format_report_markdown(sample_valid_report)
        assert "## 2. Executive Summary & Recommendation" in md
        assert "### Recommendation:" in md
        assert "FAVORABLE" in md
        assert "> **Strategic Rationale:**" in md
        assert "**Investor Profile Alignment:**" in md
        assert "### Key Supporting Reasons" in md

    def test_markdown_contains_all_specialist_sections(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Markdown output details all 5 specialist analysis sections."""
        md = format_report_markdown(sample_valid_report)
        assert "## 3. Specialist Analysis" in md
        assert "### Technical Analysis" in md
        assert "### Fundamental Analysis" in md
        assert "### News & Sentiment" in md
        assert "### SEC & Document Research" in md
        assert "### Risk Analysis" in md

    def test_markdown_contains_evidence_provenance_table(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Markdown output includes structured evidence table with references."""
        md = format_report_markdown(sample_valid_report)
        assert "## 5. Evidence & Provenance" in md
        assert "| Ref ID | Specialist | Document / Chunk | Detail |" in md
        assert "`tech_ev_0`" in md
        assert "`gross_margin`" in md

    def test_markdown_contains_mandatory_disclaimer(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Markdown output includes mandatory regulatory disclaimer by default."""
        md = format_report_markdown(sample_valid_report)
        assert "## 6. Regulatory Disclaimer" in md
        assert "not a registered investment advisor" in md.lower()
        assert "informational purposes only" in md.lower()

    def test_markdown_omits_disclaimer_when_flag_false(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Markdown output omits disclaimer when include_disclaimer=False."""
        md = format_report_markdown(sample_valid_report, include_disclaimer=False)
        assert "## 6. Regulatory Disclaimer" not in md

    def test_markdown_includes_table_of_contents_when_requested(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Markdown output generates TOC when include_table_of_contents=True."""
        md = format_report_markdown(sample_valid_report, include_table_of_contents=True)
        assert "## Table of Contents" in md
        assert "- [1. Investor Context](#1-investor-context)" in md
        assert (
            "- [2. Executive Summary & Recommendation]"
            "(#2-executive-summary--recommendation)" in md
        )

    def test_report_to_markdown_method_parity(
        self, sample_valid_report: FinalReport
    ) -> None:
        """report.to_markdown() matches format_report_markdown(report)."""
        direct_md = format_report_markdown(sample_valid_report)
        method_md = sample_valid_report.to_markdown()
        assert direct_md == method_md


# ===========================================================================
# 2. PLAIN TEXT RENDERING TESTS (12.4.2)
# ===========================================================================


class TestReportTextFormatting:
    """Test suite for Plain Text formatting (format_report_text, report.to_text)."""

    def test_text_contains_dividers_and_header(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Plain text formatting includes ASCII divider lines and clear header."""
        txt = format_report_text(sample_valid_report)
        assert "FINPILOT INVESTMENT RESEARCH REPORT: AAPL (Apple Inc.)" in txt
        assert (
            "=============================================================================="
            in txt
        )
        assert "Generated:" in txt
        assert "Completeness: 100%" in txt

    def test_text_contains_all_core_sections(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Plain text formatting includes all numbered sections."""
        txt = format_report_text(sample_valid_report)
        assert "1. INVESTOR CONTEXT" in txt
        assert "2. RECOMMENDATION & EXECUTIVE SUMMARY" in txt
        assert "3. SPECIALIST ANALYSIS BREAKDOWN" in txt
        assert "  [TECHNICAL ANALYSIS]" in txt
        assert "  [FUNDAMENTAL ANALYSIS]" in txt
        assert "  [NEWS & SENTIMENT]" in txt
        assert "  [SEC & RESEARCH]" in txt
        assert "  [RISK ASSESSMENT]" in txt
        assert "4. KEY INVESTMENT RISKS" in txt
        assert "5. EVIDENCE SOURCES" in txt
        assert "REGULATORY DISCLAIMER:" in txt

    def test_text_disclaimer_omission(self, sample_valid_report: FinalReport) -> None:
        """Plain text output omits disclaimer when include_disclaimer=False."""
        txt = format_report_text(sample_valid_report, include_disclaimer=False)
        assert "REGULATORY DISCLAIMER:" not in txt

    def test_report_to_text_method_parity(
        self, sample_valid_report: FinalReport
    ) -> None:
        """report.to_text() matches format_report_text(report)."""
        assert sample_valid_report.to_text() == format_report_text(sample_valid_report)


# ===========================================================================
# 3. STRUCTURED JSON & ROUND-TRIP TESTS (12.4.1)
# ===========================================================================


class TestReportJSONFormatting:
    """Test suite for Structured JSON serialization and round-trip integrity."""

    def test_json_validity_and_round_trip(
        self, sample_valid_report: FinalReport
    ) -> None:
        """JSON output is parseable and round-trips via model_validate_json()."""
        json_str = format_report_json(sample_valid_report)
        assert isinstance(json_str, str)

        parsed_dict = json.loads(json_str)
        assert parsed_dict["company"]["ticker"] == "AAPL"
        assert parsed_dict["recommendation"]["stance"] == "favorable"

        # Round-trip validation
        reconstructed = FinalReport.model_validate_json(json_str)
        assert reconstructed.company.ticker == sample_valid_report.company.ticker
        assert (
            reconstructed.recommendation.stance
            == sample_valid_report.recommendation.stance
        )
        assert reconstructed.is_complete is True
        assert (
            reconstructed.overall_assessment.data_completeness_ratio
            == sample_valid_report.overall_assessment.data_completeness_ratio
        )
        assert len(reconstructed.evidence_sources) == len(
            sample_valid_report.evidence_sources
        )

    def test_json_indent_options(self, sample_valid_report: FinalReport) -> None:
        """Indentation options produce compact vs pretty-printed JSON."""
        compact = format_report_json(sample_valid_report, indent=None)
        pretty = format_report_json(sample_valid_report, indent=4)
        assert "\n" not in compact
        assert "\n" in pretty


# ===========================================================================
# 4. FRONTEND DICTIONARY & UI PAYLOAD TESTS (12.4.1)
# ===========================================================================


class TestReportFrontendFormatting:
    """Test suite for frontend UI dictionary and JSON payload formatting."""

    def test_frontend_dict_structure(self, sample_valid_report: FinalReport) -> None:
        """Frontend dict includes all expected keys and card-ready sub-structures."""
        ui_data = format_report_frontend_dict(sample_valid_report)
        assert isinstance(ui_data, dict)

        expected_top_keys = {
            "meta",
            "investor_context",
            "recommendation",
            "executive_summary",
            "key_reasons",
            "important_risks",
            "specialist_breakdowns",
            "evidence_sources",
            "disclaimer",
            "rendered_markdown",
        }
        assert expected_top_keys.issubset(ui_data.keys())

        # Check badge formatting
        assert ui_data["recommendation"]["stance"] == "favorable"
        assert ui_data["recommendation"]["badge"] == "FAVORABLE"
        assert ui_data["recommendation"]["badge_color"] == "green"
        assert ui_data["recommendation"]["ui_variant"] == "success"

        # Check specialist breakdowns
        breakdowns = ui_data["specialist_breakdowns"]
        for spec in ("technical", "fundamental", "news", "research", "risk"):
            assert spec in breakdowns
            assert breakdowns[spec]["is_available"] is True
            assert breakdowns[spec]["status"] == "available"
            assert breakdowns[spec]["data"] is not None

    @pytest.mark.parametrize(
        "stance,expected_color,expected_variant",
        [
            (RecommendationStance.FAVORABLE, "green", "success"),
            (RecommendationStance.NEUTRAL, "amber", "warning"),
            (RecommendationStance.CAUTIOUS, "orange", "warning"),
            (RecommendationStance.UNFAVORABLE, "red", "destructive"),
            (RecommendationStance.INSUFFICIENT_EVIDENCE, "gray", "secondary"),
        ],
    )
    def test_frontend_stance_color_and_badge_mapping(
        self,
        sample_valid_report: FinalReport,
        stance: RecommendationStance,
        expected_color: str,
        expected_variant: str,
    ) -> None:
        """Every recommendation stance maps to the correct UI color and variant."""
        report = sample_valid_report.model_copy(deep=True)
        rec = report.recommendation.model_copy(update={"stance": stance})
        object.__setattr__(report, "recommendation", rec)

        ui_data = format_report_frontend_dict(report)
        assert ui_data["recommendation"]["badge_color"] == expected_color
        assert ui_data["recommendation"]["ui_variant"] == expected_variant

    def test_frontend_json_validity(self, sample_valid_report: FinalReport) -> None:
        """format_report_frontend_json generates valid parseable JSON."""
        json_str = format_report_frontend_json(sample_valid_report)
        parsed = json.loads(json_str)
        assert parsed["meta"]["ticker"] == "AAPL"
        assert parsed["recommendation"]["badge_color"] == "green"
        assert isinstance(parsed["rendered_markdown"], str)

    def test_report_to_frontend_methods_parity(
        self, sample_valid_report: FinalReport
    ) -> None:
        """report.to_frontend_dict() and to_frontend_json() match direct calls."""
        assert sample_valid_report.to_frontend_dict() == format_report_frontend_dict(
            sample_valid_report
        )
        assert sample_valid_report.to_frontend_json() == format_report_frontend_json(
            sample_valid_report
        )


# ===========================================================================
# 5. PARTIAL REPORTS & EDGE CASES (12.4.1 & 12.4.2)
# ===========================================================================


class TestReportPartialAndEdgeCaseFormatting:
    """Test suite for partial reports, missing specialists, and minimal data."""

    def test_partial_report_markdown_and_text(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Partial report with missing news displays warning banners in all formats."""
        report = sample_valid_report.model_copy(deep=True)
        object.__setattr__(report, "news", None)
        object.__setattr__(report, "missing_specialists", ["news"])
        object.__setattr__(report.overall_assessment, "data_completeness_ratio", 0.8)
        statuses = dict(report.specialist_statuses)
        statuses["news"] = SpecialistStatus.MISSING
        object.__setattr__(report, "specialist_statuses", statuses)

        md = format_report_markdown(report)
        assert "⚠️ **Notice: Partial Report**" in md
        assert "> ⚠️ **News & Sentiment Unavailable** (Status: MISSING)" in md

        txt = format_report_text(report)
        assert "Status: PARTIAL" in txt
        assert "[NEWS & SENTIMENT]" in txt
        assert "Status:    UNAVAILABLE" in txt

        ui_data = format_report_frontend_dict(report)
        assert ui_data["meta"]["is_complete"] is False
        assert ui_data["specialist_breakdowns"]["news"]["is_available"] is False
        assert ui_data["specialist_breakdowns"]["news"]["status"] == "missing"
        assert ui_data["specialist_breakdowns"]["news"]["data"] is None

    def test_minimal_report_formatting(self) -> None:
        """Report with minimal optional fields renders without exceptions."""
        minimal_report = FinalReport(
            company=ReportCompanyInfo(ticker="TSLA", currency="USD"),
            overall_assessment=OverallAssessmentSection(
                synthesis="Tesla assessment without specialists.",
                confidence=0.5,
                data_completeness_ratio=0.0,
            ),
            recommendation=ReportRecommendation(
                stance=RecommendationStance.NEUTRAL,
                rationale="Neutral stance based on limited available inputs.",
            ),
            key_reasons=["Limited market data."],
            important_risks=["High volatility."],
            evidence_sources=[],
            specialist_statuses={},
            disclaimer=(
                "FinPilot provides automated financial research for informational "
                "purposes only. Not a registered investment advisor. Does not "
                "constitute advice. Risk of loss."
            ),
        )

        # Markdown
        md = format_report_markdown(minimal_report)
        assert "# Investment Research Report: TSLA" in md
        assert "NEUTRAL" in md

        # Plain Text
        txt = format_report_text(minimal_report)
        assert "FINPILOT INVESTMENT RESEARCH REPORT: TSLA" in txt
        assert "[NEUTRAL]" in txt

        # Frontend JSON
        ui_json = format_report_frontend_json(minimal_report)
        parsed = json.loads(ui_json)
        assert parsed["meta"]["ticker"] == "TSLA"
        assert parsed["meta"]["company_name"] is None
        assert parsed["investor_context"]["target_profile"] is None
