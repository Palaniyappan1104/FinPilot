"""Comprehensive unit and consistency tests for Report Consistency Validator.

Phase 12.3 requirements:
1. Fully valid report passes.
2. Invalid report schema / input fails.
3. Ticker mismatch detected.
4. Company name mismatch detected.
5. Investor profile mismatch detected.
6. Horizon mismatch detected.
7. Capital mismatch detected.
8. Missing specialist represented as available detected.
9. Failed specialist represented as successful detected.
10. Unsupported numerical value detected.
11. Correct numerical provenance accepted.
12. Phantom source reference detected.
13. Valid source reference accepted.
14. Evidence attribution mismatch detected.
15. Specialist disagreement accepted as legitimate.
16. Unsupported contradiction detected.
17. Recommendation unsupported by evidence detected.
18. Insufficient-evidence consistency checked.
19. Completeness/status inconsistency detected.
20. Cross-field status/section contradiction detected.
21. Safety/guarantee language rejected.
22. Price-target language rejected.
23. Warning vs error classification works correctly.
24. Multiple validation issues reported together.
25. Valid partial report passes with appropriate warnings and no invented data.
"""

from datetime import datetime, timezone

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
from app.agents.report_consistency import (
    ReportConsistencyValidator,
    ReportValidationSeverity,
    ReportValidationStatus,
    validate_final_report,
)
from app.agents.report_generator import (
    ReportGeneratorAgent,
)
from app.agents.report_schema import (
    FinalReport,
    RecommendationStance,
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
            volume_analysis="Volume confirms breakout over consolidation.",
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


@pytest.fixture
def sample_valid_report(
    sample_unified_analysis: UnifiedSpecialistAnalysis,
) -> FinalReport:
    agent = ReportGeneratorAgent(deterministic_only=True)
    res = agent.run(sample_unified_analysis)
    assert res.data is not None
    return res.data


# ===========================================================================
# 1. STRUCTURAL AND IDENTITY VALIDATION TESTS (Checks 1-7)
# ===========================================================================


class TestReportStructuralValidation:
    """Tests 1-7: Schema validation, identity preservation, and investor context."""

    def test_01_fully_valid_report_passes(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 1: Fully valid report passes with is_valid=True and 0 failures."""
        result = validate_final_report(sample_valid_report, sample_unified_analysis)
        assert result.is_valid is True
        assert result.failures_count == 0
        assert len(result.errors) == 0
        assert result.ticker == "AAPL"
        assert result.passed_checks_count > 0

    def test_02_invalid_report_input_type_fails(
        self, sample_valid_report: FinalReport
    ) -> None:
        """Test 2: Invalid source_data type raises TypeError."""
        validator = ReportConsistencyValidator()
        with pytest.raises(TypeError, match="Unsupported source_data type"):
            validator.validate(sample_valid_report, "invalid_string_source")  # type: ignore

    def test_03_ticker_mismatch_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 3: Ticker mismatch is detected with CRITICAL failure."""
        report_data = sample_valid_report.model_dump()
        report_data["ticker"] = "MSFT"
        report_data["company"]["ticker"] = "MSFT"
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_STRUCT_TICKER_MATCH"
            and i.status == ReportValidationStatus.FAILED
            and i.severity == ReportValidationSeverity.CRITICAL
            for i in result.issues
        )

    def test_04_company_name_mismatch_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 4: Company name mismatch is detected with HIGH failure."""
        report_data = sample_valid_report.model_dump()
        report_data["company"]["name"] = "Microsoft Corporation"
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_STRUCT_COMPANY_NAME_MATCH"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_05_investor_profile_mismatch_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 5: Profile attribute changes (e.g. goal or risk) detected."""
        report_data = sample_valid_report.model_dump()
        report_data["investor_profile"]["risk_tolerance"] = "aggressive"
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_STRUCT_PROFILE_PRESERVED"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_06_horizon_mismatch_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 6: Horizon mismatch detected with HIGH failure."""
        report_data = sample_valid_report.model_dump()
        report_data["horizon"] = "10+ years ultra long term"
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_STRUCT_HORIZON_MATCH"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_07_capital_mismatch_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 7: Capital amount mismatch detected with HIGH failure."""
        report_data = sample_valid_report.model_dump()
        report_data["capital"]["amount"] = 999999.0
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_STRUCT_CAPITAL_MATCH"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )


# ===========================================================================
# 2. SPECIALIST & AVAILABILITY CONSISTENCY TESTS (Checks 8-9, 14-16)
# ===========================================================================


class TestReportSpecialistConsistency:
    """Tests 8-9, 14-16: Specialist availability and contradiction checks."""

    def test_08_missing_specialist_represented_as_available_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 8: Upstream missing specialist populated in report is rejected."""
        analysis_data = sample_unified_analysis.model_dump()
        analysis_data["specialist_statuses"]["technical"] = SpecialistStatus.MISSING
        analysis_data["technical_assessment"] = None
        analysis = UnifiedSpecialistAnalysis.model_validate(analysis_data)

        # sample_valid_report has technical populated
        result = validate_final_report(sample_valid_report, analysis)
        assert result.is_valid is False
        assert any(
            "MISSING_TECHNICAL_POPULATED" in i.check_id
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_09_failed_specialist_represented_as_successful_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 9: Upstream failed specialist represented as available is rejected."""
        analysis_data = sample_unified_analysis.model_dump()
        analysis_data["specialist_statuses"]["news"] = SpecialistStatus.FAILED
        analysis_data["news_assessment"] = None
        analysis = UnifiedSpecialistAnalysis.model_validate(analysis_data)

        # sample_valid_report has news populated
        result = validate_final_report(sample_valid_report, analysis)
        assert result.is_valid is False
        assert any(
            "FAILED_NEWS_POPULATED" in i.check_id
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_14_evidence_attribution_mismatch_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 14: Evidence item in report not in upstream pool is detected."""
        report_data = sample_valid_report.model_dump()
        report_data["evidence_sources"].append(
            {
                "specialist": "fundamental",
                "reference_id": "completely_fabricated_ref_123",
                "detail": "Fabricated evidence detail",
            }
        )
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_EVID_SOURCE_ATTRIBUTION_MISMATCH"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_15_specialist_disagreement_accepted_as_legitimate(
        self,
        sample_investor_profile: AggregatorInvestorProfile,
        sample_technical: TechnicalAnalysisOutput,
        sample_fundamental: FundamentalAnalysisOutput,
        sample_news: NewsAnalysisOutput,
        sample_research: ResearchAnalysisOutput,
        sample_risk: RiskAnalysisOutput,
    ) -> None:
        """Test 15: Conflicting specialists produce cautious stance and pass."""
        tech_down = sample_technical.model_copy(
            update={"trend": "downtrend", "technical_score": 30.0}
        )
        conflict = SignalConflict(
            topic="Technical Momentum vs Fundamental Quality",
            description="Technical is bearish while fundamentals remain strong.",
            specialist_positions={
                "technical": "downtrend / bearish",
                "fundamental": "favorable / strong",
            },
            involved_specialists=["technical", "fundamental"],
        )
        analysis = UnifiedSpecialistAnalysis(
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
            technical_assessment=tech_down,
            fundamental_assessment=sample_fundamental,
            news_assessment=sample_news,
            research_assessment=sample_research,
            risk_assessment=sample_risk,
            aggregated_evidence=[
                AggregatedEvidenceItem(
                    specialist="technical",
                    reference_id="tech_ev_0",
                    detail="Downtrend confirmed below 200 SMA",
                )
            ],
            areas_of_agreement=[],
            signal_conflicts=[conflict],
            cross_specialist_observations=[],
            overall_synthesis=(
                "Mixed signals: technical pressure against solid fundamentals."
            ),
            confidence=0.75,
        )

        agent = ReportGeneratorAgent(deterministic_only=True)
        res = agent.run(analysis)
        assert res.data is not None
        report = res.data

        result = validate_final_report(report, analysis)
        assert result.is_valid is True
        assert any(
            i.check_id == "CHK_REC_CONFLICT_AWARE"
            and i.status == ReportValidationStatus.PASSED
            for i in result.issues
        )

    def test_16_unsupported_contradiction_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 16: Narrative claiming strong uptrend when downtrend is rejected."""
        analysis_data = sample_unified_analysis.model_dump()
        analysis_data["technical_assessment"]["trend"] = "downtrend"
        analysis = UnifiedSpecialistAnalysis.model_validate(analysis_data)

        # Mutate report technical summary to state strong uptrend
        report_data = sample_valid_report.model_dump()
        report_data["technical"]["summary"] = "Stock is in a strong uptrend."
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_SPEC_TECH_CONTRADICTION"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )


# ===========================================================================
# 3. NUMERICAL PROVENANCE & EVIDENCE TESTS (Checks 10-13)
# ===========================================================================


class TestReportNumericalAndEvidenceProvenance:
    """Tests 10-13: Quantitative provenance and reference grounding."""

    def test_10_unsupported_numerical_value_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 10: Ungrounded numerical claim in narrative is detected as failure."""
        report_data = sample_valid_report.model_dump()
        report_data["key_reasons"].append(
            "Projected revenue of 99482.50 billion next quarter."
        )
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_NUM_UNGROUNDED_VALUE"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )
        assert len(result.numerical_issues) > 0

    def test_11_correct_numerical_provenance_accepted(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 11: Quantitative values present in upstream sources pass validation."""
        # 185.50 and 45.0 exist in sample_technical and sample_fundamental
        report_data = sample_valid_report.model_dump()
        report_data["key_reasons"] = [
            "Latest close price of 185.50 reflects strong buying.",
            "Gross margin of 45.0 confirms pricing power.",
        ]
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is True
        assert any(
            i.check_id == "CHK_NUM_PROVENANCE_VALID"
            and i.status == ReportValidationStatus.PASSED
            for i in result.issues
        )

    def test_12_phantom_source_reference_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 12: Phantom citations like [ref: phantom_doc] are failed."""
        report_data = sample_valid_report.model_dump()
        report_data["recommendation"][
            "rationale"
        ] += " Supported by [ref: phantom_doc_999]."
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_EVID_PHANTOM_CITATION"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )
        assert len(result.evidence_issues) > 0

    def test_13_valid_source_reference_accepted(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 13: Citations pointing to real aggregated evidence pass cleanly."""
        report_data = sample_valid_report.model_dump()
        # chunk_01 and tech_ev_0 are genuine references in sample_unified_analysis
        report_data["key_reasons"] = [
            "Services segment expansion noted [ref: chunk_01].",
            "Price above moving average [ref: tech_ev_0].",
        ]
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is True
        assert any(
            i.check_id == "CHK_EVID_PHANTOM_CITATION"
            and i.status == ReportValidationStatus.PASSED
            for i in result.issues
        )


# ===========================================================================
# 4. RECOMMENDATION & COMPLETENESS TESTS (Checks 17-20)
# ===========================================================================


class TestReportRecommendationAndCompleteness:
    """Tests 17-20: Recommendation grounding and completeness checks."""

    def test_17_recommendation_unsupported_by_evidence_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 17: Favorable stance given during active conflicts is failed."""
        conflict = SignalConflict(
            topic="Technical Momentum vs Fundamental Quality",
            description="Divergence in direction",
            specialist_positions={
                "technical": "bearish",
                "fundamental": "bullish",
            },
            involved_specialists=["technical", "fundamental"],
        )
        analysis_data = sample_unified_analysis.model_dump()
        analysis_data["signal_conflicts"] = [conflict.model_dump()]
        analysis = UnifiedSpecialistAnalysis.model_validate(analysis_data)

        # Mutate report to favorable stance with rationale that ignores tension
        report_data = sample_valid_report.model_dump()
        report_data["recommendation"]["stance"] = RecommendationStance.FAVORABLE
        report_data["recommendation"][
            "rationale"
        ] = "Everything appears positive across selected metrics and signals."
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_REC_CONFLICT_UNADDRESSED"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_18_insufficient_evidence_consistency_checked(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 18: Favorable recommendation under insufficient evidence is rejected."""
        analysis_data = sample_unified_analysis.model_dump()
        analysis_data["insufficient_evidence"] = True
        analysis = UnifiedSpecialistAnalysis.model_validate(analysis_data)

        # sample_valid_report has FAVORABLE or CAUTIOUS stance
        result = validate_final_report(sample_valid_report, analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_REC_INSUFFICIENT_EVIDENCE_STANCE"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_19_completeness_status_inconsistency_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 19: Report claiming completeness with insufficient data is detected."""
        analysis_data = sample_unified_analysis.model_dump()
        analysis_data["insufficient_evidence"] = True
        analysis = UnifiedSpecialistAnalysis.model_validate(analysis_data)

        report_data = sample_valid_report.model_dump()
        report_data["insufficient_evidence"] = True
        report_data["is_complete"] = True
        report_data["recommendation"]["stance"] = RecommendationStance.FAVORABLE
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_COMPLETENESS_INSUFFICIENT_FALSE_COMPLETE"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_20_cross_status_section_contradiction_detected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 20: Section marked missing in report but populated is failed."""
        report_data = sample_valid_report.model_dump()
        report_data["specialist_statuses"]["technical"] = SpecialistStatus.MISSING
        # report.technical remains populated!
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_CROSS_STATUS_SECTION_CONTRADICTION"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )


# ===========================================================================
# 5. SAFETY & REGULATORY COMPLIANCE TESTS (Checks 21-22)
# ===========================================================================


class TestReportRegulatorySafety:
    """Tests 21-22: Prohibited advice, price targets, and guarantee claims."""

    def test_21_safety_guarantee_language_rejected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 21: Guaranteed returns claims are rejected with CRITICAL failure."""
        report = sample_valid_report.model_copy(deep=True)
        object.__setattr__(
            report,
            "key_reasons",
            report.key_reasons
            + ["This investment offers guaranteed return with zero downside."],
        )

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_SAFETY_PROHIBITED_ADVICE"
            and i.status == ReportValidationStatus.FAILED
            and i.severity == ReportValidationSeverity.CRITICAL
            for i in result.issues
        )
        assert len(result.safety_issues) > 0

    def test_22_price_target_language_rejected(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 22: Price targets like 'target price $250' are rejected."""
        report = sample_valid_report.model_copy(deep=True)
        assert report.recommendation is not None
        object.__setattr__(
            report.recommendation,
            "rationale",
            report.recommendation.rationale
            + " Our 12-month target price is $250.00 per share.",
        )

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_SAFETY_PROHIBITED_ADVICE"
            and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )
        assert len(result.safety_issues) > 0


# ===========================================================================
# 6. COMPREHENSIVE & INTEGRATION TESTS (Checks 23-25)
# ===========================================================================


class TestReportConsistencyIntegration:
    """Tests 23-25: Multi-issue reporting, warning vs error, and partial reports."""

    def test_23_warning_vs_error_classification(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 23: Non-blocking conditions produce warnings while valid."""
        # Empty key_reasons produces a WARNING (CHK_REC_KEY_REASONS_EMPTY)
        report_data = sample_valid_report.model_dump()
        report_data["key_reasons"] = []
        modified_report = FinalReport.model_validate(report_data)

        result = validate_final_report(modified_report, sample_unified_analysis)
        assert result.is_valid is True
        assert result.warnings_count > 0
        assert any(
            i.status == ReportValidationStatus.WARNING
            and i.check_id == "CHK_REC_KEY_REASONS_EMPTY"
            for i in result.issues
        )

    def test_24_multiple_validation_issues_reported_together(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Test 24: Multiple distinct errors are reported comprehensively."""
        report = sample_valid_report.model_copy(deep=True)
        object.__setattr__(report.company, "ticker", "GOOGL")
        object.__setattr__(report, "disclaimer", "")
        object.__setattr__(
            report,
            "key_reasons",
            report.key_reasons + ["Guaranteed return of 999888.0 next year."],
        )

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is False
        assert result.failures_count >= 3
        check_ids = {i.check_id for i in result.errors}
        assert "CHK_STRUCT_TICKER_MATCH" in check_ids
        assert "CHK_SAFETY_DISCLAIMER_MISSING" in check_ids
        assert "CHK_SAFETY_PROHIBITED_ADVICE" in check_ids
        assert "CHK_NUM_UNGROUNDED_VALUE" in check_ids

    def test_25_valid_partial_report_passes_with_warnings_and_no_invented_data(
        self,
        sample_investor_profile: AggregatorInvestorProfile,
        sample_technical: TechnicalAnalysisOutput,
        sample_fundamental: FundamentalAnalysisOutput,
        sample_research: ResearchAnalysisOutput,
        sample_risk: RiskAnalysisOutput,
    ) -> None:
        """Test 25: Partial report with missing specialist passes cleanly."""
        # News is missing
        analysis = UnifiedSpecialistAnalysis(
            ticker="AAPL",
            target_company="Apple Inc.",
            investor_profile=sample_investor_profile,
            specialist_statuses={
                "technical": SpecialistStatus.AVAILABLE,
                "fundamental": SpecialistStatus.AVAILABLE,
                "news": SpecialistStatus.MISSING,
                "research": SpecialistStatus.AVAILABLE,
                "risk": SpecialistStatus.AVAILABLE,
            },
            data_completeness_ratio=0.8,
            technical_assessment=sample_technical,
            fundamental_assessment=sample_fundamental,
            news_assessment=None,
            research_assessment=sample_research,
            risk_assessment=sample_risk,
            aggregated_evidence=[
                AggregatedEvidenceItem(
                    specialist="technical",
                    reference_id="tech_ev_0",
                    detail="Price 185.50 above 50-day SMA",
                )
            ],
            areas_of_agreement=[],
            signal_conflicts=[],
            cross_specialist_observations=[],
            overall_synthesis=(
                "Robust technical and fundamental foundation despite absent news."
            ),
            confidence=0.82,
        )

        agent = ReportGeneratorAgent(deterministic_only=True)
        res = agent.run(analysis)
        assert res.data is not None
        report = res.data

        result = validate_final_report(report, analysis)
        assert result.is_valid is True
        assert result.failures_count == 0
        # Validator correctly noted missing news specialist
        assert any(
            "NEWS_STATUS_CONSISTENT" in i.check_id
            and i.status == ReportValidationStatus.PASSED
            for i in result.issues
        )
        # Warning about operating with partial coverage
        assert any(
            i.check_id == "CHK_COMPLETENESS_PARTIAL_REPORT_NOTICE"
            and i.status == ReportValidationStatus.WARNING
            for i in result.issues
        )


class TestReportConsistencyRemediationPass:
    """Explicit tests for Phase 12.3 remediation requirements.

    Covers:
    1. Specialist-aware numerical provenance (wrong specialist fails,
       correct specialist passes, capital misuse fails, attribution mismatch fails,
       ambiguous/unattributed numbers produce WARNING).
    2. Non-prescriptive recommendation consistency (preserves legitimate disagreement,
       does not force CAUTIOUS, detects unanimous contradictions, enforces honest
       insufficient evidence representation, never mutates report).
    """

    def test_26_same_number_in_wrong_specialist_does_not_validate(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Technical number (e.g. RSI 62.0) used in fundamental claim must fail."""
        report = sample_valid_report.model_copy(deep=True)
        # 62.0 is in technical (RSI=62.0), NOT in fundamental
        # Claim gross margin is 62.0% in fundamental narrative
        bad_fund = report.fundamental.model_copy(
            update={
                "profitability": ("Gross margin reached 62.0% exceeding expectations.")
            }
        )
        object.__setattr__(report, "fundamental", bad_fund)

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_NUM_PROVENANCE_MISMATCH"
            and i.status == ReportValidationStatus.FAILED
            for i in result.errors
        )

    def test_27_same_number_in_correct_specialist_does_validate(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Same number (62.0) in technical claim must pass."""
        report = sample_valid_report.model_copy(deep=True)
        # 62.0 is in technical (RSI=62.0)
        good_tech = report.technical.model_copy(
            update={
                "indicators_summary": (
                    "RSI momentum reads at 62.0 confirming strength."
                )
            }
        )
        object.__setattr__(report, "technical", good_tech)

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is True
        assert not any(
            "CHK_NUM" in i.check_id and i.status == ReportValidationStatus.FAILED
            for i in result.issues
        )

    def test_28_investor_capital_cannot_validate_analytical_claim(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Investor capital (50000.0) cannot validate an analytical metric claim."""
        report = sample_valid_report.model_copy(deep=True)
        # Capital is 50000.0 from investor profile
        # Use 50000.0 inside a key reason as an analytical claim
        object.__setattr__(
            report,
            "key_reasons",
            [
                "Daily trading volume exceeded 50000.0 shares.",
                report.key_reasons[1] if len(report.key_reasons) > 1 else "Reason 2",
            ],
        )

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_NUM_INVESTOR_CAPITAL_MISUSE"
            and i.status == ReportValidationStatus.FAILED
            for i in result.errors
        )

    def test_29_source_attributed_number_validates_only_when_source_matches(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Number cited with [ref: tech_ev_0] must exist in technical evidence."""
        report = sample_valid_report.model_copy(deep=True)
        # 45.0 is in fundamental (gross margin 45.0%), NOT in technical evidence
        # Attribute 45.0 to [ref: tech_ev_0]
        object.__setattr__(
            report,
            "key_reasons",
            [
                "Margin was measured at 45.0 [ref: tech_ev_0].",
                report.key_reasons[1] if len(report.key_reasons) > 1 else "Reason 2",
            ],
        )

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_NUM_PROVENANCE_MISMATCH"
            and i.status == ReportValidationStatus.FAILED
            for i in result.errors
        )

    def test_30_ambiguous_and_unattributed_numerical_claims_produce_warning(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Ambiguous or unattributed narrative numbers produce WARNING."""
        report = sample_valid_report.model_copy(deep=True)
        # 182.0 is in technical (sma_20 = 182.0).
        # Put 182.0 in key reasons without citation or domain keywords
        object.__setattr__(
            report,
            "key_reasons",
            [
                "The metric of 182.0 was noted during assessment.",
                report.key_reasons[1] if len(report.key_reasons) > 1 else "Reason 2",
            ],
        )

        result = validate_final_report(report, sample_unified_analysis)
        # Report is still valid (warnings are non-blocking)
        assert result.is_valid is True
        # Emits a warning for ambiguous or unattributed provenance
        warning_ids = {i.check_id for i in result.warnings}
        assert (
            "CHK_NUM_AMBIGUOUS_PROVENANCE" in warning_ids
            or "CHK_NUM_UNATTRIBUTED_PROVENANCE" in warning_ids
        )

    def test_31_legitimate_specialist_disagreement_accepted_without_forcing_cautious(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Legitimate disagreement accepted with tension, not forced CAUTIOUS."""
        report = sample_valid_report.model_copy(deep=True)
        # Stance is NEUTRAL (or BALANCED) acknowledging the conflict
        new_rec = report.recommendation.model_copy(
            update={
                "stance": RecommendationStance.NEUTRAL,
                "rationale": (
                    "Technical setup is favorable while valuation is stretched. "
                    "Balancing these opposing signals results in a neutral posture."
                ),
            }
        )
        object.__setattr__(report, "recommendation", new_rec)

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is True
        assert result.failures_count == 0

    def test_32_favorable_rec_rejected_when_evidence_unanimously_negative(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
        sample_technical: TechnicalAnalysisOutput,
        sample_fundamental: FundamentalAnalysisOutput,
    ) -> None:
        """Recommendation cannot be FAVORABLE if all specialists are unfavorable."""
        # Create an analysis where technical and fundamental are unfavorable
        bearish_tech = sample_technical.model_copy(
            update={
                "trend": "downtrend",
            }
        )
        bearish_fund = sample_fundamental.model_copy(
            update={
                "overall_assessment": "unfavorable",
            }
        )
        bearish_analysis = sample_unified_analysis.model_copy(
            update={
                "technical_assessment": bearish_tech,
                "fundamental_assessment": bearish_fund,
                "news_assessment": None,
                "research_assessment": None,
                "risk_assessment": None,
                "specialist_statuses": {
                    "technical": SpecialistStatus.AVAILABLE,
                    "fundamental": SpecialistStatus.AVAILABLE,
                    "news": SpecialistStatus.MISSING,
                    "research": SpecialistStatus.MISSING,
                    "risk": SpecialistStatus.MISSING,
                },
                "overall_synthesis": "Deteriorated conditions across all domains.",
            }
        )

        # But report claims favorable stance
        report = sample_valid_report.model_copy(deep=True)
        rec = report.recommendation.model_copy(
            update={"stance": RecommendationStance.FAVORABLE}
        )
        object.__setattr__(report, "recommendation", rec)

        result = validate_final_report(report, bearish_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_REC_UNSUPPORTED_STANCE"
            and i.status == ReportValidationStatus.FAILED
            for i in result.errors
        )

    def test_33_insufficient_evidence_honestly_represented(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """If data completeness is severely low, FAVORABLE stance is rejected."""
        low_data_analysis = sample_unified_analysis.model_copy(
            update={
                "data_completeness_ratio": 0.2,
                "insufficient_evidence": True,
                "specialist_statuses": {
                    "technical": SpecialistStatus.AVAILABLE,
                    "fundamental": SpecialistStatus.MISSING,
                    "news": SpecialistStatus.MISSING,
                    "research": SpecialistStatus.MISSING,
                    "risk": SpecialistStatus.MISSING,
                },
            }
        )

        # Report claims favorable instead of INSUFFICIENT_EVIDENCE
        report = sample_valid_report.model_copy(deep=True)
        rec = report.recommendation.model_copy(
            update={"stance": RecommendationStance.FAVORABLE}
        )
        object.__setattr__(report, "recommendation", rec)

        result = validate_final_report(report, low_data_analysis)
        assert result.is_valid is False
        assert any(
            i.check_id == "CHK_REC_INSUFFICIENT_EVIDENCE_STANCE"
            and i.status == ReportValidationStatus.FAILED
            for i in result.errors
        )

    def test_34_validator_does_not_replace_or_mutate_recommendation(
        self,
        sample_valid_report: FinalReport,
        sample_unified_analysis: UnifiedSpecialistAnalysis,
    ) -> None:
        """Validator is strictly an auditor; it never mutates report recommendation."""
        report = sample_valid_report.model_copy(deep=True)
        original_stance = report.recommendation.stance
        original_rationale = report.recommendation.rationale
        original_reasons = list(report.key_reasons)
        original_risks = list(report.important_risks)

        result = validate_final_report(report, sample_unified_analysis)
        assert result.is_valid is True

        # Assert no mutation occurred
        assert report.recommendation.stance == original_stance
        assert report.recommendation.rationale == original_rationale
        assert report.key_reasons == original_reasons
        assert report.important_risks == original_risks
