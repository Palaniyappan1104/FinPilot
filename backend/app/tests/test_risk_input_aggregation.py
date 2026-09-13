"""Tests for Risk Analyst input aggregation and schema definitions (Phase 10.1).

Verifies according to plan.md Phase 10.1:
- 10.1.1: Upstream outputs feeding Risk Analyst (technical, fundamental, news,
  and investor profile).
- 10.1.2: Lightweight interface to consume partial results if specialists failed.
- Representation of required risk categories (market, company, sector, financial,
  volatility, investor-specific).
- Structured RiskEvidenceRef provenance and data integrity.
- Safe representation and degradation when upstream signals are missing.
- Safety boundaries (rejection of buy/sell advice, price targets, guarantees).
"""

import pytest

from app.agents.base import AgentResult
from app.agents.risk_schema import (
    FundamentalRiskSignals,
    NewsRiskSignals,
    ResearchRiskSignals,
    RiskAnalysisOutput,
    RiskAnalysisValidationError,
    RiskAnalystInput,
    RiskCategory,
    RiskEvidenceRef,
    RiskFactor,
    RiskInvestorProfile,
    RiskProbability,
    RiskSeverity,
    TechnicalRiskSignals,
)
from app.agents.state import GraphState, InvestorProfile

# ===========================================================================
# 1. RISK CATEGORIES, SEVERITY, AND EVIDENCE SCHEMAS
# ===========================================================================


class TestRiskCoreSchemas:
    """Verifies domain enums, evidence references, and risk factors."""

    def test_all_expected_risk_categories_exist(self) -> None:
        """Verify all categories from plan.md 10.2 are representable in RiskCategory."""
        categories = {cat.value for cat in RiskCategory}
        expected = {
            "market",
            "company",
            "sector",
            "financial",
            "volatility",
            "investor_specific",
        }
        assert expected.issubset(categories)

    def test_risk_severity_levels(self) -> None:
        """Verify standard severity scale."""
        assert RiskSeverity.LOW == "low"
        assert RiskSeverity.MODERATE == "moderate"
        assert RiskSeverity.HIGH == "high"
        assert RiskSeverity.CRITICAL == "critical"

    def test_risk_probability_levels(self) -> None:
        """Verify standard probability scale."""
        assert RiskProbability.LOW == "low"
        assert RiskProbability.MEDIUM == "medium"
        assert RiskProbability.HIGH == "high"
        assert RiskProbability.UNCERTAIN == "uncertain"

    def test_risk_evidence_ref_construction(self) -> None:
        """Verify valid evidence reference preserves source and detail."""
        ref = RiskEvidenceRef(
            source_type="technical",
            reference_id="rsi",
            detail="RSI is 76.8 indicating overbought momentum.",
        )
        assert ref.source_type == "technical"
        assert ref.reference_id == "rsi"
        assert "76.8" in ref.detail

    def test_risk_evidence_ref_rejects_empty_fields(self) -> None:
        """Verify empty reference_id or detail raises ValidationError."""
        with pytest.raises(ValueError):
            RiskEvidenceRef(
                source_type="fundamental",
                reference_id="   ",
                detail="Some detail",
            )

        with pytest.raises(ValueError):
            RiskEvidenceRef(
                source_type="fundamental",
                reference_id="debt_to_equity",
                detail="",
            )

    def test_risk_factor_construction(self) -> None:
        """Verify complete risk factor construction with evidence citations."""
        evidence = [
            RiskEvidenceRef(
                source_type="fundamental",
                reference_id="debt_to_equity",
                detail="Debt-to-equity ratio is 2.85.",
            ),
            RiskEvidenceRef(
                source_type="news",
                reference_id="art_001",
                detail="Credit downgrade warning reported by rating agency.",
            ),
        ]
        factor = RiskFactor(
            category=RiskCategory.FINANCIAL,
            name="Elevated Leverage Risk",
            description="Company carries substantial debt relative to equity.",
            severity=RiskSeverity.HIGH,
            probability=RiskProbability.MEDIUM,
            evidence=evidence,
            insufficient_data=False,
        )
        assert factor.category == RiskCategory.FINANCIAL
        assert factor.severity == RiskSeverity.HIGH
        assert len(factor.evidence) == 2
        assert factor.evidence[0].reference_id == "debt_to_equity"

    def test_risk_factor_insufficient_data_flag(self) -> None:
        """Verify risk factor can explicitly flag insufficient data."""
        factor = RiskFactor(
            category=RiskCategory.SECTOR,
            name="Sector Headwinds",
            description="Insufficient data to evaluate sector trends.",
            severity=RiskSeverity.LOW,
            insufficient_data=True,
        )
        assert factor.insufficient_data is True


# ===========================================================================
# 2. INVESTOR PROFILE SCHEMA (SEPARATE FROM COMPANY RISK)
# ===========================================================================


class TestRiskInvestorProfile:
    """Verifies investor profile modeling and conversion."""

    def test_investor_profile_creation(self) -> None:
        """Verify investor profile captures horizon and risk tolerance."""
        profile = RiskInvestorProfile(
            time_horizon="3 months",
            risk_tolerance="conservative",
            investment_goal="capital preservation",
            capital_amount=50000.0,
            target_company="Apple Inc.",
            ticker="AAPL",
            profile_complete=True,
        )
        assert profile.time_horizon == "3 months"
        assert profile.risk_tolerance == "conservative"
        assert profile.capital_amount == 50000.0

    def test_from_investor_profile_typed_dict(self) -> None:
        """Verify parsing from GraphState InvestorProfile TypedDict."""
        raw_dict: InvestorProfile = {
            "target_company": "Microsoft Corporation",
            "ticker": "MSFT",
            "investment_goal": "long term growth",
            "time_horizon": "5 years",
            "capital_amount": 100000.0,
            "risk_tolerance": "aggressive",
            "profile_complete": True,
        }
        model = RiskInvestorProfile.from_investor_profile(raw_dict)
        assert model is not None
        assert model.ticker == "MSFT"
        assert model.risk_tolerance == "aggressive"
        assert model.time_horizon == "5 years"

    def test_from_investor_profile_none_handling(self) -> None:
        """Verify None or empty dict returns None gracefully."""
        assert RiskInvestorProfile.from_investor_profile(None) is None
        assert RiskInvestorProfile.from_investor_profile({}) is None


# ===========================================================================
# 3. UPSTREAM SIGNAL PARSERS (10.1.1)
# ===========================================================================


class TestUpstreamSignalParsers:
    """Verifies extraction of signals from Technical, Fundamental, and News."""

    def test_parse_technical_risk_signals(self) -> None:
        """Verify extraction from technical analysis dictionary."""
        tech_dict = {
            "ticker": "NVDA",
            "trend": "uptrend",
            "technical_score": 82.5,
            "indicators_summary": {
                "latest_close": 125.5,
                "rsi": {"value": 74.2, "regime": "overbought"},
            },
            "risks": ["RSI overbought", "Extended from 200 SMA"],
            "evidence": ["Price is above 50 SMA and 200 SMA"],
            "confidence": 0.90,
        }
        signals = TechnicalRiskSignals.from_output(tech_dict)
        assert signals is not None
        assert signals.ticker == "NVDA"
        assert signals.trend == "uptrend"
        assert signals.rsi == 74.2
        assert len(signals.risks) == 2
        assert signals.confidence == 0.90

    def test_parse_fundamental_risk_signals(self) -> None:
        """Verify extraction from fundamental analysis dictionary."""
        fund_dict = {
            "ticker": "TSLA",
            "financial_health": {"rating": "moderate"},
            "leverage_assessment": {"rating": "weak"},
            "cash_flow_assessment": {"rating": "strong"},
            "key_weaknesses": ["Operating margins compressed by price cuts."],
            "notable_flags": ["Margin decline YoY"],
            "overall_assessment": "neutral",
            "confidence": 0.85,
        }
        signals = FundamentalRiskSignals.from_output(fund_dict)
        assert signals is not None
        assert signals.ticker == "TSLA"
        assert signals.leverage_rating == "weak"
        assert signals.cash_flow_rating == "strong"
        assert len(signals.key_weaknesses) == 1
        assert signals.confidence == 0.85

    def test_parse_news_risk_signals(self) -> None:
        """Verify extraction from news analysis dictionary."""
        news_dict = {
            "ticker": "GOOGL",
            "overall_sentiment": "negative",
            "negative_factors": [
                {"text": "Antitrust ruling against search exclusivity."},
                {"text": "Ad revenue slowdown in Europe."},
            ],
            "important_events": [{"description": "DOJ antitrust trial ruling issued."}],
            "confidence": 0.88,
        }
        signals = NewsRiskSignals.from_output(news_dict)
        assert signals is not None
        assert signals.ticker == "GOOGL"
        assert signals.overall_sentiment == "negative"
        assert len(signals.negative_factors) == 2
        assert "Antitrust" in signals.negative_factors[0]
        assert len(signals.important_events) == 1

    def test_parse_research_risk_signals(self) -> None:
        """Verify extraction from research analyst dictionary."""
        res_dict = {
            "query": "What are 10-K risk factors?",
            "answer": "Primary risk is cloud infrastructure competition.",
            "key_findings": [{"claim": "Data center capex increased by 40%."}],
            "confidence": 0.85,
            "insufficient_evidence": False,
        }
        signals = ResearchRiskSignals.from_output(res_dict)
        assert signals is not None
        assert len(signals.findings) == 1
        assert "Data center capex" in signals.findings[0]
        assert signals.insufficient_evidence is False

    def test_failed_agent_result_returns_none(self) -> None:
        """Verify specialist execution failure produces None instead of raising."""
        failed_result = AgentResult.create_failure("Upstream timeout")
        assert TechnicalRiskSignals.from_output(failed_result) is None
        assert FundamentalRiskSignals.from_output(failed_result) is None
        assert NewsRiskSignals.from_output(failed_result) is None
        assert ResearchRiskSignals.from_output(failed_result) is None


# ===========================================================================
# 4. RISK ANALYST INPUT AGGREGATION & PARTIAL RESULTS (10.1.2)
# ===========================================================================


class TestRiskAnalystInputAggregation:
    """Verifies complete and partial input aggregation capabilities."""

    def test_all_specialists_present_aggregation(self) -> None:
        """Verify aggregation when all upstream specialists succeed."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="MSFT",
            investor_profile={
                "time_horizon": "1 year",
                "risk_tolerance": "moderate",
            },
            technical_output={
                "ticker": "MSFT",
                "trend": "uptrend",
                "risks": ["Resistance at $450"],
            },
            fundamental_output={
                "ticker": "MSFT",
                "financial_health": {"rating": "strong"},
                "key_weaknesses": ["Valuation multiple elevated"],
            },
            news_output={
                "ticker": "MSFT",
                "overall_sentiment": "positive",
                "negative_factors": [{"text": "Cloud margin pressure"}],
            },
            research_output={
                "key_findings": [{"claim": "AI capital expenditure expanding"}]
            },
        )
        assert input_data.ticker == "MSFT"
        assert input_data.has_technical_signals is True
        assert input_data.has_fundamental_signals is True
        assert input_data.has_news_signals is True
        assert input_data.has_research_signals is True
        assert input_data.has_investor_profile is True
        assert input_data.is_empty is False
        assert set(input_data.available_sources) == {
            "technical",
            "fundamental",
            "news",
            "research",
            "investor_profile",
        }
        assert input_data.missing_sources == []
        assert input_data.data_completeness_ratio == 1.0

    def test_partial_specialist_input_consumption(self) -> None:
        """Verify input when technical succeeded but fundamental & news skipped."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={
                "ticker": "AAPL",
                "trend": "downtrend",
                "risks": ["Below 200-day moving average"],
            },
            fundamental_output=None,  # Skipped
            news_output=None,  # Skipped
        )
        assert input_data.ticker == "AAPL"
        assert input_data.has_technical_signals is True
        assert input_data.has_fundamental_signals is False
        assert input_data.has_news_signals is False
        assert input_data.is_empty is False
        assert input_data.available_sources == ["technical"]
        assert set(input_data.missing_sources) == {"fundamental", "news"}
        assert input_data.data_completeness_ratio == pytest.approx(1 / 3)

    def test_from_graph_state_with_failed_and_successful_specialists(self) -> None:
        """Verify from_graph_state consumes partial results from LangGraph state."""
        state: GraphState = {
            "user_query": "Assess risk for Amazon",
            "investor_profile": {
                "target_company": "Amazon",
                "ticker": "AMZN",
                "risk_tolerance": "conservative",
                "time_horizon": "6 months",
            },
            # Technical succeeded
            "technical_result": {
                "success": True,
                "data": {
                    "ticker": "AMZN",
                    "trend": "uptrend",
                    "risks": ["High RSI"],
                },
            },
            # Fundamental failed (e.g. rate limit)
            "fundamental_result": {
                "success": False,
                "error": "HTTP 429 Too Many Requests",
                "data": None,
            },
            # News was skipped
            "news_result": None,
            "research_result": None,
        }

        input_data = RiskAnalystInput.from_graph_state(state)
        assert input_data.ticker == "AMZN"
        assert input_data.has_technical_signals is True
        assert input_data.has_fundamental_signals is False  # Failed specialist ignored
        assert input_data.has_news_signals is False  # Skipped specialist None
        assert input_data.has_investor_profile is True
        assert input_data.investor_profile.risk_tolerance == "conservative"
        assert input_data.available_sources == ["technical", "investor_profile"]
        assert set(input_data.missing_sources) == {"fundamental", "news"}

    def test_completely_empty_input_handled_gracefully(self) -> None:
        """Verify completely empty input sets is_empty flag without crashing."""
        input_data = RiskAnalystInput.from_partial_results(ticker="META")
        assert input_data.ticker == "META"
        assert input_data.is_empty is True
        assert input_data.available_sources == []
        assert set(input_data.missing_sources) == {"technical", "fundamental", "news"}
        assert input_data.data_completeness_ratio == 0.0

    def test_ticker_resolution_from_specialist_when_state_lacks_ticker(self) -> None:
        """Verify ticker inferred from specialist output if missing in state."""
        state: GraphState = {
            "user_query": "Analyze risk",
            "technical_result": {
                "success": True,
                "data": {"ticker": "AMD", "trend": "uptrend"},
            },
        }
        input_data = RiskAnalystInput.from_graph_state(state)
        assert input_data.ticker == "AMD"

    def test_unresolvable_ticker_raises_error(self) -> None:
        """Verify error is raised if ticker cannot be resolved anywhere in state."""
        state: GraphState = {
            "user_query": "Analyze risk",
        }
        with pytest.raises(ValueError, match="Unable to resolve ticker"):
            RiskAnalystInput.from_graph_state(state)


# ===========================================================================
# 5. RISK OUTPUT SCHEMA & SAFETY VALIDATION
# ===========================================================================


class TestRiskAnalysisOutputSchema:
    """Verifies output foundation schema, categorization, and safety enforcement."""

    def test_valid_risk_output_construction(self) -> None:
        """Verify valid RiskAnalysisOutput construction across multiple categories."""
        output = RiskAnalysisOutput(
            ticker="MSFT",
            overall_risk_level=RiskSeverity.MODERATE,
            market_risks=[
                RiskFactor(
                    category=RiskCategory.MARKET,
                    name="Macro Rate Sensitivity",
                    description="Valuation sensitive to benchmark yield increases.",
                    severity=RiskSeverity.MODERATE,
                )
            ],
            financial_risks=[
                RiskFactor(
                    category=RiskCategory.FINANCIAL,
                    name="Commercial Paper Maturing",
                    description="Short-term debt maturity within 6 months.",
                    severity=RiskSeverity.LOW,
                )
            ],
            investor_specific_risks=[
                RiskFactor(
                    category=RiskCategory.INVESTOR_SPECIFIC,
                    name="Horizon Mismatch",
                    description="Short horizon may not absorb capex payback cycle.",
                    severity=RiskSeverity.HIGH,
                )
            ],
            data_completeness={"technical": True, "fundamental": True, "news": False},
            insufficient_data=False,
            summary="Microsoft demonstrates strong balance sheet fundamentals.",
            confidence=0.85,
        )
        assert output.ticker == "MSFT"
        assert output.overall_risk_level == RiskSeverity.MODERATE
        assert len(output.market_risks) == 1
        assert len(output.financial_risks) == 1
        assert len(output.investor_specific_risks) == 1
        assert output.confidence == 0.85

    def test_insufficient_data_state_representation(self) -> None:
        """Verify output cleanly represents insufficient data without guessing."""
        output = RiskAnalysisOutput(
            ticker="UNKNOWN",
            overall_risk_level=None,
            insufficient_data=True,
            insufficient_data_reason="Technical and fundamental metrics unavailable.",
            confidence=0.0,
            summary="Risk assessment cannot proceed due to absence of upstream data.",
        )
        assert output.insufficient_data is True
        assert output.overall_risk_level is None
        assert output.confidence == 0.0
        assert "unavailable" in output.insufficient_data_reason

    @pytest.mark.parametrize(
        "prohibited_text",
        [
            "We issue a buy recommendation for this stock.",
            "Our strong sell recommendation is based on debt.",
            "Target price is $150 per share.",
            "The price target indicates upside.",
            "This asset offers guaranteed returns.",
            "An entirely risk-free investment opportunity.",
            "Investors must buy at current levels.",
        ],
    )
    def test_prohibited_advisory_phrases_rejected(self, prohibited_text: str) -> None:
        """Verify safety validator rejects advice, price targets, and guarantees."""
        with pytest.raises(
            RiskAnalysisValidationError, match="prohibited advisory phrase"
        ):
            RiskAnalysisOutput(
                ticker="TEST",
                summary=f"Analysis summary. {prohibited_text}",
                confidence=0.80,
            )

    def test_empty_ticker_rejected(self) -> None:
        """Verify empty ticker raises validation error."""
        with pytest.raises(ValueError, match="Ticker cannot be empty"):
            RiskAnalysisOutput(
                ticker="   ",
                summary="Valid summary.",
            )
