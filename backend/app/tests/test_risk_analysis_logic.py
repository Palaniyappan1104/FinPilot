"""Comprehensive tests for Risk Analysis Logic (Phase 10.3).

Verifies according to plan.md Phase 10.3 and strict grounding audit requirements:
- 10.3.1 Output schema: market risks, company risks, sector risks, financial risks,
  volatility risks, investor-specific risks, overall risk level, confidence.
- 10.3.2 Prompt combining quantitative signals with investor context.
- 10.3.3 Deterministic quantitative scoring strictly from genuine numeric metrics (RSI,
  technical_score). Qualitative ratings are NOT converted to arbitrary numbers.
  No list-length heuristics, no 0.50 fallback defaults.
- 10.3.4 Validate structured output (prohibited advice detection, evidence grounding,
  category consistency, confidence capping, and graceful partial input handling).
"""

from typing import Any, Dict, Optional

import pytest

from app.agents.risk import (
    RiskAnalystAgent,
    risk_analyst_node,
    validate_risk_analysis,
)
from app.agents.risk_schema import (
    RiskAnalysisOutput,
    RiskAnalysisValidationError,
    RiskAnalystInput,
    RiskCategory,
    RiskEvidenceRef,
    RiskFactor,
    RiskSeverity,
)
from app.agents.risk_scoring import (
    calculate_deterministic_risk_score,
    score_rsi,
    score_technical_score,
)
from app.core.llm.base import LLMProvider, LLMResponse
from app.core.llm.exceptions import LLMError

# ===========================================================================
# MOCK LLM PROVIDER FOR OFFLINE DETERMINISTIC TESTING
# ===========================================================================


class MockRiskLLMProvider(LLMProvider):
    """Offline test LLM provider returning configurable structured risk outputs."""

    def __init__(
        self,
        output_payload: Optional[Dict[str, Any]] = None,
        should_fail: bool = False,
        error_message: str = "Simulated LLM failure",
    ) -> None:
        self.output_payload = output_payload or {}
        self.should_fail = should_fail
        self.error_message = error_message
        self.last_prompt: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "mock_risk_llm"

    def generate(
        self,
        prompt: str,
        schema: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.last_prompt = prompt
        if self.should_fail:
            raise LLMError(self.error_message)
        import json

        return LLMResponse(
            content=json.dumps(self.output_payload),
            model="mock-risk-model",
            provider=self.provider_name,
            metadata={},
        )


def _build_sample_valid_output(
    ticker: str = "AAPL",
    overall_level: RiskSeverity = RiskSeverity.MODERATE,
    summary: str = "Objective risk assessment indicates balanced exposure.",
) -> RiskAnalysisOutput:
    """Helper creating a fully populated, grounded RiskAnalysisOutput."""
    return RiskAnalysisOutput(
        ticker=ticker,
        overall_risk_level=overall_level,
        market_risks=[
            RiskFactor(
                category=RiskCategory.MARKET,
                name="Broader Market Sentiment Risk",
                description="Technical trend suggests potential pullback.",
                severity=RiskSeverity.MODERATE,
                evidence=[
                    RiskEvidenceRef(
                        source_type="technical",
                        reference_id="trend_1",
                        detail="Technical trend reported as consolidation.",
                    )
                ],
            )
        ],
        company_risks=[
            RiskFactor(
                category=RiskCategory.COMPANY,
                name="Operating Margin Headwind",
                description="Supply chain friction identified in filings.",
                severity=RiskSeverity.MODERATE,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="weakness_1",
                        detail="Key weakness reported in margin compression.",
                    )
                ],
            )
        ],
        sector_risks=[
            RiskFactor(
                category=RiskCategory.SECTOR,
                name="Antitrust Scrutiny in Big Tech",
                description="Material antitrust inquiries in the news.",
                severity=RiskSeverity.HIGH,
                evidence=[
                    RiskEvidenceRef(
                        source_type="news",
                        reference_id="antitrust_1",
                        detail="Tech sector scrutiny headline.",
                    )
                ],
            )
        ],
        financial_risks=[
            RiskFactor(
                category=RiskCategory.FINANCIAL,
                name="Moderate Leverage Burden",
                description="Debt to equity remains manageable.",
                severity=RiskSeverity.LOW,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="lev_1",
                        detail="Leverage rated strong.",
                    )
                ],
            )
        ],
        volatility_risks=[
            RiskFactor(
                category=RiskCategory.VOLATILITY,
                name="Elevated RSI Near Overbought",
                description="RSI is 68, near overbought boundary.",
                severity=RiskSeverity.MODERATE,
                evidence=[
                    RiskEvidenceRef(
                        source_type="technical",
                        reference_id="rsi_1",
                        detail="RSI indicator reading is 68.0.",
                    )
                ],
            )
        ],
        investor_specific_risks=[
            RiskFactor(
                category=RiskCategory.INVESTOR_SPECIFIC,
                name="Horizon Mismatch Risk",
                description="Short-term capital facing medium-term consolidation.",
                severity=RiskSeverity.MODERATE,
                evidence=[
                    RiskEvidenceRef(
                        source_type="investor_profile",
                        reference_id="horizon_1",
                        detail="Investor stated 6 month investment horizon.",
                    )
                ],
            )
        ],
        summary=summary,
        confidence=0.85,
    )


# ===========================================================================
# 1. DETERMINISTIC QUANTITATIVE SCORER TESTS (10.3.3)
# ===========================================================================


class TestDeterministicRiskScorer:
    """Tests the deterministic quantitative risk scoring logic with strict grounding."""

    def test_completely_empty_input_returns_insufficient_data(self) -> None:
        """Verify empty input signals return insufficient data without score."""
        input_data = RiskAnalystInput(ticker="EMPTY")
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is True
        assert score.score is None
        assert score.level is None
        assert "No genuine quantitative metrics" in score.explanation

    def test_qualitative_ratings_alone_do_not_produce_numeric_score(self) -> None:
        """Verify qualitative strings do NOT generate a numeric score."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="QUAL_ONLY",
            fundamental_output={
                "ticker": "QUAL_ONLY",
                "financial_health": {"rating": "strong"},
                "leverage_assessment": {"rating": "strong"},
                "cash_flow_assessment": {"rating": "strong"},
                "key_weaknesses": ["Minor competition"],
            },
            technical_output={
                "ticker": "QUAL_ONLY",
                "trend": "uptrend",
                # Note: No numeric RSI, no numeric technical_score
            },
            news_output={
                "ticker": "QUAL_ONLY",
                "overall_sentiment": "positive",
                "negative_factors": [],
            },
        )
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is True
        assert score.score is None
        assert score.level is None
        assert "no genuine numeric metrics" in score.explanation.lower()

    def test_list_length_does_not_affect_risk_score(self) -> None:
        """Verify bullet-point counts (len of weaknesses/risks) do NOT change score."""
        # Input with 0 weaknesses/risks
        input_few = RiskAnalystInput.from_partial_results(
            ticker="FEW",
            technical_output={
                "ticker": "FEW",
                "rsi": {"value": 50.0},
                "technical_score": 75.0,
                "risks": [],
            },
            fundamental_output={
                "ticker": "FEW",
                "key_weaknesses": [],
            },
        )
        # Input with 10 weaknesses/risks
        input_many = RiskAnalystInput.from_partial_results(
            ticker="MANY",
            technical_output={
                "ticker": "MANY",
                "rsi": {"value": 50.0},
                "technical_score": 75.0,
                "risks": [f"Risk bullet {i}" for i in range(10)],
            },
            fundamental_output={
                "ticker": "MANY",
                "key_weaknesses": [f"Weakness bullet {i}" for i in range(10)],
            },
        )

        score_few = calculate_deterministic_risk_score(input_few)
        score_many = calculate_deterministic_risk_score(input_many)

        assert score_few.score is not None
        assert score_many.score is not None
        # List length heuristics must NOT exist: scores must be identical
        assert score_few.score == score_many.score

    def test_missing_metrics_do_not_default_to_0_50(self) -> None:
        """Verify missing metrics result in insufficient_data, never an assumed 0.50."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="NOMAGIC50",
            technical_output={
                "ticker": "NOMAGIC50",
                "trend": "sideways",
                # rsi is None, technical_score is None
            },
        )
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is True
        assert score.score is None
        assert score.level is None

    def test_genuine_rsi_and_technical_score_produce_deterministic_score(self) -> None:
        """Verify genuine numeric metrics produce grounded deterministic scores."""
        # Bullish / healthy technicals
        input_bullish = RiskAnalystInput.from_partial_results(
            ticker="BULLISH",
            technical_output={
                "ticker": "BULLISH",
                "rsi": {"value": 52.0},
                "technical_score": 85.0,
            },
        )
        score_bullish = calculate_deterministic_risk_score(input_bullish)
        assert score_bullish.insufficient_data is False
        assert score_bullish.score is not None
        assert score_bullish.score < 0.30
        assert score_bullish.level == RiskSeverity.LOW
        assert score_bullish.rsi_risk == 0.20
        assert score_bullish.technical_score_risk == 0.15

        # Distressed / bearish technicals
        input_bearish = RiskAnalystInput.from_partial_results(
            ticker="BEARISH",
            technical_output={
                "ticker": "BEARISH",
                "rsi": {"value": 18.0},  # Deep oversold / breakdown (<20 -> 0.85)
                "technical_score": 15.0,  # Severe breakdown ((100-15)/100 -> 0.85)
            },
        )
        score_bearish = calculate_deterministic_risk_score(input_bearish)
        assert score_bearish.insufficient_data is False
        assert score_bearish.score is not None
        assert score_bearish.score >= 0.80
        assert score_bearish.level in {RiskSeverity.HIGH, RiskSeverity.CRITICAL}

    def test_partial_numeric_availability_rsi_only(self) -> None:
        """Verify scoring functions when only RSI is present."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="RSI_ONLY",
            technical_output={
                "ticker": "RSI_ONLY",
                "rsi": {"value": 50.0},
            },
        )
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is False
        assert score.score is not None
        assert score.rsi_risk == 0.20
        assert score.technical_score_risk is None
        assert score.component_breakdown["active_metrics"] == ["rsi"]

    def test_partial_numeric_availability_technical_score_only(self) -> None:
        """Verify scoring functions when only technical_score is present."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="SCORE_ONLY",
            technical_output={
                "ticker": "SCORE_ONLY",
                "technical_score": 80.0,
            },
        )
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is False
        assert score.score is not None
        assert score.technical_score_risk == 0.20
        assert score.rsi_risk is None
        assert score.component_breakdown["active_metrics"] == ["technical_score"]

    def test_investor_tolerance_does_not_manufacture_score_without_quantitative_data(
        self,
    ) -> None:
        """Verify investor profile alone cannot manufacture a numeric score."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="NOMETRICS",
            fundamental_output={
                "ticker": "NOMETRICS",
                "financial_health": {"rating": "weak"},
            },
            investor_profile={"risk_tolerance": "conservative"},
        )
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is True
        assert score.score is None
        assert score.level is None

    def test_conservative_investor_profile_amplifies_existing_quantitative_score(
        self,
    ) -> None:
        """Verify conservative profile increases score when genuine metrics exist."""
        input_base = RiskAnalystInput.from_partial_results(
            ticker="TEST",
            technical_output={
                "ticker": "TEST",
                "rsi": {"value": 50.0},
                "technical_score": 50.0,
            },
            investor_profile=None,
        )
        input_cons = RiskAnalystInput.from_partial_results(
            ticker="TEST",
            technical_output={
                "ticker": "TEST",
                "rsi": {"value": 50.0},
                "technical_score": 50.0,
            },
            investor_profile={"risk_tolerance": "conservative"},
        )
        base_score = calculate_deterministic_risk_score(input_base)
        cons_score = calculate_deterministic_risk_score(input_cons)

        assert base_score.score is not None
        assert cons_score.score is not None
        assert cons_score.score > base_score.score
        assert cons_score.investor_penalty_multiplier == 1.25

    def test_aggressive_investor_profile_dampens_existing_quantitative_score(
        self,
    ) -> None:
        """Verify aggressive profile reduces score when genuine metrics exist."""
        input_base = RiskAnalystInput.from_partial_results(
            ticker="TEST",
            technical_output={
                "ticker": "TEST",
                "rsi": {"value": 50.0},
                "technical_score": 50.0,
            },
            investor_profile=None,
        )
        input_agg = RiskAnalystInput.from_partial_results(
            ticker="TEST",
            technical_output={
                "ticker": "TEST",
                "rsi": {"value": 50.0},
                "technical_score": 50.0,
            },
            investor_profile={"risk_tolerance": "aggressive"},
        )
        base_score = calculate_deterministic_risk_score(input_base)
        agg_score = calculate_deterministic_risk_score(input_agg)

        assert base_score.score is not None
        assert agg_score.score is not None
        assert agg_score.score < base_score.score
        assert agg_score.investor_penalty_multiplier == 0.80

    def test_score_rsi_boundary_conditions(self) -> None:
        """Verify score_rsi handles out-of-range or None values cleanly."""
        assert score_rsi(None) is None
        assert score_rsi(-5.0) is None
        assert score_rsi(105.0) is None
        assert score_rsi(50.0) == 0.20
        assert score_rsi(35.0) == 0.45
        assert score_rsi(75.0) == 0.70
        assert score_rsi(15.0) == 0.85

    def test_score_technical_score_boundary_conditions(self) -> None:
        """Verify score_technical_score handles out-of-range or None values cleanly."""
        assert score_technical_score(None) is None
        assert score_technical_score(-10.0) is None
        assert score_technical_score(110.0) is None
        assert score_technical_score(100.0) == 0.0
        assert score_technical_score(0.0) == 1.0
        assert score_technical_score(50.0) == 0.5


# ===========================================================================
# 2. STRUCTURED OUTPUT VALIDATION TESTS (10.3.4)
# ===========================================================================


class TestRiskAnalysisValidation:
    """Tests the structured output validator and safety reconciler."""

    def test_valid_output_passes_validation(self) -> None:
        """Verify a well-grounded, non-advisory RiskAnalysisOutput passes validation."""
        output = _build_sample_valid_output("AAPL")
        input_data = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={"ticker": "AAPL", "trend": "sideways"},
            fundamental_output={
                "ticker": "AAPL",
                "financial_health": {"rating": "strong"},
            },
            news_output={"ticker": "AAPL", "overall_sentiment": "positive"},
            investor_profile={"time_horizon": "6 months"},
        )
        validated = validate_risk_analysis(output, input_data)
        assert validated.ticker == "AAPL"
        assert validated.overall_risk_level == RiskSeverity.MODERATE

    @pytest.mark.parametrize(
        "bad_phrase",
        [
            "We issue a buy recommendation for Apple.",
            "Our strong sell recommendation is based on macro weakness.",
            "Target price is $250.",
            "The price target indicates 20% upside.",
            "This asset provides guaranteed returns.",
            "An entirely risk-free investment profile.",
            "Investors must buy shares before earnings.",
            "You should sell your holdings.",
        ],
    )
    def test_prohibited_phrases_in_summary_rejected(self, bad_phrase: str) -> None:
        """Verify prohibited advisory phrases in summary trigger error."""
        with pytest.raises(RiskAnalysisValidationError) as exc_info:
            _build_sample_valid_output("AAPL", summary=bad_phrase)

        assert "prohibited advisory phrase" in str(exc_info.value).lower()

    def test_prohibited_phrases_in_risk_factor_rejected(self) -> None:
        """Verify prohibited advisory phrases in factor descriptions trigger error."""
        output = _build_sample_valid_output("AAPL")
        output.market_risks[0].description = (
            "Investors must sell before the market crashes."
        )
        input_data = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={"ticker": "AAPL"},
            fundamental_output={"ticker": "AAPL"},
            news_output={"ticker": "AAPL"},
            investor_profile={"time_horizon": "6 months"},
        )

        with pytest.raises(RiskAnalysisValidationError) as exc_info:
            validate_risk_analysis(output, input_data)

        assert "prohibited advisory" in str(exc_info.value).lower()

    def test_missing_evidence_in_factual_factor_raises_error(self) -> None:
        """Verify factual factor with empty evidence list triggers validation error."""
        output = _build_sample_valid_output("AAPL")
        output.company_risks[0].evidence = []  # Omit required evidence
        input_data = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={"ticker": "AAPL"},
            fundamental_output={"ticker": "AAPL"},
            news_output={"ticker": "AAPL"},
            investor_profile={"time_horizon": "6 months"},
        )

        with pytest.raises(RiskAnalysisValidationError) as exc_info:
            validate_risk_analysis(output, input_data)

        assert "lacks supporting evidence citations" in str(exc_info.value)

    def test_citing_unavailable_specialist_raises_error(self) -> None:
        """Verify factor citing a specialist not provided triggers validation error."""
        output = _build_sample_valid_output("AAPL")
        # Input only has technical; fundamental, news, and investor_profile are missing
        input_data = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={"ticker": "AAPL", "trend": "uptrend"},
        )

        with pytest.raises(RiskAnalysisValidationError) as exc_info:
            validate_risk_analysis(output, input_data)

        assert "which was not provided in the input" in str(exc_info.value)

    def test_category_list_reconciles_mismatched_factor_category(self) -> None:
        """Verify factor placed in market_risks has its category aligned to MARKET."""
        output = _build_sample_valid_output("AAPL")
        output.market_risks[0].category = RiskCategory.FINANCIAL

        input_data = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={"ticker": "AAPL", "trend": "sideways"},
            fundamental_output={"ticker": "AAPL"},
            news_output={"ticker": "AAPL"},
            investor_profile={"time_horizon": "6 months"},
        )
        validated = validate_risk_analysis(output, input_data)
        assert validated.market_risks[0].category == RiskCategory.MARKET

    def test_empty_input_handled_gracefully(self) -> None:
        """Verify empty input sets insufficient_data=True, overall_risk_level=None."""
        output = RiskAnalysisOutput(
            ticker="AAPL",
            overall_risk_level=None,
            insufficient_data=True,
            summary="No upstream specialist signals were available.",
            confidence=0.0,
        )
        input_data = RiskAnalystInput(ticker="AAPL")

        validated = validate_risk_analysis(output, input_data)
        assert validated.insufficient_data is True
        assert validated.overall_risk_level is None
        assert validated.confidence == 0.0

    def test_sparse_data_caps_confidence(self) -> None:
        """Verify input with <50% data completeness caps confidence at 0.65."""
        output = _build_sample_valid_output("AAPL")
        output.confidence = 0.95
        # Only technical is provided (1/5 = 20% completeness)
        input_data = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={"ticker": "AAPL", "trend": "uptrend"},
        )
        output.company_risks[0].evidence = [
            RiskEvidenceRef(
                source_type="technical",
                reference_id="trend_1",
                detail="Technical trend.",
            )
        ]
        output.sector_risks[0].evidence = [
            RiskEvidenceRef(
                source_type="technical",
                reference_id="trend_1",
                detail="Technical trend.",
            )
        ]
        output.financial_risks[0].evidence = [
            RiskEvidenceRef(
                source_type="technical",
                reference_id="trend_1",
                detail="Technical trend.",
            )
        ]
        output.investor_specific_risks[0].evidence = [
            RiskEvidenceRef(
                source_type="technical",
                reference_id="trend_1",
                detail="Technical trend.",
            )
        ]

        validated = validate_risk_analysis(output, input_data)
        assert validated.confidence <= 0.65

    def test_qualitative_risk_factors_remain_usable_when_quantitative_metrics_absent(
        self,
    ) -> None:
        """Verify LLM can analyze qualitative factors even if numeric metrics absent."""
        output = _build_sample_valid_output("AAPL")
        input_data = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            fundamental_output={
                "ticker": "AAPL",
                "financial_health": {"rating": "weak"},
            },
            news_output={"ticker": "AAPL", "overall_sentiment": "negative"},
            technical_output={"ticker": "AAPL", "trend": "downtrend"},
            investor_profile={"time_horizon": "1 year"},
        )
        # Quantitative scoring baseline is absent (no numeric RSI / score)
        det_score = calculate_deterministic_risk_score(input_data)
        assert det_score.insufficient_data is True

        validated = validate_risk_analysis(
            output, input_data, deterministic_assessment=det_score
        )
        assert validated.ticker == "AAPL"
        assert validated.deterministic_risk_score is None
        assert validated.deterministic_risk_level is None
        # Grounded qualitative risk factors are successfully preserved
        assert len(validated.financial_risks) > 0
        assert len(validated.company_risks) > 0


# ===========================================================================
# 3. AGENT EXECUTION TESTS (10.3.1, 10.3.2)
# ===========================================================================


class TestRiskAnalystAgent:
    """Tests the end-to-end RiskAnalystAgent and risk_analyst_node adapter."""

    def test_agent_run_success_with_quantitative_and_qualitative_signals(self) -> None:
        """Verify agent execution with genuine numeric and qualitative signals."""
        sample_output = _build_sample_valid_output("NVDA")
        mock_provider = MockRiskLLMProvider(
            output_payload=sample_output.model_dump(mode="json")
        )
        agent = RiskAnalystAgent(provider=mock_provider)

        input_data = RiskAnalystInput.from_partial_results(
            ticker="NVDA",
            technical_output={
                "ticker": "NVDA",
                "trend": "uptrend",
                "rsi": {"value": 55.0},
                "technical_score": 75.0,
            },
            fundamental_output={
                "ticker": "NVDA",
                "financial_health": {"rating": "strong"},
            },
            news_output={"ticker": "NVDA", "overall_sentiment": "positive"},
            investor_profile={"time_horizon": "3 years", "risk_tolerance": "moderate"},
        )

        result = agent.run(input_data)

        assert result.success is True
        assert isinstance(result.data, RiskAnalysisOutput)
        assert result.data.ticker == "NVDA"
        # Genuine quantitative baseline is populated
        assert result.data.deterministic_risk_score is not None
        assert result.data.deterministic_risk_level is not None
        assert mock_provider.last_prompt is not None
        assert "NVDA" in mock_provider.last_prompt

    def test_agent_run_success_with_qualitative_signals_only(self) -> None:
        """Verify agent execution with qualitative signals only sets score to None."""
        sample_output = _build_sample_valid_output("NVDA")
        mock_provider = MockRiskLLMProvider(
            output_payload=sample_output.model_dump(mode="json")
        )
        agent = RiskAnalystAgent(provider=mock_provider)

        input_data = RiskAnalystInput.from_partial_results(
            ticker="NVDA",
            technical_output={"ticker": "NVDA", "trend": "uptrend"},
            fundamental_output={
                "ticker": "NVDA",
                "financial_health": {"rating": "strong"},
            },
            news_output={"ticker": "NVDA", "overall_sentiment": "positive"},
            investor_profile={"time_horizon": "3 years", "risk_tolerance": "moderate"},
        )

        result = agent.run(input_data)

        assert result.success is True
        assert isinstance(result.data, RiskAnalysisOutput)
        assert result.data.ticker == "NVDA"
        # No genuine numeric metrics were provided, so deterministic score must be None
        assert result.data.deterministic_risk_score is None
        assert result.data.deterministic_risk_level is None
        assert mock_provider.last_prompt is not None

    def test_agent_run_completely_empty_input_deterministic_fallback(self) -> None:
        """Verify empty input does not call LLM and returns insufficient data result."""
        mock_provider = MockRiskLLMProvider()
        agent = RiskAnalystAgent(provider=mock_provider)

        input_data = RiskAnalystInput(ticker="EMPTY")
        result = agent.run(input_data)

        assert result.success is True
        assert result.data.insufficient_data is True
        assert result.data.overall_risk_level is None
        assert result.data.deterministic_risk_score is None
        assert result.data.confidence == 0.0
        # Mock provider was never invoked
        assert mock_provider.last_prompt is None

    def test_agent_run_handles_llm_failure_gracefully(self) -> None:
        """Verify LLM provider failure returns AgentResult failure without throwing."""
        mock_provider = MockRiskLLMProvider(
            should_fail=True,
            error_message="API quota exceeded",
        )
        agent = RiskAnalystAgent(provider=mock_provider)

        input_data = RiskAnalystInput.from_partial_results(
            ticker="FAILCORP",
            technical_output={"ticker": "FAILCORP"},
        )
        result = agent.run(input_data)

        assert result.success is False
        assert "API quota exceeded" in str(result.error)

    def test_agent_run_catches_prohibited_advice_from_llm(self) -> None:
        """Verify LLM hallucinating buy recommendation causes AgentResult failure."""
        bad_output_dict = _build_sample_valid_output("BAD").model_dump(mode="json")
        bad_output_dict["summary"] = "Investors must buy this stock at current levels."
        mock_provider = MockRiskLLMProvider(output_payload=bad_output_dict)
        agent = RiskAnalystAgent(provider=mock_provider)

        input_data = RiskAnalystInput.from_partial_results(
            ticker="BAD",
            technical_output={"ticker": "BAD"},
            fundamental_output={"ticker": "BAD"},
            news_output={"ticker": "BAD"},
        )
        result = agent.run(input_data)

        assert result.success is False
        assert "prohibited advisory" in str(result.error).lower()

    def test_risk_analyst_node_updates_graph_state(self) -> None:
        """Verify risk_analyst_node executes and updates 'risk_result' in GraphState."""
        sample_output = _build_sample_valid_output("MSFT")
        mock_provider = MockRiskLLMProvider(
            output_payload=sample_output.model_dump(mode="json")
        )
        agent = RiskAnalystAgent(provider=mock_provider)

        state = {
            "cio_decision": {"ticker": "MSFT"},
            "technical_result": {
                "success": True,
                "data": {"ticker": "MSFT", "trend": "uptrend"},
            },
            "fundamental_result": {
                "success": True,
                "data": {"ticker": "MSFT", "financial_health": {"rating": "strong"}},
            },
            "news_result": {
                "success": True,
                "data": {"ticker": "MSFT", "overall_sentiment": "positive"},
            },
            "investor_profile": {"time_horizon": "5 years"},
        }

        update = risk_analyst_node(state, agent=agent)

        assert "risk_result" in update
        assert update["risk_result"]["ticker"] == "MSFT"
        assert update["risk_result"]["overall_risk_level"] is not None
