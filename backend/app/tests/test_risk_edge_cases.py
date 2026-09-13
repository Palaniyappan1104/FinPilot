"""Comprehensive edge case and robustness tests for the Risk Analyst (Phase 10.4).

Fulfills plan.md Phase 10.4:
- 10.4.1 Unit tests with varying investor profiles against the same company
  (verifying quantitative scores and qualitative interpretations differ).
- 10.4.2 Unit tests with missing upstream data (verifying graceful degradation).
- 10.4.3 Edge cases: highly volatile stock + low risk tolerance; stable stock +
  high risk tolerance.

Also rigorously covers:
- Missing, partial, or failed upstream specialist outputs in GraphState.
- Conflicting signals across specialists (bullish technical vs. distressed fundamental).
- Invalid, NaN, infinite, negative, or out-of-range numeric inputs.
- Malformed structured LLM outputs, LLM timeouts, and provider errors.
- Unrecognized or extreme investor profile fields (negative capital, strange tolerance).
- Provenance grounding violations, fabricated sources, and prohibited advice variants.
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
    RiskInvestorProfile,
    RiskSeverity,
)
from app.agents.risk_scoring import (
    calculate_deterministic_risk_score,
    calculate_investor_multiplier,
    map_score_to_severity,
    score_rsi,
    score_technical_score,
)
from app.core.llm.base import LLMProvider, LLMResponse
from app.core.llm.exceptions import LLMError

# ===========================================================================
# OFFLINE TEST HELPERS & MOCK PROVIDER
# ===========================================================================


class MockEdgeCaseLLMProvider(LLMProvider):
    """Mock LLM provider supporting dynamic responses and simulated failure modes."""

    def __init__(
        self,
        output_payload: Optional[Dict[str, Any]] = None,
        raw_content: Optional[str] = None,
        should_fail: bool = False,
        error_message: str = "Simulated edge-case failure",
    ) -> None:
        self.output_payload = output_payload or {}
        self.raw_content = raw_content
        self.should_fail = should_fail
        self.error_message = error_message
        self.last_prompt: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "mock_edge_case_llm"

    def generate(
        self,
        prompt: str,
        schema: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.last_prompt = prompt
        if self.should_fail:
            raise LLMError(self.error_message)
        import json

        if self.raw_content is not None:
            content = self.raw_content
        else:
            content = json.dumps(self.output_payload)

        return LLMResponse(
            content=content,
            model="mock-edge-model",
            provider=self.provider_name,
            metadata={},
        )


def _build_grounded_output(
    ticker: str = "TEST",
    overall_level: RiskSeverity = RiskSeverity.MODERATE,
    summary: str = "Objective evaluation of financial and technical factors.",
) -> RiskAnalysisOutput:
    """Create a fully grounded baseline output for test mutation."""
    return RiskAnalysisOutput(
        ticker=ticker,
        overall_risk_level=overall_level,
        market_risks=[
            RiskFactor(
                category=RiskCategory.MARKET,
                name="Market Cycle Posture",
                description=(
                    "Technical trend suggests potential broader equity pullback."
                ),
                severity=RiskSeverity.MODERATE,
                evidence=[
                    RiskEvidenceRef(
                        source_type="technical",
                        reference_id="trend_1",
                        detail="Technical trend reported as sideways consolidation.",
                    )
                ],
            )
        ],
        company_risks=[
            RiskFactor(
                category=RiskCategory.COMPANY,
                name="Margin Pressure",
                description="Operating headwinds noted in fundamental filings.",
                severity=RiskSeverity.MODERATE,
                evidence=[
                    RiskEvidenceRef(
                        source_type="fundamental",
                        reference_id="margin_1",
                        detail="Key weakness reported in margin compression.",
                    )
                ],
            )
        ],
        sector_risks=[
            RiskFactor(
                category=RiskCategory.SECTOR,
                name="Regulatory Shift",
                description="Sector regulatory headline noted in news.",
                severity=RiskSeverity.HIGH,
                evidence=[
                    RiskEvidenceRef(
                        source_type="news",
                        reference_id="reg_1",
                        detail="Industry regulatory inquiry headline.",
                    )
                ],
            )
        ],
        financial_risks=[
            RiskFactor(
                category=RiskCategory.FINANCIAL,
                name="Leverage Level",
                description="Debt is manageable under current coverage.",
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
                name="Momentum Reading",
                description="RSI is 55.0, within stable momentum bounds.",
                severity=RiskSeverity.LOW,
                evidence=[
                    RiskEvidenceRef(
                        source_type="technical",
                        reference_id="rsi_1",
                        detail="RSI indicator reading is 55.0.",
                    )
                ],
            )
        ],
        investor_specific_risks=[
            RiskFactor(
                category=RiskCategory.INVESTOR_SPECIFIC,
                name="Horizon Alignment",
                description="Time horizon aligns with cyclical duration.",
                severity=RiskSeverity.LOW,
                evidence=[
                    RiskEvidenceRef(
                        source_type="investor_profile",
                        reference_id="horizon_1",
                        detail="Stated time horizon is 3 years.",
                    )
                ],
            )
        ],
        summary=summary,
        confidence=0.85,
    )


# ===========================================================================
# 1. PLAN.MD 10.4 EXPLICIT REQUIREMENTS TESTS (10.4.1, 10.4.2, 10.4.3)
# ===========================================================================


class TestPlan10_4Requirements:
    """Directly verifies plan.md 10.4.1, 10.4.2, and 10.4.3 specifications."""

    def test_10_4_1_varying_investor_profiles_against_same_company(self) -> None:
        """10.4.1: Verify risk output changes across different profiles."""
        ticker = "TSLA"
        tech_payload = {
            "ticker": ticker,
            "rsi": {"value": 65.0},
            "technical_score": 50.0,
        }

        # 1. Conservative investor
        input_cons = RiskAnalystInput.from_partial_results(
            ticker=ticker,
            technical_output=tech_payload,
            investor_profile={
                "risk_tolerance": "conservative",
                "time_horizon": "1 year",
            },
        )
        score_cons = calculate_deterministic_risk_score(input_cons)

        # 2. Moderate investor
        input_mod = RiskAnalystInput.from_partial_results(
            ticker=ticker,
            technical_output=tech_payload,
            investor_profile={"risk_tolerance": "moderate", "time_horizon": "3 years"},
        )
        score_mod = calculate_deterministic_risk_score(input_mod)

        # 3. Aggressive investor
        input_agg = RiskAnalystInput.from_partial_results(
            ticker=ticker,
            technical_output=tech_payload,
            investor_profile={
                "risk_tolerance": "aggressive",
                "time_horizon": "5 years",
            },
        )
        score_agg = calculate_deterministic_risk_score(input_agg)

        # Assert scores strictly differ and order correctly:
        # Conservative > Moderate > Aggressive
        assert score_cons.score is not None
        assert score_mod.score is not None
        assert score_agg.score is not None

        assert score_cons.score > score_mod.score
        assert score_mod.score > score_agg.score

        assert score_cons.investor_penalty_multiplier == 1.25
        assert score_mod.investor_penalty_multiplier == 1.00
        assert (
            score_agg.investor_penalty_multiplier == 0.72
        )  # 0.80 * 0.90 (long horizon)

    def test_10_4_2_missing_upstream_data_graceful_degradation(self) -> None:
        """10.4.2: Verify loss of upstream data degrades confidence gracefully."""
        output = _build_grounded_output("AAPL")

        # Full 5 sources available
        input_full = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={"ticker": "AAPL"},
            fundamental_output={"ticker": "AAPL"},
            news_output={"ticker": "AAPL"},
            research_output={"key_findings": [{"claim": "Patent filing"}]},
            investor_profile={"risk_tolerance": "moderate"},
        )
        val_full = validate_risk_analysis(output.model_copy(deep=True), input_full)
        assert val_full.confidence == 0.85

        # 1 primary source available (1/3 = 33% completeness < 50% threshold)
        input_sparse = RiskAnalystInput.from_partial_results(
            ticker="AAPL",
            technical_output={"ticker": "AAPL"},
        )
        output_sparse = output.model_copy(deep=True)
        # Retarget factors to allowed source (technical)
        for factor_list in [
            output_sparse.market_risks,
            output_sparse.company_risks,
            output_sparse.sector_risks,
            output_sparse.financial_risks,
            output_sparse.volatility_risks,
            output_sparse.investor_specific_risks,
        ]:
            factor_list[0].evidence = [
                RiskEvidenceRef(
                    source_type="technical", reference_id="tech_1", detail="Detail"
                )
            ]

        val_sparse = validate_risk_analysis(output_sparse, input_sparse)
        assert val_sparse.confidence <= 0.65

        # 0 sources available (empty input)
        input_empty = RiskAnalystInput(ticker="AAPL")
        empty_out = RiskAnalysisOutput(
            ticker="AAPL",
            overall_risk_level=None,
            insufficient_data=True,
            summary="Empty",
            confidence=0.0,
        )
        val_empty = validate_risk_analysis(empty_out, input_empty)
        assert val_empty.confidence == 0.0
        assert val_empty.insufficient_data is True
        assert val_empty.overall_risk_level is None

    def test_10_4_3_volatile_stock_with_low_risk_tolerance(self) -> None:
        """10.4.3: Volatile stock + low risk tolerance gives critical risk."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="MEME",
            technical_output={
                "ticker": "MEME",
                "rsi": {"value": 85.0},  # Overbought / extreme (0.85)
                "technical_score": 10.0,  # Severe breakdown ((100-10)/100 = 0.90)
            },
            investor_profile={
                "risk_tolerance": "conservative",
                "time_horizon": "1 month",
            },
        )
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is False
        assert score.score is not None
        assert score.score >= 0.90
        assert score.level == RiskSeverity.CRITICAL
        # 1.25 (conservative) * 1.15 (short horizon on high risk) = ~1.44x multiplier
        assert score.investor_penalty_multiplier > 1.25

    def test_10_4_3_stable_stock_with_high_risk_tolerance(self) -> None:
        """10.4.3: Stable stock + high risk tolerance produces dampened low risk."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="BLUECHIP",
            technical_output={
                "ticker": "BLUECHIP",
                "rsi": {"value": 50.0},  # Stable neutral (0.20)
                "technical_score": 90.0,  # Strong posture (0.10)
            },
            investor_profile={
                "risk_tolerance": "aggressive",
                "time_horizon": "10 years",
            },
        )
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is False
        assert score.score is not None
        assert score.score < 0.20
        assert score.level == RiskSeverity.LOW
        # 0.80 (aggressive) * 0.90 (long horizon) = 0.72x multiplier
        assert score.investor_penalty_multiplier == 0.72


# ===========================================================================
# 2. NUMERIC EDGE CASES & SANITIZATION (NaN, Inf, Out-of-Range)
# ===========================================================================


class TestNumericSanitizationEdgeCases:
    """Verifies invalid, NaN, infinite, negative, or out-of-range floats."""

    def test_rsi_nan_and_inf_returns_none(self) -> None:
        """Verify score_rsi rejects NaN, +Inf, and -Inf without raising exceptions."""
        assert score_rsi(float("nan")) is None
        assert score_rsi(float("inf")) is None
        assert score_rsi(float("-inf")) is None

    def test_rsi_negative_and_out_of_range_returns_none(self) -> None:
        """Verify score_rsi rejects values outside [0.0, 100.0]."""
        assert score_rsi(-0.01) is None
        assert score_rsi(-100.0) is None
        assert score_rsi(100.01) is None
        assert score_rsi(500.0) is None

    def test_technical_score_nan_and_inf_returns_none(self) -> None:
        """Verify score_technical_score rejects NaN, +Inf, and -Inf."""
        assert score_technical_score(float("nan")) is None
        assert score_technical_score(float("inf")) is None
        assert score_technical_score(float("-inf")) is None

    def test_technical_score_out_of_range_returns_none(self) -> None:
        """Verify score_technical_score rejects values outside [0.0, 100.0]."""
        assert score_technical_score(-1.0) is None
        assert score_technical_score(100.1) is None

    def test_both_metrics_nan_yields_insufficient_data(self) -> None:
        """Verify input with all NaN numeric indicators yields insufficient_data."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="NAN_CORP",
            technical_output={
                "ticker": "NAN_CORP",
                "rsi": {"value": float("nan")},
                "technical_score": float("nan"),
            },
        )
        score = calculate_deterministic_risk_score(input_data)

        assert score.insufficient_data is True
        assert score.score is None
        assert score.level is None

    def test_investor_multiplier_handles_nan_base_score_gracefully(self) -> None:
        """Verify calculate_investor_multiplier returns 1.0 if base_score is NaN/Inf."""
        input_data = RiskAnalystInput(ticker="TEST")
        assert calculate_investor_multiplier(input_data, float("nan")) == 1.0
        assert calculate_investor_multiplier(input_data, float("inf")) == 1.0
        assert calculate_investor_multiplier(input_data, None) == 1.0

    def test_map_score_to_severity_boundary_invariants(self) -> None:
        """Verify score to severity boundary mapping across standard thresholds."""
        assert map_score_to_severity(0.0) == RiskSeverity.LOW
        assert map_score_to_severity(0.299) == RiskSeverity.LOW
        assert map_score_to_severity(0.30) == RiskSeverity.MODERATE
        assert map_score_to_severity(0.599) == RiskSeverity.MODERATE
        assert map_score_to_severity(0.60) == RiskSeverity.HIGH
        assert map_score_to_severity(0.849) == RiskSeverity.HIGH
        assert map_score_to_severity(0.85) == RiskSeverity.CRITICAL
        assert map_score_to_severity(1.0) == RiskSeverity.CRITICAL


# ===========================================================================
# 3. CONFLICTING SIGNALS ACROSS SPECIALISTS
# ===========================================================================


class TestConflictingSpecialistSignals:
    """Verifies synthesis when specialists report sharply opposing signals."""

    def test_bullish_technicals_vs_distressed_fundamentals(self) -> None:
        """Verify quantitative reflects technicals while distress is captured."""
        input_data = RiskAnalystInput.from_partial_results(
            ticker="CONFLICT",
            technical_output={
                "ticker": "CONFLICT",
                "trend": "uptrend",
                "rsi": {"value": 52.0},  # Low technical risk
                "technical_score": 85.0,  # Low technical risk
            },
            fundamental_output={
                "ticker": "CONFLICT",
                "financial_health": {"rating": "weak"},
                "leverage_assessment": {"rating": "weak"},
                "key_weaknesses": ["Severe solvency pressure", "Debt default risk"],
            },
            news_output={
                "ticker": "CONFLICT",
                "overall_sentiment": "negative",
                "negative_factors": [{"text": "Credit rating downgraded to junk"}],
            },
            investor_profile={"risk_tolerance": "conservative"},
        )

        # Quantitative baseline computes strictly from genuine technical metrics
        det_score = calculate_deterministic_risk_score(input_data)
        assert det_score.insufficient_data is False
        assert det_score.score is not None
        # Technical metrics are bullish, so quantitative technical risk baseline is low
        assert det_score.score <= 0.30

        # LLM receives both the low technical score AND the severe fundamental distress
        output_payload = _build_grounded_output("CONFLICT")
        output_payload.financial_risks[0].severity = RiskSeverity.CRITICAL
        output_payload.financial_risks[0].name = "Insolvency Risk"
        output_payload.financial_risks[0].evidence = [
            RiskEvidenceRef(
                source_type="fundamental",
                reference_id="weak_1",
                detail="Default risk reported in fundamentals.",
            )
        ]
        # LLM synthesizes overall risk as HIGH despite low technical score
        output_payload.overall_risk_level = RiskSeverity.HIGH

        mock_provider = MockEdgeCaseLLMProvider(
            output_payload=output_payload.model_dump(mode="json")
        )
        agent = RiskAnalystAgent(provider=mock_provider)
        result = agent.run(input_data)

        assert result.success is True
        # Quantitative score preserves deterministic technical math
        assert result.data.deterministic_risk_score <= 0.30
        # Qualitative factors appropriately reflect fundamental insolvency
        assert result.data.financial_risks[0].severity == RiskSeverity.CRITICAL


# ===========================================================================
# 4. INVESTOR PROFILE EDGE CASES
# ===========================================================================


class TestInvestorProfileEdgeCases:
    """Verifies extreme, invalid, or malformed investor profile context."""

    def test_negative_or_nan_capital_amount_sanitized_to_none(self) -> None:
        """Verify negative or NaN capital_amount in dict is filtered to None."""
        prof_negative = RiskInvestorProfile.from_investor_profile(
            {"capital_amount": -50000.0, "risk_tolerance": "conservative"}
        )
        assert prof_negative is not None
        assert prof_negative.capital_amount is None

        prof_nan = RiskInvestorProfile.from_investor_profile(
            {"capital_amount": float("nan"), "risk_tolerance": "moderate"}
        )
        assert prof_nan is not None
        assert prof_nan.capital_amount is None

        prof_inf = RiskInvestorProfile.from_investor_profile(
            {"capital_amount": float("inf"), "risk_tolerance": "moderate"}
        )
        assert prof_inf is not None
        assert prof_inf.capital_amount is None

    def test_unrecognized_risk_tolerance_defaults_to_neutral_multiplier(self) -> None:
        """Verify bizarre risk tolerance strings default to multiplier 1.0."""
        for strange_tol in ["yolo", "diamond_hands", "12345", ""]:
            input_data = RiskAnalystInput.from_partial_results(
                ticker="TEST",
                technical_output={
                    "ticker": "TEST",
                    "rsi": {"value": 50.0},
                    "technical_score": 50.0,
                },
                investor_profile={"risk_tolerance": strange_tol},
            )
            score = calculate_deterministic_risk_score(input_data)
            assert score.investor_penalty_multiplier == 1.0

    def test_extremely_large_capital_amount_handled_safely(self) -> None:
        """Verify astronomical capital figures (e.g. $100T) do not overflow."""
        huge_amount = 1e14  # $100 Trillion
        prof = RiskInvestorProfile.from_investor_profile(
            {"capital_amount": huge_amount, "risk_tolerance": "moderate"}
        )
        assert prof is not None
        assert prof.capital_amount == huge_amount


# ===========================================================================
# 5. GRAPHSTATE ADAPTER & PARTIAL FAILURE EDGE CASES
# ===========================================================================


class TestGraphStatePartialFailures:
    """Verifies risk_analyst_node handling of failed or malformed GraphState inputs."""

    def test_specialist_failure_in_graphstate_handled_gracefully(self) -> None:
        """Verify specialist node returning success=False does not crash node."""
        state = {
            "cio_decision": {"ticker": "DEGRADED"},
            "technical_result": {
                "success": False,
                "error": "Timeout connecting to technical indicator service",
            },
            "fundamental_result": {
                "success": True,
                "data": {
                    "ticker": "DEGRADED",
                    "financial_health": {"rating": "strong"},
                },
            },
            "news_result": {
                "success": False,
                "error": "News provider 503 unavailable",
            },
            "investor_profile": {"risk_tolerance": "moderate"},
        }

        sample_out = _build_grounded_output("DEGRADED")
        # Retarget evidence to fundamental (since technical and news failed)
        for cat in [
            sample_out.market_risks,
            sample_out.company_risks,
            sample_out.sector_risks,
            sample_out.financial_risks,
            sample_out.volatility_risks,
        ]:
            cat[0].evidence = [
                RiskEvidenceRef(
                    source_type="fundamental", reference_id="fund_1", detail="OK"
                )
            ]

        mock_provider = MockEdgeCaseLLMProvider(
            output_payload=sample_out.model_dump(mode="json")
        )
        agent = RiskAnalystAgent(provider=mock_provider)

        update = risk_analyst_node(state, agent=agent)

        assert "risk_result" in update
        assert update["risk_result"]["ticker"] == "DEGRADED"
        assert update["risk_result"]["data_completeness"]["technical"] is False
        assert update["risk_result"]["data_completeness"]["fundamental"] is True
        assert update["risk_result"]["data_completeness"]["news"] is False

    def test_missing_ticker_in_graphstate_returns_failure_result(self) -> None:
        """Verify state missing ticker in cio_decision and specialists fails cleanly."""
        empty_state = {"messages": []}
        update = risk_analyst_node(empty_state)

        assert "risk_result" in update
        assert update["risk_result"]["success"] is False
        assert "ticker" in str(update["risk_result"]["error"]).lower()


# ===========================================================================
# 6. SAFETY, PROHIBITED ADVICE & PROVENANCE EDGE CASES
# ===========================================================================


class TestSafetyAndProvenanceEdgeCases:
    """Verifies edge cases in prohibited advice and evidence citations."""

    @pytest.mark.parametrize(
        "sneaky_advisory_phrase",
        [
            "Strong BUY recommendation for long term accounts.",
            "We provide a SELL recommendation due to sector headwinds.",
            "Investors must buy ahead of next quarter.",
            "You should sell immediately.",
            "Guaranteed returns of 15% annually.",
            "Risk-free yield opportunity.",
            "Target price of $180 per share.",
            "The price target indicates upside potential.",
        ],
    )
    def test_obfuscated_and_mixed_case_advisory_language_caught(
        self, sneaky_advisory_phrase: str
    ) -> None:
        """Verify regex patterns catch variations of prohibited advice."""
        with pytest.raises(RiskAnalysisValidationError):
            _build_grounded_output("TEST", summary=sneaky_advisory_phrase)

    def test_fabricating_unprovided_source_type_rejected(self) -> None:
        """Verify citation with source type not present in input is rejected."""
        output = _build_grounded_output("TEST")
        # Evidence cites 'news', but input provides only 'technical'
        input_data = RiskAnalystInput.from_partial_results(
            ticker="TEST",
            technical_output={"ticker": "TEST"},
        )
        with pytest.raises(RiskAnalysisValidationError) as exc:
            validate_risk_analysis(output, input_data)
        assert "which was not provided in the input" in str(exc.value)

    def test_empty_or_whitespace_evidence_fields_rejected(self) -> None:
        """Verify RiskEvidenceRef rejects whitespace-only reference_id or detail."""
        with pytest.raises(ValueError):
            RiskEvidenceRef(
                source_type="technical", reference_id="   ", detail="Detail"
            )

        with pytest.raises(ValueError):
            RiskEvidenceRef(source_type="technical", reference_id="ref_1", detail="   ")

    def test_insufficient_data_factor_does_not_require_evidence(self) -> None:
        """Verify factor with insufficient_data=True does not require citations."""
        output = _build_grounded_output("TEST")
        output.volatility_risks[0].insufficient_data = True
        output.volatility_risks[0].evidence = (
            []
        )  # Empty evidence is allowed when insufficient_data=True

        input_data = RiskAnalystInput.from_partial_results(
            ticker="TEST",
            technical_output={"ticker": "TEST"},
            fundamental_output={"ticker": "TEST"},
            news_output={"ticker": "TEST"},
            investor_profile={"risk_tolerance": "moderate"},
        )
        # Should not raise validation error
        validated = validate_risk_analysis(output, input_data)
        assert validated.volatility_risks[0].insufficient_data is True


# ===========================================================================
# 7. LLM STRUCTURED OUTPUT & PROVIDER FAILURE MODES
# ===========================================================================


class TestLLMStructuredOutputFailureModes:
    """Verifies behavior under corrupt, truncated, or failing LLM responses."""

    def test_malformed_json_response_returns_agent_failure(self) -> None:
        """Verify provider returning broken JSON string produces AgentResult failure."""
        mock_provider = MockEdgeCaseLLMProvider(raw_content="{'broken_json': true,")
        agent = RiskAnalystAgent(provider=mock_provider)

        input_data = RiskAnalystInput.from_partial_results(
            ticker="BADJSON",
            technical_output={"ticker": "BADJSON"},
        )
        result = agent.run(input_data)

        assert result.success is False
        assert (
            "structured validation failed" in str(result.error).lower()
            or "json" in str(result.error).lower()
        )

    def test_llm_timeout_exception_returns_clean_agent_failure(self) -> None:
        """Verify provider timeout returns AgentResult failure without throwing."""
        mock_provider = MockEdgeCaseLLMProvider(
            should_fail=True,
            error_message="Gateway Timeout 504: Model generation exceeded 30s",
        )
        agent = RiskAnalystAgent(provider=mock_provider)

        input_data = RiskAnalystInput.from_partial_results(
            ticker="TIMEOUT",
            technical_output={"ticker": "TIMEOUT"},
        )
        result = agent.run(input_data)

        assert result.success is False
        assert "Gateway Timeout 504" in str(result.error)
