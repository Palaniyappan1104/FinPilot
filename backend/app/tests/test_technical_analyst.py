"""Unit tests for Phase 7.3 Technical Analyst Agent.

All tests run 100% offline using deterministic mock LLM providers.
ZERO real Gemini API or network calls are made.

Verifies:
- Valid structured technical analysis output generation and parsing.
- LLM correctly interprets supplied trend and cannot alter it.
- LLM cannot alter deterministic technical_score.
- LLM cannot alter supplied numerical indicator values.
- Unsupported/invented values where None was supplied are rejected.
- Missing/null indicators are handled and preserved correctly.
- Ticker mismatch is rejected.
- Prohibited recommendation / price-target content is rejected.
- Confidence remains interpretation confidence and caps on sparse data.
- Insufficient-data TechnicalMetrics are handled safely.
- Malformed structured LLM output is rejected gracefully via AgentResult.
- Deterministic grounding validation behaves consistently.
- LangGraph node adapter contract.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from app.agents.state import GraphState
from app.agents.technical import (
    TechnicalAnalystAgent,
    format_technical_prompt,
    technical_analyst_node,
)
from app.agents.technical_schema import (
    SupportResistanceSummary,
    TechnicalAnalysisOutput,
    TechnicalAnalystInput,
)
from app.core.llm import (
    LLMError,
    LLMProvider,
    LLMResponse,
)
from app.models.technical_metrics import (
    MACDMetrics,
    MovingAverageMetrics,
    RSIMetrics,
    SupportResistanceMetrics,
    TechnicalMetrics,
    TechnicalScoreBreakdown,
    VolumeMetrics,
)


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for Phase 7.3 Technical Analyst tests."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_technical_provider",
    ) -> None:
        self.responses = responses or []
        self.call_count = 0
        self.prompts_received: List[str] = []
        self.fail_with = fail_with
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        schema: Optional[Any] = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)

        if self.fail_with is not None:
            raise self.fail_with

        if not self.responses:
            raise RuntimeError("MockLLMProvider: No responses queued.")

        resp_idx = min(self.call_count - 1, len(self.responses) - 1)
        content = self.responses[resp_idx]

        return LLMResponse(
            content=content,
            provider=self._name,
            model="mock-model",
        )


@pytest.fixture
def sample_technical_metrics() -> TechnicalMetrics:
    """Pre-calculated healthy technical metrics mirroring a strong uptrend."""
    return TechnicalMetrics(
        ticker="NVDA",
        latest_close=130.0,
        calculated_at=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
        candle_count=250,
        moving_averages=MovingAverageMetrics(
            sma_20=125.0,
            sma_50=120.0,
            sma_200=110.0,
            ema_20=126.0,
            ema_50=121.0,
            ema_200=111.0,
            ema_12=128.0,
            ema_26=123.0,
        ),
        rsi=RSIMetrics(rsi_14=62.5),
        macd=MACDMetrics(
            macd_line=2.5,
            signal_line=1.5,
            histogram=1.0,
            fast_period=12,
            slow_period=26,
            signal_period=9,
        ),
        volume=VolumeMetrics(
            latest_volume=50_000_000.0,
            average_volume_20d=40_000_000.0,
            volume_ratio=1.25,
        ),
        support_resistance=SupportResistanceMetrics(
            primary_support=122.0,
            primary_resistance=135.0,
            support_levels=[122.0, 115.0],
            resistance_levels=[135.0, 140.0],
        ),
        trend="uptrend",
        technical_score=85.0,
        score_breakdown=TechnicalScoreBreakdown(
            ma_alignment_score=25.0,
            rsi_score=25.0,
            macd_score=25.0,
            volume_score=20.0,
        ),
    )


@pytest.fixture
def valid_output_dict(sample_technical_metrics: TechnicalMetrics) -> Dict[str, Any]:
    """Valid structured output dictionary matching sample_technical_metrics."""
    m = sample_technical_metrics
    return {
        "ticker": m.ticker,
        "trend": m.trend,
        "indicators_summary": {
            "latest_close": m.latest_close,
            "moving_averages": m.moving_averages.model_dump(),
            "rsi": m.rsi.model_dump(),
            "macd": m.macd.model_dump(),
            "volume": m.volume.model_dump(),
        },
        "support_resistance": {
            "primary_support": m.support_resistance.primary_support,
            "primary_resistance": m.support_resistance.primary_resistance,
            "support_levels": m.support_resistance.support_levels,
            "resistance_levels": m.support_resistance.resistance_levels,
        },
        "technical_score": m.technical_score,
        "score_breakdown": (
            m.score_breakdown.model_dump() if m.score_breakdown else None
        ),
        "interpretation": {
            "overall_summary": (
                "NVDA exhibits a robust bullish posture across all core "
                "technical pillars."
            ),
            "trend_analysis": (
                "The stock is in a clear uptrend with price above ascending "
                "20, 50, and 200 SMAs."
            ),
            "moving_averages_analysis": (
                "Moving averages show strong stacked bullish alignment "
                "(SMA20 > SMA50 > SMA200)."
            ),
            "momentum_analysis": (
                "RSI at 62.5 indicates solid bullish momentum below overbought "
                "territory; MACD shows positive histogram expansion."
            ),
            "volume_analysis": (
                "Volume is 1.25x the 20-day average, confirming active "
                "institutional accumulation."
            ),
            "support_resistance_analysis": (
                "Nearest support is anchored at 122.0, with primary overhead "
                "resistance at 135.0."
            ),
        },
        "evidence": [
            "Price of 130.0 trades above SMA20 (125.0) and SMA50 (120.0).",
            "RSI-14 is in the constructive 60-70 momentum band at 62.5.",
            "MACD histogram is positive at 1.0 with MACD line above signal.",
            "Volume ratio of 1.25 confirms upward price movement.",
        ],
        "risks": [
            "Overhead resistance at 135.0 may prompt short-term profit taking.",
            (
                "A drop below the 122.0 primary support would threaten the "
                "short-term trend."
            ),
        ],
        "confidence": 0.90,
    }


# ---------------------------------------------------------------------------
# 1. Valid Structured Technical Analysis Output Tests
# ---------------------------------------------------------------------------


def test_valid_structured_technical_analysis_output(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Verify full end-to-end execution of TechnicalAnalystAgent with valid output."""
    mock_provider = MockLLMProvider(responses=[json.dumps(valid_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is True
    assert isinstance(result.data, TechnicalAnalysisOutput)
    output: TechnicalAnalysisOutput = result.data
    assert output.ticker == "NVDA"
    assert output.trend == "uptrend"
    assert output.technical_score == 85.0
    assert output.confidence == 0.90
    assert len(output.evidence) == 4
    assert len(output.risks) == 2
    assert mock_provider.call_count == 1


def test_prompt_contains_critical_instructions(
    sample_technical_metrics: TechnicalMetrics,
):
    """Verify format_technical_prompt injects required instructions and dynamic data."""
    input_data = TechnicalAnalystInput(
        metrics=sample_technical_metrics,
        time_horizon="6 months",
        risk_tolerance="moderate",
        task_description="Evaluate breakout probability near resistance.",
    )
    prompt = format_technical_prompt(input_data)

    assert "You are FinPilot's Technical Analysis interpretation specialist." in prompt
    assert "NEVER calculate an indicator yourself." in prompt
    assert "NEVER create a price target." in prompt
    assert "NEVER issue a buy/sell recommendation" in prompt
    assert "Ticker: NVDA" in prompt
    assert "Time Horizon: 6 months" in prompt
    assert "Risk Tolerance: moderate" in prompt
    assert "Evaluate breakout probability near resistance." in prompt
    assert "130.0" in prompt


# ---------------------------------------------------------------------------
# 2. Trend & Score Invariance Grounding Tests
# ---------------------------------------------------------------------------


def test_llm_cannot_alter_trend(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Validation must reject when LLM attempts to alter the deterministic trend."""
    bad_output_dict = dict(valid_output_dict)
    bad_output_dict["trend"] = "downtrend"  # metrics has 'uptrend'

    mock_provider = MockLLMProvider(responses=[json.dumps(bad_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "Trend mismatch" in result.error


def test_llm_cannot_alter_technical_score(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Validation must reject when LLM attempts to alter technical score."""
    bad_output_dict = dict(valid_output_dict)
    bad_output_dict["technical_score"] = 99.0  # metrics has 85.0

    mock_provider = MockLLMProvider(responses=[json.dumps(bad_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "Technical score mismatch" in result.error


# ---------------------------------------------------------------------------
# 3. Indicator Value Grounding Tests
# ---------------------------------------------------------------------------


def test_llm_cannot_alter_supplied_indicator_values(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Validation must reject when LLM alters an existing indicator value."""
    bad_output_dict = json.loads(json.dumps(valid_output_dict))
    bad_output_dict["indicators_summary"]["rsi"]["rsi_14"] = 80.0  # metrics has 62.5

    mock_provider = MockLLMProvider(responses=[json.dumps(bad_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "RSI-14 mismatch" in result.error


def test_llm_cannot_invent_missing_indicator(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Validation must reject if LLM invents a value for a null indicator."""
    metrics_with_null = sample_technical_metrics.model_copy(deep=True)
    metrics_with_null.moving_averages.sma_200 = None

    bad_output_dict = json.loads(json.dumps(valid_output_dict))
    bad_output_dict["indicators_summary"]["moving_averages"]["sma_200"] = 110.0

    mock_provider = MockLLMProvider(responses=[json.dumps(bad_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(metrics_with_null)

    assert result.success is False
    assert "Moving average 'sma_200' mismatch" in result.error


def test_missing_null_indicator_handled_correctly(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """When an indicator is null, LLM preserving null passes validation."""
    metrics_with_null = sample_technical_metrics.model_copy(deep=True)
    metrics_with_null.moving_averages.sma_200 = None

    ok_output_dict = json.loads(json.dumps(valid_output_dict))
    ok_output_dict["indicators_summary"]["moving_averages"]["sma_200"] = None

    mock_provider = MockLLMProvider(responses=[json.dumps(ok_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(metrics_with_null)

    assert result.success is True
    assert result.data.indicators_summary.moving_averages.sma_200 is None


def test_altered_score_breakdown_rejected(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Validation must reject when LLM alters score breakdown components."""
    bad_output_dict = json.loads(json.dumps(valid_output_dict))
    bad_output_dict["score_breakdown"]["rsi_score"] = 10.0  # metrics has 20.0

    mock_provider = MockLLMProvider(responses=[json.dumps(bad_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "Score breakdown 'rsi_score' mismatch" in result.error


def test_altered_support_resistance_levels_rejected(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Validation must reject when LLM alters support or resistance level lists."""
    # Test altered support level value
    bad_output_dict = json.loads(json.dumps(valid_output_dict))
    bad_output_dict["support_resistance"]["support_levels"][0] = 99.0

    mock_provider = MockLLMProvider(responses=[json.dumps(bad_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "Support level index 0 mismatch" in result.error

    # Test mismatched count
    bad_output_dict2 = json.loads(json.dumps(valid_output_dict))
    bad_output_dict2["support_resistance"]["resistance_levels"].append(200.0)

    mock_provider2 = MockLLMProvider(responses=[json.dumps(bad_output_dict2)])
    agent2 = TechnicalAnalystAgent(provider=mock_provider2)

    result2 = agent2.run(sample_technical_metrics)

    assert result2.success is False
    assert "Resistance levels count mismatch" in result2.error


# ---------------------------------------------------------------------------
# 4. Ticker & Safety / Prohibited Content Validation Tests
# ---------------------------------------------------------------------------


def test_ticker_mismatch_rejected(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Validation must reject if LLM returns a mismatched ticker."""
    bad_output_dict = dict(valid_output_dict)
    bad_output_dict["ticker"] = "MSFT"

    mock_provider = MockLLMProvider(responses=[json.dumps(bad_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "Ticker mismatch" in result.error


@pytest.mark.parametrize(
    "prohibited_text",
    [
        "We issue a Strong Buy recommendation based on the moving average stack.",
        "Investors should buy immediately before the breakout.",
        "FinPilot gives NVDA a buy rating at current levels.",
        "Analyst rating: buy based on MACD confirmation.",
        "We maintain a price target of $160 for NVDA over the next 12 months.",
        "The target price for this technical move is 145.0.",
        "Technical analysis models indicate a projected price of $150.",
        "Technical analysis indicates a guaranteed profit on this trade.",
        "This asset provides guaranteed returns given the setup.",
        "This setup offers a risk-free return given the support boundary.",
        "A certain return is anticipated from this momentum.",
    ],
)
def test_prohibited_content_rejected(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
    prohibited_text: str,
):
    """Validation must reject any prohibited recommendations or price targets."""
    bad_output_dict = json.loads(json.dumps(valid_output_dict))
    bad_output_dict["interpretation"]["overall_summary"] = prohibited_text

    mock_provider = MockLLMProvider(responses=[json.dumps(bad_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "Prohibited advice or recommendation" in result.error


def test_legitimate_technical_language_allowed(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Legitimate technical analysis descriptions must pass validation cleanly."""
    clean_output_dict = json.loads(json.dumps(valid_output_dict))
    clean_output_dict["interpretation"]["overall_summary"] = (
        "Price is above SMA20 and SMA50, testing resistance near the supplied level. "
        "Support is established at 115.0."
    )
    clean_output_dict["interpretation"]["support_resistance_analysis"] = (
        "Primary resistance is above current price at 130.0. "
        "Primary support is below current price at 115.0."
    )

    mock_provider = MockLLMProvider(responses=[json.dumps(clean_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is True
    assert "Price is above SMA20" in result.data.interpretation.overall_summary


# ---------------------------------------------------------------------------
# 5. Confidence & Sparse Data Handling Tests
# ---------------------------------------------------------------------------


def test_confidence_capped_on_sparse_data(
    valid_output_dict: Dict[str, Any],
):
    """Confidence must cap to <= 0.50 when historical data is sparse (<20 candles)."""
    sparse_metrics = TechnicalMetrics(
        ticker="SPARSE",
        latest_close=50.0,
        candle_count=10,  # sparse history
        moving_averages=MovingAverageMetrics(),
        rsi=RSIMetrics(),
        macd=MACDMetrics(),
        volume=VolumeMetrics(latest_volume=10_000.0),
        support_resistance=SupportResistanceMetrics(),
        trend=None,
        technical_score=None,
        score_breakdown=None,
    )

    sparse_output_dict = json.loads(json.dumps(valid_output_dict))
    sparse_output_dict["ticker"] = "SPARSE"
    sparse_output_dict["trend"] = None
    sparse_output_dict["technical_score"] = None
    sparse_output_dict["score_breakdown"] = None
    sparse_output_dict["indicators_summary"]["latest_close"] = 50.0
    sparse_output_dict["indicators_summary"][
        "moving_averages"
    ] = MovingAverageMetrics().model_dump()
    sparse_output_dict["indicators_summary"]["rsi"] = RSIMetrics().model_dump()
    sparse_output_dict["indicators_summary"]["macd"] = MACDMetrics().model_dump()
    sparse_output_dict["indicators_summary"]["volume"] = VolumeMetrics(
        latest_volume=10_000.0
    ).model_dump()
    sparse_output_dict["support_resistance"] = SupportResistanceSummary().model_dump()
    sparse_output_dict["confidence"] = 0.85  # Overconfident on sparse data

    mock_provider = MockLLMProvider(responses=[json.dumps(sparse_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sparse_metrics)

    assert result.success is True
    assert result.data.confidence == pytest.approx(0.50)  # Capped to 0.50


# ---------------------------------------------------------------------------
# 6. Error Isolation & Malformed LLM Output Tests
# ---------------------------------------------------------------------------


def test_malformed_llm_json_rejected(
    sample_technical_metrics: TechnicalMetrics,
):
    """Verify that unparseable non-JSON from LLM returns structured failure."""
    mock_provider = MockLLMProvider(responses=["NOT_VALID_JSON{broken"])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "structured validation failed" in result.error


def test_llm_provider_error_handled_gracefully(
    sample_technical_metrics: TechnicalMetrics,
):
    """Verify that LLMProvider exceptions are caught and reported as failure."""
    mock_provider = MockLLMProvider(
        fail_with=LLMError("API connection timeout", provider="mock")
    )
    agent = TechnicalAnalystAgent(provider=mock_provider)

    result = agent.run(sample_technical_metrics)

    assert result.success is False
    assert "LLM provider error" in result.error


# ---------------------------------------------------------------------------
# 7. LangGraph Node Adapter Tests
# ---------------------------------------------------------------------------


def test_technical_analyst_node_success(
    sample_technical_metrics: TechnicalMetrics,
    valid_output_dict: Dict[str, Any],
):
    """Verify technical_analyst_node updates GraphState correctly on success."""
    mock_provider = MockLLMProvider(responses=[json.dumps(valid_output_dict)])
    agent = TechnicalAnalystAgent(provider=mock_provider)

    state: GraphState = {
        "technical_metrics": sample_technical_metrics,
        "investor_profile": {"time_horizon": "1 year", "risk_tolerance": "moderate"},
        "cio_decision": {
            "specialist_tasks": {
                "technical": {"task_description": "Analyze key levels."}
            }
        },
    }

    node_update = technical_analyst_node(state, agent=agent)

    assert "technical_result" in node_update
    tr = node_update["technical_result"]
    assert tr["specialist"] == "technical"
    assert tr["status"] == "completed"
    assert tr["success"] is True
    assert tr["data"]["ticker"] == "NVDA"
    assert tr["confidence"] == 0.90


def test_technical_analyst_node_missing_metrics():
    """Verify node returns graceful failure when metrics are absent."""
    state: GraphState = {"investor_profile": {}}
    node_update = technical_analyst_node(state)

    assert "technical_result" in node_update
    tr = node_update["technical_result"]
    assert tr["status"] == "failed"
    assert tr["success"] is False
    assert "No technical metrics available" in tr["error"]
