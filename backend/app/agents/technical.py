"""Technical Analyst Agent implementation for FinPilot.

Phase 7.3 implements the specialist agent responsible for qualitative
interpretation of deterministic technical metrics computed in Phase 7.2.

Core Architectural Principles:
- The deterministic Phase 7.2 engine owns all technical calculations.
- The LLM is an interpretation layer only: it never calculates, modifies, or
  invents values.
- Factual numbers MUST originate from the supplied TechnicalMetrics.
- Missing values (None) must remain explicitly unknown.
- Prohibits buy/sell/hold recommendations, target prices, or return guarantees.
- Confidence strictly reflects interpretation quality and data completeness.
- Grounding validation rejects any altered metrics or safety violations.
- Exposes a LangGraph-compatible adapter without modifying main graph orchestration.
"""

import json
import math
from typing import Any, Dict, List, Optional, Union

from app.agents.base import AgentResult, BaseAgent
from app.agents.state import GraphState
from app.agents.technical_schema import (
    TechnicalAnalysisOutput,
    TechnicalAnalysisValidationError,
    TechnicalAnalystInput,
)
from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import (
    LLMError,
    LLMStructuredOutputError,
)
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger
from app.models.technical_metrics import TechnicalMetrics

logger = get_logger("app.agents.technical")

TECHNICAL_SYSTEM_PROMPT = (
    "You are FinPilot's Technical Analysis interpretation specialist.\n"
    "The supplied technical indicators were calculated deterministically by the\n"
    "FinPilot technical metrics engine.\n\n"
    "Your task is ONLY to interpret and explain the supplied values.\n\n"
    "CRITICAL BOUNDARIES:\n"
    "1. NEVER calculate an indicator yourself.\n"
    "2. NEVER invent or estimate a value that is not supplied.\n"
    "3. NEVER alter a supplied numerical value.\n"
    "4. NEVER create a price target.\n"
    "5. NEVER issue a buy/sell recommendation or rating.\n"
    "6. NEVER claim guaranteed returns or risk-free outcomes.\n\n"
    "GROUNDING & INTERPRETATION RULES:\n"
    "- Use only the supplied TechnicalMetrics payload.\n"
    "- If an indicator is unavailable (null), explicitly state that it is unavailable\n"
    "  and do not infer or fabricate it.\n"
    "- Explain:\n"
    "  * overall trend direction\n"
    "  * moving-average alignment and stacking\n"
    "  * RSI momentum regime\n"
    "  * MACD relationship and histogram dynamics\n"
    "  * volume confirmation\n"
    "  * support/resistance context\n"
    "  * deterministic technical score\n"
    "  * relevant technical risks\n"
    "- Your reasoning must remain consistent with the supplied numbers.\n"
    "- Confidence represents technical interpretation quality and data completeness, "
    "NOT an investment-return probability."
)

PROHIBITED_PHRASES = [
    "buy recommendation",
    "sell recommendation",
    "hold recommendation",
    "recommend buying",
    "recommend selling",
    "recommend to buy",
    "recommend to sell",
    "should buy",
    "should sell",
    "must buy",
    "must sell",
    "strong buy",
    "strong sell",
    "buy rating",
    "sell rating",
    "hold rating",
    "rating: buy",
    "rating: sell",
    "rating: hold",
    "price target",
    "target price",
    "target of $",
    "price goal",
    "projected price",
    "projected future price",
    "guaranteed return",
    "guaranteed returns",
    "guaranteed profit",
    "guaranteed profits",
    "risk-free return",
    "risk-free profit",
    "certain return",
    "certain profit",
]


def format_technical_prompt(input_data: TechnicalAnalystInput) -> str:
    """Construct grounded prompt for the Technical Analyst Agent.

    Args:
        input_data: Validated TechnicalAnalystInput containing TechnicalMetrics.

    Returns:
        str: Grounded LLM prompt containing deterministic data and investor context.
    """
    metrics = input_data.metrics
    metrics_dict = metrics.model_dump(mode="json")

    sections = [
        TECHNICAL_SYSTEM_PROMPT,
        "\n--- TARGET SECURITY ---",
        f"Ticker: {metrics.ticker}",
        f"Latest Close: {metrics.latest_close}",
        f"Evaluated Candles: {metrics.candle_count}",
        f"Calculated At: {metrics.calculated_at.isoformat()}",
    ]

    # Optional investor context (for emphasis / horizon alignment only)
    if input_data.time_horizon or input_data.risk_tolerance:
        sections.append("\n--- INVESTOR CONTEXT (FOR EMPHASIS ONLY) ---")
        if input_data.time_horizon:
            sections.append(f"Time Horizon: {input_data.time_horizon}")
        if input_data.risk_tolerance:
            sections.append(f"Risk Tolerance: {input_data.risk_tolerance}")

    if input_data.task_description:
        sections.append(f"Task Guidance: {input_data.task_description}")

    # Inject complete deterministic technical metrics
    sections.append("\n=== TECHNICAL METRICS PAYLOAD (DETERMINISTIC) ===")
    sections.append(json.dumps(metrics_dict, indent=2, default=str))
    sections.append("=== END METRICS PAYLOAD ===")
    sections.append(
        "\nProvide your structured technical assessment strictly interpreting "
        "the above metrics."
    )

    return "\n".join(sections)


def _matches_float(a: Optional[float], b: Optional[float], tol: float = 1e-4) -> bool:
    """Check if two optional float values match within relative tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def validate_technical_analysis_grounding(
    output: TechnicalAnalysisOutput,
    metrics: TechnicalMetrics,
) -> None:
    """Enforce deterministic consistency, grounding, and safety on LLM output.

    Validates:
    - Ticker identity matches supplied metrics.
    - Trend classification exactly matches Phase 7.2 deterministic trend.
    - Technical score exactly matches Phase 7.2 deterministic score.
    - All reported indicators match supplied values; no fabricated or altered numbers.
    - Unavailable/null indicators cannot be reported as present.
    - No prohibited buy/sell recommendations or price targets.
    - Confidence is bounded and appropriately reflects data sufficiency.

    Raises:
        TechnicalAnalysisValidationError: If any grounding or safety rule is violated.
    """
    # 1. Ticker validation
    if output.ticker.strip().upper() != metrics.ticker.strip().upper():
        raise TechnicalAnalysisValidationError(
            f"Ticker mismatch: expected '{metrics.ticker}', got '{output.ticker}'."
        )

    # 2. Trend validation
    if output.trend != metrics.trend:
        raise TechnicalAnalysisValidationError(
            f"Trend mismatch: expected '{metrics.trend}', got '{output.trend}'."
        )

    # 3. Technical score validation
    if not _matches_float(output.technical_score, metrics.technical_score):
        raise TechnicalAnalysisValidationError(
            f"Technical score mismatch: expected {metrics.technical_score}, "
            f"got {output.technical_score}."
        )
    # Bitwise exact preservation
    output.technical_score = metrics.technical_score

    # 4. Indicators summary grounding validation
    ind = output.indicators_summary
    if not _matches_float(ind.latest_close, metrics.latest_close):
        raise TechnicalAnalysisValidationError(
            f"Latest close mismatch: expected {metrics.latest_close}, "
            f"got {ind.latest_close}."
        )

    # Moving averages checks
    ma_pairs = [
        ("sma_20", ind.moving_averages.sma_20, metrics.moving_averages.sma_20),
        ("sma_50", ind.moving_averages.sma_50, metrics.moving_averages.sma_50),
        ("sma_200", ind.moving_averages.sma_200, metrics.moving_averages.sma_200),
        ("ema_20", ind.moving_averages.ema_20, metrics.moving_averages.ema_20),
        ("ema_50", ind.moving_averages.ema_50, metrics.moving_averages.ema_50),
        ("ema_200", ind.moving_averages.ema_200, metrics.moving_averages.ema_200),
        ("ema_12", ind.moving_averages.ema_12, metrics.moving_averages.ema_12),
        ("ema_26", ind.moving_averages.ema_26, metrics.moving_averages.ema_26),
    ]
    for name, out_val, exp_val in ma_pairs:
        if not _matches_float(out_val, exp_val):
            raise TechnicalAnalysisValidationError(
                f"Moving average '{name}' mismatch: expected {exp_val}, got {out_val}."
            )

    # RSI checks
    if not _matches_float(ind.rsi.rsi_14, metrics.rsi.rsi_14):
        raise TechnicalAnalysisValidationError(
            f"RSI-14 mismatch: expected {metrics.rsi.rsi_14}, got {ind.rsi.rsi_14}."
        )

    # MACD checks
    macd_pairs = [
        ("macd_line", ind.macd.macd_line, metrics.macd.macd_line),
        ("signal_line", ind.macd.signal_line, metrics.macd.signal_line),
        ("histogram", ind.macd.histogram, metrics.macd.histogram),
    ]
    for name, out_val, exp_val in macd_pairs:
        if not _matches_float(out_val, exp_val):
            raise TechnicalAnalysisValidationError(
                f"MACD '{name}' mismatch: expected {exp_val}, got {out_val}."
            )

    # Volume checks
    vol_pairs = [
        ("latest_volume", ind.volume.latest_volume, metrics.volume.latest_volume),
        (
            "average_volume_20d",
            ind.volume.average_volume_20d,
            metrics.volume.average_volume_20d,
        ),
        ("volume_ratio", ind.volume.volume_ratio, metrics.volume.volume_ratio),
    ]
    for name, out_val, exp_val in vol_pairs:
        if not _matches_float(out_val, exp_val):
            raise TechnicalAnalysisValidationError(
                f"Volume metric '{name}' mismatch: expected {exp_val}, got {out_val}."
            )

    # Support / Resistance checks
    sr = output.support_resistance
    if not _matches_float(
        sr.primary_support, metrics.support_resistance.primary_support
    ):
        raise TechnicalAnalysisValidationError(
            f"Primary support mismatch: expected "
            f"{metrics.support_resistance.primary_support}, got {sr.primary_support}."
        )
    if not _matches_float(
        sr.primary_resistance, metrics.support_resistance.primary_resistance
    ):
        raise TechnicalAnalysisValidationError(
            f"Primary resistance mismatch: expected "
            f"{metrics.support_resistance.primary_resistance}, "
            f"got {sr.primary_resistance}."
        )
    if len(sr.support_levels) != len(metrics.support_resistance.support_levels):
        raise TechnicalAnalysisValidationError(
            "Support levels count mismatch: expected "
            f"{len(metrics.support_resistance.support_levels)}, "
            f"got {len(sr.support_levels)}."
        )
    for i, (out_lvl, exp_lvl) in enumerate(
        zip(sr.support_levels, metrics.support_resistance.support_levels)
    ):
        if not _matches_float(out_lvl, exp_lvl):
            raise TechnicalAnalysisValidationError(
                f"Support level index {i} mismatch: expected {exp_lvl}, "
                f"got {out_lvl}."
            )
    if len(sr.resistance_levels) != len(metrics.support_resistance.resistance_levels):
        raise TechnicalAnalysisValidationError(
            "Resistance levels count mismatch: expected "
            f"{len(metrics.support_resistance.resistance_levels)}, "
            f"got {len(sr.resistance_levels)}."
        )
    for i, (out_lvl, exp_lvl) in enumerate(
        zip(sr.resistance_levels, metrics.support_resistance.resistance_levels)
    ):
        if not _matches_float(out_lvl, exp_lvl):
            raise TechnicalAnalysisValidationError(
                f"Resistance level index {i} mismatch: expected {exp_lvl}, "
                f"got {out_lvl}."
            )
    output.support_resistance.support_levels = list(
        metrics.support_resistance.support_levels
    )
    output.support_resistance.resistance_levels = list(
        metrics.support_resistance.resistance_levels
    )

    # Score breakdown checks
    if metrics.score_breakdown is None:
        if output.score_breakdown is not None:
            raise TechnicalAnalysisValidationError(
                "Score breakdown should be None when metrics.score_breakdown is None."
            )
    else:
        if output.score_breakdown is None:
            raise TechnicalAnalysisValidationError(
                "Score breakdown missing: expected deterministic score components."
            )
        breakdown_pairs = [
            (
                "ma_alignment_score",
                output.score_breakdown.ma_alignment_score,
                metrics.score_breakdown.ma_alignment_score,
            ),
            (
                "rsi_score",
                output.score_breakdown.rsi_score,
                metrics.score_breakdown.rsi_score,
            ),
            (
                "macd_score",
                output.score_breakdown.macd_score,
                metrics.score_breakdown.macd_score,
            ),
            (
                "volume_score",
                output.score_breakdown.volume_score,
                metrics.score_breakdown.volume_score,
            ),
        ]
        for name, out_val, exp_val in breakdown_pairs:
            if not _matches_float(out_val, exp_val):
                raise TechnicalAnalysisValidationError(
                    f"Score breakdown '{name}' mismatch: expected {exp_val}, "
                    f"got {out_val}."
                )
        output.score_breakdown = metrics.score_breakdown.model_copy()

    # 5. Prohibited phrases check across all narrative fields
    narratives_to_check: List[str] = (
        [
            output.interpretation.overall_summary,
            output.interpretation.trend_analysis,
            output.interpretation.moving_averages_analysis,
            output.interpretation.momentum_analysis,
            output.interpretation.volume_analysis,
            output.interpretation.support_resistance_analysis,
        ]
        + output.evidence
        + output.risks
    )

    for text in narratives_to_check:
        lower_text = text.lower()
        for phrase in PROHIBITED_PHRASES:
            if phrase in lower_text:
                raise TechnicalAnalysisValidationError(
                    f"Prohibited advice or recommendation phrase detected: '{phrase}'."
                )

    # 6. Confidence validation and sparse-data capping
    if metrics.candle_count < 20 or metrics.technical_score is None:
        if output.confidence > 0.50:
            logger.debug(
                "Capping confidence from %.2f to 0.50 due to insufficient "
                "technical data.",
                output.confidence,
            )
            output.confidence = 0.50


class TechnicalAnalystAgent(BaseAgent):
    """Specialist agent producing grounded technical market analysis."""

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        """Initialize Technical Analyst Agent with optional LLMProvider.

        Args:
            provider: Optional LLMProvider instance (defaults to factory setting).
        """
        super().__init__()
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        """Get or lazily initialize the LLMProvider."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    @property
    def name(self) -> str:
        """Unique identifier name of the agent."""
        return "technical_analyst"

    @property
    def input_schema(self) -> type:
        """Expected input schema."""
        return TechnicalAnalystInput

    @property
    def output_schema(self) -> type:
        """Structured output schema."""
        return TechnicalAnalysisOutput

    def run(
        self,
        input_data: Union[TechnicalAnalystInput, TechnicalMetrics, Dict[str, Any]],
    ) -> AgentResult:
        """Execute grounded technical analysis.

        Args:
            input_data: TechnicalAnalystInput, TechnicalMetrics, or dictionary.

        Returns:
            AgentResult: Successful result containing TechnicalAnalysisOutput,
            or failure result with descriptive error.
        """
        if isinstance(input_data, TechnicalAnalystInput):
            parsed_input = input_data
        elif isinstance(input_data, TechnicalMetrics):
            parsed_input = TechnicalAnalystInput(metrics=input_data)
        elif isinstance(input_data, dict):
            try:
                if "metrics" in input_data:
                    parsed_input = TechnicalAnalystInput.model_validate(input_data)
                else:
                    metrics = TechnicalMetrics.model_validate(input_data)
                    parsed_input = TechnicalAnalystInput(metrics=metrics)
            except Exception as err:
                logger.error("Failed to parse TechnicalAnalystInput: %s", err)
                return AgentResult.create_failure(
                    error=f"Invalid technical input payload: {err}"
                )
        else:
            return AgentResult.create_failure(
                error=(
                    "Input must be an instance of TechnicalAnalystInput, "
                    "TechnicalMetrics, or dict."
                )
            )

        prompt = format_technical_prompt(parsed_input)

        try:
            output: TechnicalAnalysisOutput = generate_structured(
                provider=self.provider,
                prompt=prompt,
                schema=TechnicalAnalysisOutput,
            )
            # Apply strict grounding and safety validation
            validate_technical_analysis_grounding(
                output=output,
                metrics=parsed_input.metrics,
            )
            return AgentResult.create_success(
                data=output,
                confidence=output.confidence,
            )
        except TechnicalAnalysisValidationError as err:
            logger.error("Grounding validation failed for TechnicalAnalyst: %s", err)
            return AgentResult.create_failure(
                error=f"Technical analysis grounding validation failed: {err}"
            )
        except LLMStructuredOutputError as err:
            logger.error(
                "Structured output generation failed for TechnicalAnalyst: %s",
                err,
            )
            return AgentResult.create_failure(
                error=f"Technical analysis structured validation failed: {err}"
            )
        except LLMError as err:
            logger.error("LLM Provider failed during technical analysis: %s", err)
            return AgentResult.create_failure(
                error=f"LLM provider error during technical analysis: {err}"
            )
        except Exception as err:
            logger.error(
                "Unexpected error during technical analysis execution: %s",
                err,
                exc_info=True,
            )
            return AgentResult.create_failure(
                error=f"Unexpected technical analysis error: {err}"
            )


def technical_analyst_node(
    state: GraphState,
    agent: Optional[TechnicalAnalystAgent] = None,
) -> Dict[str, Any]:
    """LangGraph-compatible node adapter for the Technical Analyst Agent.

    Extracts TechnicalMetrics and investor context from state, invokes
    the TechnicalAnalystAgent, and updates 'technical_result' in state.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured TechnicalAnalystAgent.

    Returns:
        Dict[str, Any]: State update mapping for 'technical_result'.
    """
    active_agent = agent or TechnicalAnalystAgent()
    metrics_data = (
        state.get("technical_metrics")
        or state.get("metrics")
        or state.get("technicals")
    )

    if not metrics_data:
        logger.warning("No technical metrics found in GraphState; returning failure.")
        failure = AgentResult.create_failure(
            error="No technical metrics available in GraphState for analysis."
        )
        return {
            "technical_result": {
                "specialist": "technical",
                "status": "failed",
                "success": False,
                "error": failure.error,
            }
        }

    profile = state.get("investor_profile") or {}
    cio_decision = state.get("cio_decision") or {}
    tasks = cio_decision.get("specialist_tasks") or {}
    technical_task = tasks.get("technical") or {}

    analyst_input = TechnicalAnalystInput(
        metrics=(
            metrics_data
            if isinstance(metrics_data, TechnicalMetrics)
            else TechnicalMetrics.model_validate(metrics_data)
        ),
        time_horizon=profile.get("time_horizon"),
        risk_tolerance=profile.get("risk_tolerance"),
        task_description=technical_task.get("task_description"),
    )

    result = active_agent.run(analyst_input)
    if result.success and isinstance(result.data, TechnicalAnalysisOutput):
        return {
            "technical_result": {
                "specialist": "technical",
                "status": "completed",
                "success": True,
                "data": result.data.model_dump(),
                "confidence": result.confidence,
            }
        }

    return {
        "technical_result": {
            "specialist": "technical",
            "status": "failed",
            "success": False,
            "error": result.error or "Unknown technical analysis failure.",
        }
    }
