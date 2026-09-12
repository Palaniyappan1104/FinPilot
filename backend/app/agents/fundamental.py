"""Fundamental Analyst Agent implementation for FinPilot.

Phase 6.3 implements the specialist agent responsible for qualitative
interpretation of deterministic fundamental metrics computed in Phase 6.2.

Core Architectural Principles:
- The deterministic Phase 6.2 engine owns financial calculations.
- The LLM is an interpretation layer only: it never calculates, modifies, or
  invents values.
- Factual numbers MUST originate from the supplied FundamentalMetrics.
- Missing values (None) must remain explicitly unknown.
- Zero must remain zero; negative values are interpreted as losses/burdens.
- Investor profile context guides materiality emphasis without modifying metrics.
- Exposes a LangGraph-compatible adapter without modifying main graph orchestration.
"""

import json
from typing import Any, Dict, Optional, Union

from app.agents.base import AgentResult, BaseAgent
from app.agents.fundamental_schema import (
    VALID_DETERMINISTIC_METRIC_NAMES,
    FundamentalAnalysisOutput,
    FundamentalAnalystInput,
)
from app.agents.state import GraphState
from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import (
    LLMError,
    LLMStructuredOutputError,
)
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger
from app.models.fundamental_metrics import FundamentalMetrics

logger = get_logger("app.agents.fundamental")

FUNDAMENTAL_SYSTEM_PROMPT = (
    "You are FinPilot's Fundamental Analyst Agent.\n"
    "Your responsibility is to analyze and interpret pre-calculated "
    "fundamental financial\n"
    "metrics provided for a company.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. DATA INTEGRITY: All financial values, ratios, percentages, margins, "
    "and growth\n"
    "   rates you cite MUST come directly from the provided metrics payload. "
    "DO NOT calculate,\n"
    "   estimate, modify, or invent numbers.\n"
    "2. MISSING VALUES: If a metric is null or missing, you MUST treat it as "
    "unknown/unavailable.\n"
    "   Do not assume missing values are zero. Explicitly state that data is "
    "insufficient when\n"
    "   evaluating that dimension.\n"
    "3. ZERO-CROSSING SEMANTICS:\n"
    "   - If earnings_turnaround is True, explicitly state the company turned "
    "profitable from prior losses.\n"
    "   - If earnings_deficit_turnaround is True, explicitly state the company "
    "slipped into a net loss.\n"
    "   - If both_periods_deficit is True, note that the company posted "
    "consecutive net losses.\n"
    "   - Apply the same interpretations to EPS turnaround flags.\n"
    "   Do not report percentage changes as standard organic growth if a "
    "deficit is involved.\n"
    "4. VALUATION PROVENANCE:\n"
    "   - Note whether P/E and P/B multiples are provider-reported or derived "
    "fallbacks.\n"
    "5. NEGATIVE EQUITY:\n"
    "   - If negative_equity is True, highlight that stockholders' equity is in "
    "deficit,\n"
    "     which distorts standard leverage and return ratios.\n"
    "6. EVIDENCE TRACEABILITY:\n"
    "   - In supporting_metrics for each dimension, list the metric names from "
    "the payload\n"
    "     that substantiate your rating and explanation.\n"
    "7. CATEGORICAL ASSESSMENT:\n"
    "   - Assign overall_assessment as 'favorable', 'neutral', 'unfavorable', "
    "or 'insufficient_data'.\n"
    "   - Assign dimension ratings as 'strong', 'moderate', 'weak', 'neutral', "
    "or 'insufficient_data'.\n"
    "8. NO RECOMMENDATIONS:\n"
    "   - Do NOT provide buy/sell/hold ratings, target prices, or portfolio "
    "allocation advice.\n"
    "     Your role is strictly objective fundamental analysis."
)


def format_fundamental_prompt(input_data: FundamentalAnalystInput) -> str:
    """Construct grounded prompt for the Fundamental Analyst Agent.

    Args:
        input_data: Validated FundamentalAnalystInput.

    Returns:
        str: Grounded LLM prompt containing deterministic data and investor context.
    """
    metrics = input_data.metrics
    metrics_dict = metrics.model_dump(mode="json")

    sections = [
        FUNDAMENTAL_SYSTEM_PROMPT,
        "\n--- TARGET SECURITY ---",
        f"Ticker: {metrics.ticker}",
        f"Company Name: {metrics.company_name or 'Not specified'}",
        f"Reporting Currency: {metrics.currency or 'Not specified'}",
        f"Fiscal Period: {metrics.fiscal_year or 'Unknown'} "
        f"(Ending {metrics.period_end_date or 'Unknown'})",
    ]

    # Optional investor profile context (materiality emphasis only)
    if input_data.time_horizon or input_data.risk_tolerance:
        sections.append("\n--- INVESTOR CONTEXT (FOR EMPHASIS ONLY) ---")
        if input_data.time_horizon:
            sections.append(f"Time Horizon: {input_data.time_horizon}")
        if input_data.risk_tolerance:
            sections.append(f"Risk Tolerance: {input_data.risk_tolerance}")

    if input_data.task_description:
        sections.append(f"Task Guidance: {input_data.task_description}")

    # Inject complete deterministic metrics
    sections.append("\n=== COMPANY FUNDAMENTAL METRICS (DETERMINISTIC) ===")
    sections.append(json.dumps(metrics_dict, indent=2, default=str))
    sections.append("=== END METRICS PAYLOAD ===")
    sections.append(
        "\nProvide your structured fundamental assessment strictly interpreting "
        "the above metrics."
    )

    return "\n".join(sections)


def validate_fundamental_analysis_consistency(
    output: FundamentalAnalysisOutput,
    metrics: FundamentalMetrics,
) -> None:
    """Enforce deterministic consistency and safety boundaries on LLM output.

    Guards against:
    - Hallucinated or non-existent metric names in supporting_metrics
    - Inconsistent financial health rating when negative equity is present
    - Rating profitability or growth as strong when both periods are in deficit
    - Overconfidence when historical periods are sparse (IPO / 1-year data)
    """
    # 1. Clean hallucinated / unknown supporting metric names
    dimensions = [
        output.financial_health,
        output.growth_assessment,
        output.profitability_assessment,
        output.valuation_assessment,
        output.leverage_assessment,
        output.cash_flow_assessment,
    ]
    for dim in dimensions:
        cleaned_metrics = []
        for metric_name in dim.supporting_metrics:
            if metric_name in VALID_DETERMINISTIC_METRIC_NAMES:
                cleaned_metrics.append(metric_name)
            else:
                logger.debug(
                    "Unrecognized supporting metric name '%s' in %s assessment",
                    metric_name,
                    dim,
                )
        dim.supporting_metrics = cleaned_metrics

    # 2. Negative equity guard: cannot rate financial health as "strong"
    if metrics.profitability.negative_equity or metrics.leverage.negative_equity:
        if output.financial_health.rating == "strong":
            logger.warning(
                "Correcting financial_health from 'strong' to 'weak' (negative equity)."
            )
            output.financial_health.rating = "weak"

    # 3. Dual deficit guard: cannot rate growth or profitability as "strong"
    if metrics.growth.both_periods_deficit:
        if output.profitability_assessment.rating == "strong":
            logger.warning(
                "Correcting profitability from 'strong' to 'weak' (dual deficit)."
            )
            output.profitability_assessment.rating = "weak"
        if output.growth_assessment.rating == "strong":
            logger.warning("Correcting growth from 'strong' to 'weak' (dual deficit).")
            output.growth_assessment.rating = "weak"

    # 4. Sparse data confidence cap
    if not metrics.revenue_history or len(metrics.revenue_history) <= 1:
        if output.confidence > 0.70:
            logger.debug(
                "Capping confidence from %.2f to 0.70 due to sparse history.",
                output.confidence,
            )
            output.confidence = 0.70


class FundamentalAnalystAgent(BaseAgent):
    """Specialist agent producing grounded fundamental financial analysis."""

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        """Initialize Fundamental Analyst Agent with optional LLMProvider.

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
        return "fundamental_analyst"

    @property
    def input_schema(self) -> type:
        """Expected input schema."""
        return FundamentalAnalystInput

    @property
    def output_schema(self) -> type:
        """Structured output schema."""
        return FundamentalAnalysisOutput

    def run(
        self,
        input_data: Union[FundamentalAnalystInput, FundamentalMetrics, Dict[str, Any]],
    ) -> AgentResult:
        """Execute grounded fundamental analysis.

        Args:
            input_data: FundamentalAnalystInput, FundamentalMetrics, or dictionary.

        Returns:
            AgentResult: Successful result containing FundamentalAnalysisOutput,
            or failure result with descriptive error.
        """
        # Normalize input
        if isinstance(input_data, FundamentalAnalystInput):
            parsed_input = input_data
        elif isinstance(input_data, FundamentalMetrics):
            parsed_input = FundamentalAnalystInput(metrics=input_data)
        elif isinstance(input_data, dict):
            try:
                if "metrics" in input_data:
                    parsed_input = FundamentalAnalystInput.model_validate(input_data)
                else:
                    # Assume dict is raw FundamentalMetrics
                    metrics = FundamentalMetrics.model_validate(input_data)
                    parsed_input = FundamentalAnalystInput(metrics=metrics)
            except Exception as err:
                logger.error("Failed to parse FundamentalAnalystInput: %s", err)
                return AgentResult.create_failure(
                    error=f"Invalid fundamental input payload: {err}"
                )
        else:
            return AgentResult.create_failure(
                error=(
                    "Input must be an instance of FundamentalAnalystInput, "
                    "FundamentalMetrics, or dict."
                )
            )

        prompt = format_fundamental_prompt(parsed_input)

        try:
            output: FundamentalAnalysisOutput = generate_structured(
                provider=self.provider,
                prompt=prompt,
                schema=FundamentalAnalysisOutput,
            )
            # Apply consistency and grounding sanity checks
            validate_fundamental_analysis_consistency(
                output=output,
                metrics=parsed_input.metrics,
            )
            return AgentResult.create_success(
                data=output,
                confidence=output.confidence,
            )
        except LLMStructuredOutputError as err:
            logger.error(
                "Structured output generation failed for FundamentalAnalyst: %s",
                err,
            )
            return AgentResult.create_failure(
                error=f"Fundamental analysis structured validation failed: {err}"
            )
        except LLMError as err:
            logger.error("LLM Provider failed during fundamental analysis: %s", err)
            return AgentResult.create_failure(
                error=f"LLM provider error during fundamental analysis: {err}"
            )
        except Exception as err:
            logger.error(
                "Unexpected error during fundamental analysis execution: %s",
                err,
                exc_info=True,
            )
            return AgentResult.create_failure(
                error=f"Unexpected fundamental analysis error: {err}"
            )


def fundamental_analyst_node(
    state: GraphState,
    agent: Optional[FundamentalAnalystAgent] = None,
) -> Dict[str, Any]:
    """LangGraph-compatible node adapter for the Fundamental Analyst Agent.

    Extracts FundamentalMetrics and investor context from state, invokes
    the FundamentalAnalystAgent, and updates 'fundamental_result' in state.

    NOTE: In Phase 6.3, this node is tested in isolation. It does NOT modify
    the existing main orchestration graph topology or replace specialist stubs.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured FundamentalAnalystAgent.

    Returns:
        Dict[str, Any]: State update mapping for 'fundamental_result'.
    """
    active_agent = agent or FundamentalAnalystAgent()
    metrics_data = (
        state.get("fundamental_metrics")
        or state.get("metrics")
        or state.get("fundamentals")
    )

    if not metrics_data:
        logger.warning("No fundamental metrics found in GraphState; returning failure.")
        failure = AgentResult.create_failure(
            error="No fundamental metrics available in GraphState for analysis."
        )
        return {
            "fundamental_result": {
                "specialist": "fundamental",
                "status": "failed",
                "success": False,
                "error": failure.error,
            }
        }

    profile = state.get("investor_profile") or {}
    cio_decision = state.get("cio_decision") or {}
    tasks = cio_decision.get("specialist_tasks") or {}
    fundamental_task = tasks.get("fundamental") or {}

    analyst_input = FundamentalAnalystInput(
        metrics=(
            metrics_data
            if isinstance(metrics_data, FundamentalMetrics)
            else FundamentalMetrics.model_validate(metrics_data)
        ),
        time_horizon=profile.get("time_horizon"),
        risk_tolerance=profile.get("risk_tolerance"),
        task_description=fundamental_task.get("task_description"),
    )

    result = active_agent.run(analyst_input)
    if result.success and isinstance(result.data, FundamentalAnalysisOutput):
        return {
            "fundamental_result": {
                "specialist": "fundamental",
                "status": "completed",
                "success": True,
                "data": result.data.model_dump(),
                "confidence": result.confidence,
            }
        }

    return {
        "fundamental_result": {
            "specialist": "fundamental",
            "status": "failed",
            "success": False,
            "error": result.error or "Unknown fundamental analysis failure.",
        }
    }
