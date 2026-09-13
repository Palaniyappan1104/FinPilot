"""Risk Analyst Agent implementation for FinPilot (Phase 10.3).

Specialist agent producing structured, objective, investor-aware risk assessments
synthesizing upstream signals from Technical, Fundamental, News, and Research Vault.
"""

from typing import Any, Dict, List, Optional, Union

from app.agents.base import AgentResult, BaseAgent
from app.agents.risk_prompt import format_risk_prompt
from app.agents.risk_schema import (
    PROHIBITED_ADVICE_PATTERNS,
    RiskAnalysisOutput,
    RiskAnalysisValidationError,
    RiskAnalystInput,
    RiskCategory,
    RiskFactor,
)
from app.agents.risk_scoring import (
    DeterministicRiskScore,
    calculate_deterministic_risk_score,
)
from app.agents.state import GraphState
from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import LLMError, LLMStructuredOutputError
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger

logger = get_logger("app.agents.risk")


# ===========================================================================
# STRUCTURED OUTPUT VALIDATION & RECONCILIATION (10.3.4)
# ===========================================================================


def _check_prohibited_language(text: str, field_name: str) -> None:
    """Check a text field against prohibited investment advice patterns."""
    for pattern in PROHIBITED_ADVICE_PATTERNS:
        if pattern.search(text):
            raise RiskAnalysisValidationError(
                f"Prohibited advisory language matching '{pattern.pattern}' "
                f"detected in {field_name}. FinPilot provides decision support only."
            )


def validate_risk_analysis(
    output: RiskAnalysisOutput,
    input_data: RiskAnalystInput,
    deterministic_assessment: Optional[DeterministicRiskScore] = None,
) -> RiskAnalysisOutput:
    """Validate structured risk output against grounding, citation, and safety rules.

    Args:
        output: Raw RiskAnalysisOutput produced by LLM or fallback.
        input_data: Original validated RiskAnalystInput.
        deterministic_assessment: Optional pre-computed deterministic score baseline.

    Returns:
        RiskAnalysisOutput: Fully validated and reconciled risk output.

    Raises:
        RiskAnalysisValidationError: If safety or grounding validation fails.
    """
    # 1. Safety check on narrative summary
    _check_prohibited_language(output.summary, "summary")

    # Map category lists to their designated RiskCategory enum
    category_map: Dict[RiskCategory, List[RiskFactor]] = {
        RiskCategory.MARKET: output.market_risks,
        RiskCategory.COMPANY: output.company_risks,
        RiskCategory.SECTOR: output.sector_risks,
        RiskCategory.FINANCIAL: output.financial_risks,
        RiskCategory.VOLATILITY: output.volatility_risks,
        RiskCategory.INVESTOR_SPECIFIC: output.investor_specific_risks,
    }

    allowed_sources = set(input_data.available_sources)
    if input_data.investor_profile:
        allowed_sources.add("investor_profile")

    # 2. Validate all factors across categories
    for expected_cat, factor_list in category_map.items():
        for factor in factor_list:
            # Check prohibited text in factor fields
            _check_prohibited_language(factor.name, f"factor name '{factor.name}'")
            _check_prohibited_language(
                factor.description, f"factor description '{factor.name}'"
            )

            # Reconcile category alignment
            if factor.category != expected_cat:
                factor.category = expected_cat

            # Evidence and grounding validation
            if not factor.insufficient_data:
                if not factor.evidence:
                    raise RiskAnalysisValidationError(
                        f"Risk factor '{factor.name}' in category {expected_cat.value} "
                        "is marked as factual but lacks supporting evidence citations."
                    )
                for ev in factor.evidence:
                    _check_prohibited_language(
                        ev.detail, f"evidence detail in '{factor.name}'"
                    )
                    if ev.source_type not in allowed_sources:
                        raise RiskAnalysisValidationError(
                            f"Risk factor '{factor.name}' cited source "
                            f"'{ev.source_type}', which was not provided in the input."
                        )
                    if not ev.reference_id or not ev.reference_id.strip():
                        raise RiskAnalysisValidationError(
                            f"Evidence citation in risk factor '{factor.name}' "
                            "has an empty reference_id."
                        )

    # 3. Deterministic score reconciliation (10.3.3)
    if (
        deterministic_assessment is not None
        and not deterministic_assessment.insufficient_data
        and deterministic_assessment.score is not None
    ):
        output.deterministic_risk_score = deterministic_assessment.score
        output.deterministic_risk_level = deterministic_assessment.level
        # If LLM omitted overall_risk_level, apply the deterministic baseline
        if output.overall_risk_level is None:
            output.overall_risk_level = deterministic_assessment.level
    else:
        output.deterministic_risk_score = None
        output.deterministic_risk_level = None

    # 4. Data completeness mapping
    output.data_completeness = {
        "technical": input_data.has_technical_signals,
        "fundamental": input_data.has_fundamental_signals,
        "news": input_data.has_news_signals,
        "research": input_data.has_research_signals,
        "investor_profile": input_data.has_investor_profile,
    }

    # 5. Insufficient data & confidence adjustments
    if input_data.is_empty:
        output.insufficient_data = True
        output.overall_risk_level = None
        output.confidence = 0.0
        if not output.insufficient_data_reason:
            output.insufficient_data_reason = (
                "No upstream specialist signals were available for analysis."
            )
    elif input_data.data_completeness_ratio < 0.50:
        output.confidence = round(min(output.confidence, 0.65), 2)

    return output


# ===========================================================================
# RISK ANALYST AGENT CLASS
# ===========================================================================


class RiskAnalystAgent(BaseAgent):
    """Specialist agent synthesizing multi-source signals into risk assessments."""

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        """Initialize Risk Analyst Agent with optional LLMProvider.

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
        return "risk_analyst"

    @property
    def input_schema(self) -> type:
        """Expected input schema."""
        return RiskAnalystInput

    @property
    def output_schema(self) -> type:
        """Structured output schema."""
        return RiskAnalysisOutput

    def run(
        self,
        input_data: Union[RiskAnalystInput, Dict[str, Any], GraphState],
    ) -> AgentResult:
        """Execute grounded risk assessment.

        Args:
            input_data: RiskAnalystInput, dictionary, or GraphState.

        Returns:
            AgentResult: Successful result containing RiskAnalysisOutput,
            or failure result with descriptive error.
        """
        # 1. Normalize input
        if isinstance(input_data, RiskAnalystInput):
            parsed_input = input_data
        elif isinstance(input_data, dict):
            try:
                if (
                    "technical_signals" in input_data
                    or "fundamental_signals" in input_data
                ):
                    parsed_input = RiskAnalystInput.model_validate(input_data)
                elif "technical_result" in input_data or "cio_decision" in input_data:
                    parsed_input = RiskAnalystInput.from_graph_state(input_data)
                elif "ticker" in input_data:
                    parsed_input = RiskAnalystInput.from_partial_results(
                        ticker=input_data["ticker"],
                        investor_profile=input_data.get("investor_profile"),
                        technical_output=input_data.get("technical_output"),
                        fundamental_output=input_data.get("fundamental_output"),
                        news_output=input_data.get("news_output"),
                        research_output=input_data.get("research_output"),
                        task_description=input_data.get("task_description"),
                    )
                else:
                    return AgentResult.create_failure(
                        error=(
                            "Unable to parse RiskAnalystInput: missing ticker "
                            "or signals."
                        )
                    )
            except Exception as err:
                logger.error("Failed to parse RiskAnalystInput: %s", err)
                return AgentResult.create_failure(
                    error=f"Invalid risk input payload: {err}"
                )
        else:
            return AgentResult.create_failure(
                error="Input must be an instance of RiskAnalystInput or dict."
            )

        # 2. Immediate deterministic handling for completely empty inputs
        if parsed_input.is_empty:
            empty_output = RiskAnalysisOutput(
                ticker=parsed_input.ticker,
                overall_risk_level=None,
                insufficient_data=True,
                insufficient_data_reason=(
                    "No upstream specialist signals were available for analysis."
                ),
                summary=(
                    f"Risk analysis could not proceed for {parsed_input.ticker} "
                    "due to missing upstream specialist signals."
                ),
                confidence=0.0,
            )
            empty_output = validate_risk_analysis(empty_output, parsed_input)
            return AgentResult.create_success(data=empty_output, confidence=0.0)

        # 3. Compute deterministic quantitative baseline (10.3.3)
        det_assessment = calculate_deterministic_risk_score(parsed_input)

        # 4. Format grounded prompt
        prompt = format_risk_prompt(
            parsed_input,
            deterministic_assessment=det_assessment,
        )

        # 5. Call LLM for structured analysis
        try:
            raw_output: RiskAnalysisOutput = generate_structured(
                provider=self.provider,
                prompt=prompt,
                schema=RiskAnalysisOutput,
            )

            # 6. Apply deterministic validation and reconciliation (10.3.4)
            validated_output = validate_risk_analysis(
                output=raw_output,
                input_data=parsed_input,
                deterministic_assessment=det_assessment,
            )

            return AgentResult.create_success(
                data=validated_output,
                confidence=validated_output.confidence,
            )
        except RiskAnalysisValidationError as err:
            logger.error("Validation failed for RiskAnalyst: %s", err)
            return AgentResult.create_failure(
                error=f"Risk analysis validation failed: {err}"
            )
        except LLMStructuredOutputError as err:
            logger.error("Structured output generation failed for RiskAnalyst: %s", err)
            return AgentResult.create_failure(
                error=f"Risk analysis structured validation failed: {err}"
            )
        except LLMError as err:
            logger.error("LLM Provider failed during risk analysis: %s", err)
            return AgentResult.create_failure(
                error=f"LLM provider error during risk analysis: {err}"
            )
        except Exception as err:
            logger.error(
                "Unexpected error during risk analysis execution: %s",
                err,
                exc_info=True,
            )
            return AgentResult.create_failure(
                error=f"Unexpected risk analysis error: {err}"
            )


# ===========================================================================
# LANGGRAPH NODE ADAPTER (10.3)
# ===========================================================================


def risk_analyst_node(
    state: GraphState,
    agent: Optional[RiskAnalystAgent] = None,
) -> Dict[str, Any]:
    """LangGraph-compatible node adapter for the Risk Analyst Agent.

    Extracts specialist results from state, invokes the RiskAnalystAgent,
    and updates 'risk_result' in GraphState.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured RiskAnalystAgent.

    Returns:
        Dict[str, Any]: State update mapping for 'risk_result'.
    """
    active_agent = agent or RiskAnalystAgent()
    try:
        input_data = RiskAnalystInput.from_graph_state(state)
    except Exception as err:
        logger.error("Failed to construct RiskAnalystInput from GraphState: %s", err)
        fail_result = AgentResult.create_failure(
            error=f"Failed to assemble risk inputs: {err}"
        )
        return fail_result.to_state_update("risk_result")

    result = active_agent.run(input_data)
    if result.success and hasattr(result.data, "model_dump"):
        return {"risk_result": result.data.model_dump()}
    return result.to_state_update("risk_result")
