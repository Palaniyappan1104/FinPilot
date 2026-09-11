"""Chief Investment Officer (CIO) / Router Agent implementation for FinPilot.

Phase 5.2 implements the CIO Agent responsible for analyzing the clarified request
and investor profile, determining which specialist research agents are relevant,
and generating a structured routing decision with focused task parameters.
"""

from typing import Any, Dict, List, Optional, Union

from app.agents.base import AgentResult, BaseAgent
from app.agents.cio_schema import (
    CORE_SPECIALISTS,
    CIOInput,
    CIORoutingDecision,
    SpecialistName,
    SpecialistTask,
)
from app.agents.state import GraphState
from app.agents.tools import Tool
from app.core.llm.base import LLMProvider
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger

logger = get_logger("app.agents.cio")

CIO_SYSTEM_PROMPT = (
    "You are FinPilot's Chief Investment Officer (CIO) and Central Router Agent.\n"
    "Your responsibility is to analyze the clarified user request and "
    "investor profile,\n"
    "and determine which specialist research agents should be invoked to "
    "answer the query.\n\n"
    "Available Specialist Agents:\n"
    "1. technical: Analyzes price action, charts, indicators, and trends.\n"
    "2. fundamental: Analyzes financial statements, valuation, and balance sheet.\n"
    "3. news: Analyzes market news, media sentiment, and catalysts.\n"
    "4. research: Analyzes user-uploaded documents (10-K, 10-Q, PDFs).\n"
    "5. risk: Evaluates risk metrics, volatility, downside risk.\n\n"
    "Routing Rules:\n"
    "1. FULL INVESTMENT ANALYSIS: Route to 'technical', 'fundamental', "
    "'news', and 'risk'. Include 'research' ONLY IF documents are available.\n"
    "2. TECHNICAL / MOMENTUM QUERY: Route to 'technical' and 'risk'.\n"
    "3. FUNDAMENTAL / VALUATION QUERY: Route to 'fundamental' and 'risk'.\n"
    "4. NEWS / SENTIMENT QUERY: Route to 'news' and 'risk'.\n"
    "5. RESEARCH / DOCUMENT QUERY: Route to 'research' ONLY IF documents are "
    "available. If no documents are available, DO NOT route to 'research'.\n"
    "6. DOCUMENT AVAILABILITY: You MUST NOT select 'research' when "
    "documents_available is False.\n"
    "7. UNCERTAIN INTENT: If the request is ambiguous or broad, route to core: "
    "['technical', 'fundamental', 'news', 'risk'].\n"
    "8. For each selected specialist, provide focused task instructions.\n"
    "9. DO NOT calculate financial metrics or invent prices. You are a ROUTER."
)


def format_cio_prompt(
    normalized_query: str,
    intent_type: str,
    entities: Dict[str, Any],
    investor_profile: Dict[str, Any],
    documents_available: bool,
) -> str:
    """Construct prompt for the CIO / Router Agent.

    Args:
        normalized_query: Cleaned natural language user request.
        intent_type: Classified query intent.
        entities: Extracted request entities.
        investor_profile: Investor preferences and constraints.
        documents_available: Whether documents are attached.

    Returns:
        str: Formatted LLM prompt.
    """
    company = (
        entities.get("company")
        or investor_profile.get("target_company")
        or "Unknown Company"
    )
    ticker = entities.get("ticker") or investor_profile.get("ticker")
    time_horizon = entities.get("time_horizon") or investor_profile.get("time_horizon")
    capital_amount = entities.get("capital_amount") or investor_profile.get(
        "capital_amount"
    )
    risk_tolerance = entities.get("risk_tolerance") or investor_profile.get(
        "risk_tolerance"
    )

    cap_str = str(capital_amount) if capital_amount is not None else "Not specified"
    sections = [
        CIO_SYSTEM_PROMPT,
        "--- User Request Details ---",
        f'User Query: "{normalized_query}"',
        f"Intent Type: {intent_type}",
        f"Target Company: {company}",
        f"Ticker: {ticker or 'Not specified'}",
        f"Capital Amount: {cap_str}",
        f"Time Horizon: {time_horizon or 'Not specified'}",
        f"Risk Tolerance: {risk_tolerance or 'Not specified'}",
        f"Documents Available: {documents_available}",
        "----------------------------",
        "Generate a structured CIORoutingDecision selecting relevant specialists.",
    ]

    return "\n\n".join(sections)


def create_fallback_routing_decision(
    target_company: str,
    ticker: Optional[str] = None,
    documents_available: bool = False,
    time_horizon: Optional[str] = None,
    capital_amount: Optional[float] = None,
    risk_tolerance: Optional[str] = None,
    reasoning: str = "Fallback applied: routed to core specialists.",
) -> CIORoutingDecision:
    """Construct a deterministic fallback routing decision.

    Invoked when LLM generation fails, authentication fails, or routing is invalid.
    Routes to all 4 core specialists, plus research only if documents are available.

    Args:
        target_company: Target company name.
        ticker: Optional ticker symbol.
        documents_available: Whether documents are attached.
        time_horizon: Optional investment duration.
        capital_amount: Optional capital allocation.
        risk_tolerance: Optional risk appetite.
        reasoning: Rationale describing the fallback trigger.

    Returns:
        CIORoutingDecision: Validated fallback routing decision.
    """
    selected: List[SpecialistName] = list(CORE_SPECIALISTS)
    if documents_available:
        selected.append(SpecialistName.RESEARCH)

    base_params: Dict[str, Any] = {"company": target_company}
    if ticker:
        base_params["ticker"] = ticker
    if capital_amount is not None:
        base_params["capital_amount"] = capital_amount
    if time_horizon:
        base_params["time_horizon"] = time_horizon
    if risk_tolerance:
        base_params["risk_tolerance"] = risk_tolerance

    risk_label = risk_tolerance or "investor"
    tasks: Dict[SpecialistName, SpecialistTask] = {
        SpecialistName.TECHNICAL: SpecialistTask(
            specialist=SpecialistName.TECHNICAL,
            task_description=(
                f"Analyze technical indicators and trends for {target_company}."
            ),
            parameters=dict(base_params),
        ),
        SpecialistName.FUNDAMENTAL: SpecialistTask(
            specialist=SpecialistName.FUNDAMENTAL,
            task_description=(
                f"Analyze financial statements and valuation for {target_company}."
            ),
            parameters=dict(base_params),
        ),
        SpecialistName.NEWS: SpecialistTask(
            specialist=SpecialistName.NEWS,
            task_description=(
                f"Analyze news sentiment and catalysts for {target_company}."
            ),
            parameters=dict(base_params),
        ),
        SpecialistName.RISK: SpecialistTask(
            specialist=SpecialistName.RISK,
            task_description=f"Evaluate risk profile for {risk_label} profile.",
            parameters=dict(base_params),
        ),
    }

    if documents_available:
        tasks[SpecialistName.RESEARCH] = SpecialistTask(
            specialist=SpecialistName.RESEARCH,
            task_description=f"Review financial filings for {target_company}.",
            parameters=dict(base_params),
        )

    return CIORoutingDecision(
        target_company=target_company,
        ticker=ticker,
        selected_specialists=selected,
        specialist_tasks=tasks,
        reasoning=reasoning,
        fallback_applied=True,
    )


class CIOAgent(BaseAgent):
    """Chief Investment Officer (CIO) / Router Agent for FinPilot.

    Coordinates research workflows by determining which specialist agents
    should be invoked and generating targeted task parameters for each.
    """

    name = "cio_router"
    input_schema = CIOInput
    output_schema = CIORoutingDecision

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        tools: Optional[list[Tool]] = None,
    ) -> None:
        """Initialize the CIOAgent.

        Args:
            provider: Optional LLMProvider instance. Lazily fetched if None.
            tools: Optional sequence of tools available to this agent.
        """
        super().__init__(tools=tools)
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        """Return the configured LLMProvider, resolving default if unassigned."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def run(
        self,
        input_data: Union[CIOInput, GraphState, Dict[str, Any]],
    ) -> AgentResult:
        """Process investment request and produce a structured CIORoutingDecision.

        Args:
            input_data: CIOInput, GraphState, or dictionary with clarified_request
                and investor_profile.

        Returns:
            AgentResult: Successful result containing CIORoutingDecision data,
            or structured fallback decision if LLM generation/parsing fails.
        """
        # 1. Normalize input into clarified_request, investor_profile,
        # and documents_available
        if isinstance(input_data, CIOInput):
            clarified_req = input_data.clarified_request or {}
            profile = input_data.investor_profile or {}
            docs_available = bool(input_data.documents_available)
        elif isinstance(input_data, dict):
            clarified_req = input_data.get("clarified_request") or {}
            profile = input_data.get("investor_profile") or {}
            docs_available = bool(input_data.get("documents_available", False))
        else:
            return AgentResult.create_failure(
                error="Input must be an instance of CIOInput or a state dictionary.",
            )

        entities = clarified_req.get("entities") or {}
        normalized_query = clarified_req.get(
            "normalized_query",
            input_data.get("user_query", "") if isinstance(input_data, dict) else "",
        )
        intent_type = clarified_req.get("intent_type", "investment_analysis")

        target_company = (
            entities.get("company")
            or profile.get("target_company")
            or "Unknown Company"
        )
        ticker = entities.get("ticker") or profile.get("ticker")
        time_horizon = entities.get("time_horizon") or profile.get("time_horizon")
        capital_amount = entities.get("capital_amount") or profile.get("capital_amount")
        risk_tolerance = entities.get("risk_tolerance") or profile.get("risk_tolerance")

        # 2. Check for explicit upstream errors
        if clarified_req.get("intent_type") == "error":
            logger.warning(
                "Upstream request error detected; applying fallback routing."
            )
            fallback = create_fallback_routing_decision(
                target_company=target_company,
                ticker=ticker,
                documents_available=docs_available,
                time_horizon=time_horizon,
                capital_amount=capital_amount,
                risk_tolerance=risk_tolerance,
                reasoning="Fallback applied: upstream conversation error.",
            )
            return AgentResult.create_success(data=fallback)

        # 3. Build prompt and invoke LLM with structured output
        prompt = format_cio_prompt(
            normalized_query=normalized_query,
            intent_type=intent_type,
            entities=entities,
            investor_profile=profile,
            documents_available=docs_available,
        )

        try:
            decision: CIORoutingDecision = generate_structured(
                provider=self.provider,
                prompt=prompt,
                schema=CIORoutingDecision,
            )
        except Exception as exc:
            logger.error(
                "CIO LLM structured generation failed: %s; invoking fallback.", exc
            )
            fallback = create_fallback_routing_decision(
                target_company=target_company,
                ticker=ticker,
                documents_available=docs_available,
                time_horizon=time_horizon,
                capital_amount=capital_amount,
                risk_tolerance=risk_tolerance,
                reasoning=f"Fallback applied due to provider/parsing failure: {exc}",
            )
            return AgentResult.create_success(data=fallback)

        # 4. Post-processing and strict rule enforcement
        selected = list(decision.selected_specialists)
        tasks = dict(decision.specialist_tasks)

        # Rule: Research Analyst cannot be selected when documents_available is False
        if not docs_available and SpecialistName.RESEARCH in selected:
            logger.info("Removing Research Analyst because documents_available=False.")
            selected = [s for s in selected if s != SpecialistName.RESEARCH]
            tasks.pop(SpecialistName.RESEARCH, None)
            decision.reasoning += (
                " (Research Analyst omitted as no documents are available.)"
            )

        # Rule: If selected specialists became empty, fallback to core
        if not selected:
            logger.warning("Selected specialists list empty; invoking fallback.")
            fallback = create_fallback_routing_decision(
                target_company=target_company,
                ticker=ticker,
                documents_available=docs_available,
                time_horizon=time_horizon,
                capital_amount=capital_amount,
                risk_tolerance=risk_tolerance,
                reasoning="Fallback applied: empty specialist selection.",
            )
            return AgentResult.create_success(data=fallback)

        # Ensure parameters in specialist_tasks carry forward relevant state
        for spec, task in tasks.items():
            if not task.parameters.get("company"):
                task.parameters["company"] = decision.target_company
            if decision.ticker and not task.parameters.get("ticker"):
                task.parameters["ticker"] = decision.ticker
            if time_horizon and not task.parameters.get("time_horizon"):
                task.parameters["time_horizon"] = time_horizon
            if capital_amount is not None and not task.parameters.get("capital_amount"):
                task.parameters["capital_amount"] = capital_amount
            if risk_tolerance and not task.parameters.get("risk_tolerance"):
                task.parameters["risk_tolerance"] = risk_tolerance

        decision.selected_specialists = selected
        decision.specialist_tasks = tasks

        return AgentResult.create_success(data=decision)
