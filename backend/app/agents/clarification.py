"""Clarification Agent implementation for FinPilot.

Phase 3.3 implements the Clarification Agent responsible for determining whether
required investor information is complete for a requested analysis.

Workflow:
    ConversationOutput / clarified request
            ↓
    Clarification Agent
            ↓
    Determine whether required investor information is complete (deterministic)
            ↓
    If incomplete → generate targeted clarification questions (LLM or deterministic)
            ↓
    If complete → indicate clarification is not needed
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from app.agents.base import AgentResult, BaseAgent
from app.agents.clarification_schema import (
    ClarificationInput,
    ClarificationOutput,
    ClarificationQuestionsModel,
)
from app.agents.conversation_schema import ConversationOutput
from app.agents.tools import Tool
from app.core.llm.base import LLMProvider
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger

logger = get_logger("app.agents.clarification")

INVESTMENT_INTENTS = {
    "investment_analysis",
    "investment_decision",
    "investment",
    "portfolio_review",
    "portfolio_allocation",
    "buy_recommendation",
    "stock_recommendation",
    "allocation",
    "advisory",
}

INVESTMENT_KEYWORDS = [
    "should i invest",
    "can i invest",
    "want to invest",
    "planning to invest",
    "plan to invest",
    "invest in",
    "buy shares of",
    "buying shares",
    "allocate capital",
    "investment advice",
]

DEFAULT_PARAM_QUESTIONS: Dict[str, str] = {
    "company": "Which company or stock ticker would you like to analyze?",
    "capital_amount": "How much capital are you planning to invest?",
    "time_horizon": (
        "What is your intended investment duration (e.g. 1 year, 5 years)?"
    ),
    "risk_tolerance": (
        "What is your risk tolerance (conservative, moderate, or aggressive)?"
    ),
}

CLARIFICATION_SYSTEM_PROMPT = (
    "You are FinPilot's clarification specialist.\n"
    "Your responsibility is to generate concise, natural, user-facing questions "
    "to collect missing parameters needed for investment analysis.\n\n"
    "Rules:\n"
    "1. Generate questions ONLY for the missing parameters specified.\n"
    "2. DO NOT ask for information that has already been provided.\n"
    "3. DO NOT answer the user's investment query.\n"
    "4. DO NOT provide recommendations, analysis, or financial advice.\n"
    "5. DO NOT invent or assume default values for missing parameters.\n"
    "6. Avoid duplicate or redundant questions.\n"
    "7. Keep each question clear, courteous, and targeted."
)


def is_investment_intent(intent_type: str, query: str = "") -> bool:
    """Determine whether an intent or query represents an investment decision.

    Args:
        intent_type: Classified intent type string.
        query: Normalized or raw user query string.

    Returns:
        bool: True if the request requires full investor parameters.
    """
    clean_intent = intent_type.strip().lower()
    if clean_intent in INVESTMENT_INTENTS:
        return True

    clean_query = query.strip().lower()
    return any(keyword in clean_query for keyword in INVESTMENT_KEYWORDS)


def deduplicate_questions(questions: List[str]) -> List[str]:
    """Remove duplicate or empty questions while preserving order.

    Args:
        questions: List of raw question strings.

    Returns:
        List[str]: Cleaned list of unique questions.
    """
    seen = set()
    unique: List[str] = []
    for q in questions:
        if not isinstance(q, str):
            continue
        cleaned = q.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique.append(cleaned)
    return unique


def determine_missing_parameters(
    conversation_output: ConversationOutput,
    existing_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Deterministically check which required parameters are missing.

    Merges current ConversationOutput with any existing context/profile from
    previous conversation turns.

    Args:
        conversation_output: Current turn's ConversationOutput.
        existing_profile: Optional profile/parameters already captured in session.

    Returns:
        Tuple[List[str], Dict[str, Any]]:
            - missing_parameters: List of missing parameter names.
            - resolved_entities: Dictionary of merged parameter values.
    """
    profile = existing_profile or {}

    # Merge context: current turn takes precedence; fallback to existing profile
    company = conversation_output.company
    if not company:
        company = profile.get("company") or profile.get("target_company")

    capital_amount = conversation_output.capital_amount
    if capital_amount is None:
        capital_amount = profile.get("capital_amount")

    time_horizon = conversation_output.time_horizon
    if not time_horizon:
        time_horizon = profile.get("time_horizon")

    risk_tolerance = conversation_output.risk_tolerance
    if not risk_tolerance:
        risk_tolerance = profile.get("risk_tolerance")

    resolved_entities: Dict[str, Any] = {
        "company": company,
        "capital_amount": capital_amount,
        "time_horizon": time_horizon,
        "risk_tolerance": risk_tolerance,
    }

    missing: List[str] = []

    if is_investment_intent(
        intent_type=conversation_output.intent_type,
        query=conversation_output.normalized_query,
    ):
        # Full investor profile required for investment decision
        if not company:
            missing.append("company")
        if capital_amount is None:
            missing.append("capital_amount")
        if not time_horizon:
            missing.append("time_horizon")
        if not risk_tolerance:
            missing.append("risk_tolerance")
    else:
        # Factual or informational request: investor parameters not required.
        # If it's a general inquiry or concept definition, no company is required.
        clean_intent = conversation_output.intent_type.strip().lower()
        if clean_intent not in {"general_inquiry", "educational", "out_of_scope"}:
            query_lower = conversation_output.normalized_query.lower()
            # If query is a general definition, don't require company
            is_definition_query = (
                query_lower.startswith("what is a ")
                or query_lower.startswith("what is an ")
                or query_lower.startswith("explain ")
                or query_lower.startswith("what does ")
                or "definition" in query_lower
            )
            if not is_definition_query:
                company_specific_keywords = [
                    "p/e",
                    "pe ratio",
                    "earnings",
                    "revenue",
                    "balance sheet",
                    "stock price",
                    "fundamentals",
                    "financials",
                    "valuation",
                ]
                is_company_metric_query = any(
                    k in query_lower for k in company_specific_keywords
                )
                if is_company_metric_query and not company:
                    missing.append("company")

    return missing, resolved_entities


def format_clarification_prompt(
    normalized_query: str,
    missing_parameters: List[str],
    provided_parameters: Dict[str, Any],
) -> str:
    """Format prompt for LLM-based targeted clarification question generation.

    Args:
        normalized_query: Cleaned user query.
        missing_parameters: List of missing parameter names.
        provided_parameters: Dictionary of already provided parameter values.

    Returns:
        str: Fully formatted prompt.
    """
    missing_str = ", ".join(missing_parameters)
    sections = [
        CLARIFICATION_SYSTEM_PROMPT,
        f'User query:\n"{normalized_query}"',
        f"Missing required parameters that must be collected: {missing_str}",
    ]

    provided_entries = [
        f"{k}: {v}" for k, v in provided_parameters.items() if v is not None
    ]
    if provided_entries:
        prov_str = ", ".join(provided_entries)
        sections.append(
            f"Parameters already provided (DO NOT ASK FOR THESE): {prov_str}"
        )

    sections.append(
        "Generate concise, courteous, natural-language clarification questions "
        "asking ONLY for the missing parameters."
    )
    return "\n\n".join(sections)


class ClarificationAgent(BaseAgent):
    """Agent that evaluates investor parameters and clarifies missing fields.

    Converts a ClarificationInput into a structured ClarificationOutput.
    Uses deterministic completeness evaluation and targeted question generation.
    """

    name = "clarification_agent"
    input_schema = ClarificationInput
    output_schema = ClarificationOutput

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        tools: Optional[list[Tool]] = None,
        use_llm: bool = True,
    ) -> None:
        """Initialize the ClarificationAgent.

        Args:
            provider: Optional LLMProvider instance. Lazily fetched if None.
            tools: Optional sequence of tools available to this agent.
            use_llm: Whether to use LLM to generate questions (default True).
                     When False, uses deterministic template questions.
        """
        super().__init__(tools=tools)
        self._provider = provider
        self._use_llm = use_llm

    @property
    def provider(self) -> LLMProvider:
        """Return configured LLMProvider, resolving default if unassigned."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def run(
        self,
        input_data: Union[ClarificationInput, ConversationOutput, Dict[str, Any]],
    ) -> AgentResult:
        """Evaluate completeness and generate clarification questions if needed.

        Args:
            input_data: ClarificationInput, ConversationOutput, or dictionary.

        Returns:
            AgentResult: Successful result containing ClarificationOutput
            (with confidence=None for uncalibrated semantic extraction),
            or failed result with error description.
        """
        # Normalize input_data into a typed ClarificationInput
        if isinstance(input_data, ConversationOutput):
            clarification_input = ClarificationInput(conversation_output=input_data)
        elif isinstance(input_data, ClarificationInput):
            clarification_input = input_data
        elif isinstance(input_data, dict):
            try:
                if "conversation_output" in input_data:
                    clarification_input = ClarificationInput.model_validate(input_data)
                elif "normalized_query" in input_data:
                    conv_out = ConversationOutput.model_validate(input_data)
                    clarification_input = ClarificationInput(
                        conversation_output=conv_out
                    )
                elif "clarified_request" in input_data:
                    # Support adapting from a GraphState-style dictionary
                    cr = input_data.get("clarified_request") or {}
                    entities = cr.get("entities") or {}
                    conv_out = ConversationOutput(
                        normalized_query=cr.get(
                            "normalized_query", input_data.get("user_query", "")
                        ),
                        intent_type=cr.get("intent_type", "investment_analysis"),
                        company=entities.get("company"),
                        capital_amount=entities.get("capital_amount"),
                        time_horizon=entities.get("time_horizon"),
                        risk_tolerance=entities.get("risk_tolerance"),
                    )
                    clarification_input = ClarificationInput(
                        conversation_output=conv_out,
                        existing_profile=input_data.get("investor_profile"),
                    )
                else:
                    return AgentResult.create_failure(
                        error=(
                            "Unrecognized dictionary format for "
                            "ClarificationAgent input."
                        ),
                    )
            except Exception as err:
                return AgentResult.create_failure(
                    error=f"Failed to parse input dictionary: {err}",
                )
        else:
            return AgentResult.create_failure(
                error=(
                    "Input must be an instance of ClarificationInput, "
                    "ConversationOutput, or dict."
                ),
            )

        conv = clarification_input.conversation_output

        # 1. Deterministic completeness check
        missing_parameters, resolved_entities = determine_missing_parameters(
            conversation_output=conv,
            existing_profile=clarification_input.existing_profile,
        )

        clarification_needed = len(missing_parameters) > 0
        clarification_questions: List[str] = []

        # 2. Question generation if clarification is required
        if clarification_needed:
            if self._use_llm:
                prompt = format_clarification_prompt(
                    normalized_query=conv.normalized_query,
                    missing_parameters=missing_parameters,
                    provided_parameters=resolved_entities,
                )
                try:
                    questions_model = generate_structured(
                        provider=self.provider,
                        prompt=prompt,
                        schema=ClarificationQuestionsModel,
                    )
                    deduped = deduplicate_questions(questions_model.questions)
                    if deduped:
                        clarification_questions = deduped
                    else:
                        # Fallback to default questions if LLM returns empty list
                        clarification_questions = [
                            DEFAULT_PARAM_QUESTIONS[param]
                            for param in missing_parameters
                            if param in DEFAULT_PARAM_QUESTIONS
                        ]
                except Exception as err:
                    logger.error(
                        "ClarificationAgent question generation failed: %s", err
                    )
                    return AgentResult.create_failure(
                        error=f"Clarification question generation failed: {err}",
                    )
            else:
                # Deterministic question generation
                clarification_questions = [
                    DEFAULT_PARAM_QUESTIONS[param]
                    for param in missing_parameters
                    if param in DEFAULT_PARAM_QUESTIONS
                ]

        # 3. Assemble validated ClarificationOutput without invented defaults
        output = ClarificationOutput(
            clarification_needed=clarification_needed,
            missing_parameters=missing_parameters,
            clarification_questions=clarification_questions,
            normalized_query=conv.normalized_query,
            intent_type=conv.intent_type,
            company=resolved_entities.get("company"),
            capital_amount=resolved_entities.get("capital_amount"),
            time_horizon=resolved_entities.get("time_horizon"),
            risk_tolerance=resolved_entities.get("risk_tolerance"),
        )

        return AgentResult.create_success(
            data=output,
            confidence=None,
        )

    execute = run
