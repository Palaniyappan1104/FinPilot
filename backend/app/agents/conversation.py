"""Conversation Agent implementation for FinPilot.

Phase 3.2 implements the Conversation Agent responsible for interpreting
raw natural-language user queries, utilizing conversation history when available,
and producing validated structured output (ConversationOutput).
"""

from typing import List, Optional, Union

from app.agents.base import AgentResult, BaseAgent
from app.agents.conversation_schema import (
    ChatMessage,
    ConversationInput,
    ConversationOutput,
)
from app.agents.tools import Tool
from app.core.llm.base import LLMProvider
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger

logger = get_logger("app.agents.conversation")

SYSTEM_INSTRUCTION = (
    "You are FinPilot's conversation understanding component.\n"
    "Your sole task is to interpret the user's request, normalize their query,\n"
    "and extract structured financial entities if explicitly stated or clearly\n"
    "inferable from context.\n\n"
    "Guidelines:\n"
    "1. Interpret the user's current request in context with conversation history.\n"
    "2. Produce only the requested structured fields matching the output schema.\n"
    "3. Normalize the user's intent into 'normalized_query' preserving meaning.\n"
    "4. Classify intent type into 'intent_type' (e.g. 'stock_research',\n"
    "   'investment_analysis', 'portfolio_review', 'general_inquiry').\n"
    "5. Extract target company name or ticker symbol into 'company' if mentioned.\n"
    "6. Extract investment capital into 'capital_amount' as a number if mentioned.\n"
    "7. Extract time horizon into 'time_horizon' (e.g. '5 years') if mentioned.\n"
    "8. Extract risk tolerance into 'risk_tolerance' if mentioned.\n"
    "9. DO NOT invent missing information. Use null/None for unprovided fields.\n"
    "10. DO NOT perform financial analysis, calculate indicators, or give advice."
)


def format_conversation_prompt(
    user_message: str,
    conversation_history: Optional[List[ChatMessage]] = None,
) -> str:
    """Construct the prompt for conversation understanding and entity extraction.

    Args:
        user_message: Current natural language input from the user.
        conversation_history: Optional prior messages in the conversation.

    Returns:
        str: Fully formatted prompt for the LLM.
    """
    sections = [SYSTEM_INSTRUCTION]

    if conversation_history:
        history_lines = ["Conversation history:"]
        for msg in conversation_history:
            role_label = msg.role.capitalize()
            history_lines.append(f"{role_label}: {msg.content}")
        sections.append("\n".join(history_lines))

    sections.append(f'Current user message:\n"{user_message}"')
    sections.append(
        "Extract the structured ConversationOutput according to the schema."
    )
    return "\n\n".join(sections)


class ConversationAgent(BaseAgent):
    """Agent that normalizes user requests and extracts investment parameters.

    Converts a ConversationInput into a structured ConversationOutput using
    an LLMProvider and bounded structured output parsing.
    """

    name = "conversation_agent"
    input_schema = ConversationInput
    output_schema = ConversationOutput

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        tools: Optional[list[Tool]] = None,
    ) -> None:
        """Initialize the ConversationAgent.

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

    def run(self, input_data: Union[ConversationInput, str]) -> AgentResult:
        """Process conversation input and return an AgentResult.

        Takes a ConversationInput (or raw query string), formats the conversation
        prompt (including prior history if available), and requests structured output.

        Args:
            input_data: ConversationInput instance or raw user message string.

        Returns:
            AgentResult: Successful result containing ConversationOutput data
            (with confidence=None for uncalibrated semantic extraction),
            or failed result with error description.
        """
        if isinstance(input_data, str):
            try:
                input_data = ConversationInput(user_message=input_data)
            except Exception as err:
                return AgentResult.create_failure(
                    error=f"Invalid conversation input: {err}",
                )
        elif not isinstance(input_data, ConversationInput):
            return AgentResult.create_failure(
                error="Input must be an instance of ConversationInput or str.",
            )

        prompt = format_conversation_prompt(
            user_message=input_data.user_message,
            conversation_history=input_data.conversation_history,
        )

        try:
            output = generate_structured(
                provider=self.provider,
                prompt=prompt,
                schema=ConversationOutput,
            )
            return AgentResult.create_success(
                data=output,
                confidence=None,
            )
        except Exception as err:
            logger.error("ConversationAgent execution failed: %s", err)
            return AgentResult.create_failure(
                error=f"Conversation understanding failed: {err}",
            )

    execute = run
