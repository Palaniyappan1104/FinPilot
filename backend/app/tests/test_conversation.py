"""Unit tests for Phase 3.2 ConversationAgent.

All tests are offline and mock the LLMProvider. No real Gemini API calls
or network requests are made.
"""

import json
from typing import Any, List, Optional

from app.agents import (
    AgentResult,
    ChatMessage,
    ConversationAgent,
    ConversationInput,
    ConversationOutput,
    GraphState,
    create_initial_state,
)
from app.agents.conversation import format_conversation_prompt
from app.core.llm import (
    LLMAuthenticationError,
    LLMProvider,
    LLMResponse,
)


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for unit testing ConversationAgent."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
    ) -> None:
        self.responses = responses or []
        self.call_count = 0
        self.prompts_received: List[str] = []
        self.fail_with = fail_with

    @property
    def provider_name(self) -> str:
        return "mock_conversation_llm"

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
            content = "{}"
        else:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            content = self.responses[idx]

        return LLMResponse(
            content=content,
            model="mock-conversation-v1",
            provider=self.provider_name,
        )


# ---------------------------------------------------------------------------
# 1. Basic Investment Request Test
# ---------------------------------------------------------------------------


def test_basic_investment_request():
    """Verify extraction of company, capital amount, and time horizon."""
    mock_json = json.dumps(
        {
            "normalized_query": "Should I invest 100000 in TCS for 5 years?",
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": 100000.0,
            "time_horizon": "5 years",
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    inp = ConversationInput(user_message="Should I invest ₹100000 in TCS for 5 years?")
    result = agent.run(inp)

    assert result.success is True
    assert result.error is None
    # Uncalibrated semantic confidence is represented as None
    assert result.confidence is None
    assert isinstance(result.data, ConversationOutput)

    output = result.data
    assert output.company == "TCS"
    assert output.capital_amount == 100000.0
    assert output.time_horizon == "5 years"
    assert output.risk_tolerance is None
    assert output.intent_type == "investment_analysis"


# ---------------------------------------------------------------------------
# 2. Request with Risk Tolerance Test
# ---------------------------------------------------------------------------


def test_request_with_risk_tolerance():
    """Verify extraction including explicit risk tolerance."""
    mock_json = json.dumps(
        {
            "normalized_query": (
                "Should I invest 50000 in Infosys for 3 years with conservative risk?"
            ),
            "intent_type": "investment_analysis",
            "company": "Infosys",
            "capital_amount": 50000.0,
            "time_horizon": "3 years",
            "risk_tolerance": "conservative",
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    inp = ConversationInput(
        user_message=(
            "Should I invest 50000 in Infosys for 3 years? "
            "I am a conservative investor."
        )
    )
    result = agent.run(inp)

    assert result.success is True
    assert result.confidence is None
    output = result.data
    assert output.company == "Infosys"
    assert output.capital_amount == 50000.0
    assert output.time_horizon == "3 years"
    assert output.risk_tolerance == "conservative"


# ---------------------------------------------------------------------------
# 3. Missing Optional Information Test
# ---------------------------------------------------------------------------


def test_missing_optional_information():
    """Verify missing entities remain None without synthetic defaults."""
    mock_json = json.dumps(
        {
            "normalized_query": "Analyze TCS stock.",
            "intent_type": "stock_research",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    result = agent.run("Analyze TCS.")

    assert result.success is True
    assert result.confidence is None
    output = result.data
    assert output.company == "TCS"
    assert output.capital_amount is None
    assert output.time_horizon is None
    assert output.risk_tolerance is None


# ---------------------------------------------------------------------------
# 4. Conversation History Context Test
# ---------------------------------------------------------------------------


def test_conversation_history_context():
    """Verify multi-turn history is included in prompt and informs extraction."""
    mock_json = json.dumps(
        {
            "normalized_query": "Invest 100000 in TCS for 5 years.",
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": 100000.0,
            "time_horizon": "5 years",
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    history = [
        ChatMessage(role="user", content="Should I invest in TCS?"),
        ChatMessage(role="assistant", content="How much are you planning to invest?"),
    ]
    inp = ConversationInput(
        user_message="Around ₹1 lakh for 5 years.",
        conversation_history=history,
    )
    result = agent.run(inp)

    assert result.success is True
    assert result.confidence is None
    output = result.data
    assert output.company == "TCS"
    assert output.capital_amount == 100000.0
    assert output.time_horizon == "5 years"

    # Verify that conversation history was included in the formatted prompt
    assert len(provider.prompts_received) == 1
    sent_prompt = provider.prompts_received[0]
    assert "Conversation history:" in sent_prompt
    assert "User: Should I invest in TCS?" in sent_prompt
    assert "Assistant: How much are you planning to invest?" in sent_prompt
    assert 'Current user message:\n"Around ₹1 lakh for 5 years."' in sent_prompt


# ---------------------------------------------------------------------------
# 5. LLM Failure Handling Test
# ---------------------------------------------------------------------------


def test_llm_failure_handling():
    """Verify provider exception produces a graceful failed AgentResult."""
    provider = MockLLMProvider(
        fail_with=LLMAuthenticationError(
            message="Mock auth failure",
            provider="mock",
        )
    )
    agent = ConversationAgent(provider=provider)

    result = agent.run("Analyze Reliance")

    assert result.success is False
    assert result.data is None
    assert result.confidence is None
    assert "Mock auth failure" in result.error


# ---------------------------------------------------------------------------
# 6. Structured Output Validation / Error Behavior Test
# ---------------------------------------------------------------------------


def test_structured_output_malformed_json_exhaustion():
    """Verify malformed JSON from provider results in structured failure."""
    provider = MockLLMProvider(responses=["not valid json 1", "not valid json 2"])
    agent = ConversationAgent(provider=provider)

    result = agent.run("Analyze TCS")

    assert result.success is False
    assert result.data is None
    assert "Structured output validation failed" in result.error
    assert provider.call_count == 2  # Reused Phase 2 bounded retry


def test_structured_output_retry_success():
    """Verify bounded retry recovers if attempt 1 is malformed and attempt 2 valid."""
    valid_json = json.dumps(
        {
            "normalized_query": "Analyze HDFC",
            "intent_type": "research",
            "company": "HDFC",
        }
    )
    provider = MockLLMProvider(responses=["bad json attempt 1", valid_json])
    agent = ConversationAgent(provider=provider)

    result = agent.run("Analyze HDFC")

    assert result.success is True
    assert result.confidence is None
    assert result.data.company == "HDFC"
    assert provider.call_count == 2


# ---------------------------------------------------------------------------
# 7. Agent Interface & Graph Integration Tests
# ---------------------------------------------------------------------------


def test_conversation_agent_properties():
    """Verify agent contract metadata."""
    agent = ConversationAgent(provider=MockLLMProvider())
    assert agent.name == "conversation_agent"
    assert agent.input_schema == ConversationInput
    assert agent.output_schema == ConversationOutput


def test_conversation_agent_run_returns_agent_result():
    """Verify agent.run() returns an AgentResult focused on conversation domain."""
    mock_json = json.dumps(
        {
            "normalized_query": "Analyze TCS fundamentals",
            "intent_type": "stock_research",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    inp = ConversationInput(user_message="Analyze TCS fundamentals")
    result = agent.run(inp)

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.confidence is None
    assert isinstance(result.data, ConversationOutput)
    assert result.data.company == "TCS"


def test_conversation_agent_graph_layer_adaptation():
    """Verify graph integration layer adapts ConversationAgent to GraphState."""
    mock_json = json.dumps(
        {
            "normalized_query": "Analyze TCS fundamentals",
            "intent_type": "stock_research",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    state = create_initial_state(user_query="Analyze TCS fundamentals")

    # Graph-layer adapter function translating between GraphState and AgentResult
    def conversation_node(s: GraphState) -> dict:
        inp = ConversationInput(user_message=s["user_query"])
        res = agent.run(inp)
        if res.success and isinstance(res.data, ConversationOutput):
            return {"clarified_request": res.data.to_clarified_request()}
        return {"clarified_request": None}

    update = conversation_node(state)
    assert "clarified_request" in update
    clarified = update["clarified_request"]
    assert clarified is not None
    assert clarified["normalized_query"] == "Analyze TCS fundamentals"
    assert clarified["intent_type"] == "stock_research"
    assert clarified["entities"] == {"company": "TCS"}


def test_format_conversation_prompt_without_history():
    """Verify format_conversation_prompt produces prompt without history section."""
    prompt = format_conversation_prompt(user_message="Hello FinPilot")
    assert "Conversation history:" not in prompt
    assert 'Current user message:\n"Hello FinPilot"' in prompt
