"""Unit tests for Phase 3.3 ClarificationAgent.

All tests are completely offline and use MockLLMProvider. No real Gemini API
calls, credentials, or network requests are used.
"""

import json
from typing import Any, List, Optional

from app.agents import (
    ClarificationAgent,
    ClarificationInput,
    ClarificationOutput,
    ConversationOutput,
    GraphState,
    create_initial_state,
)
from app.agents.clarification import (
    deduplicate_questions,
    format_clarification_prompt,
    is_investment_intent,
)
from app.core.llm import (
    LLMAuthenticationError,
    LLMProvider,
    LLMResponse,
)


class MockClarificationProvider(LLMProvider):
    """Deterministic mock provider for unit testing ClarificationAgent."""

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
        return "mock_clarification_llm"

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
            content = '{"questions": []}'
        else:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            content = self.responses[idx]

        return LLMResponse(
            content=content,
            model="mock-clarification-v1",
            provider=self.provider_name,
        )


# ---------------------------------------------------------------------------
# 1. Complete Investment Request Test
# ---------------------------------------------------------------------------


def test_complete_investment_request_no_clarification_needed():
    """Verify that a fully specified investment query requires no clarification."""
    conv_out = ConversationOutput(
        normalized_query="Invest 100000 in TCS for 5 years with moderate risk.",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=100000.0,
        time_horizon="5 years",
        risk_tolerance="moderate",
    )

    provider = MockClarificationProvider()
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is True
    assert result.error is None
    assert result.confidence is None
    # No LLM calls should be made when information is complete
    assert provider.call_count == 0

    output = result.data
    assert isinstance(output, ClarificationOutput)
    assert output.clarification_needed is False
    assert output.missing_parameters == []
    assert output.clarification_questions == []
    assert output.company == "TCS"
    assert output.capital_amount == 100000.0
    assert output.time_horizon == "5 years"
    assert output.risk_tolerance == "moderate"


# ---------------------------------------------------------------------------
# 2. Missing Duration Test
# ---------------------------------------------------------------------------


def test_missing_duration_triggers_clarification():
    """Verify missing duration triggers clarification and asks for horizon."""
    conv_out = ConversationOutput(
        normalized_query="Should I invest 100000 in TCS with moderate risk?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=100000.0,
        time_horizon=None,
        risk_tolerance="moderate",
    )

    mock_llm_json = json.dumps(
        {"questions": ["What is your intended investment duration (e.g. 3 years)?"]}
    )
    provider = MockClarificationProvider(responses=[mock_llm_json])
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is True
    assert provider.call_count == 1
    # Check that prompt mentions time_horizon
    assert "time_horizon" in provider.prompts_received[0]

    output = result.data
    assert output.clarification_needed is True
    assert output.missing_parameters == ["time_horizon"]
    assert len(output.clarification_questions) == 1
    q = output.clarification_questions[0].lower()
    assert "duration" in q or "year" in q
    # Preserved fields without invention
    assert output.company == "TCS"
    assert output.capital_amount == 100000.0
    assert output.time_horizon is None
    assert output.risk_tolerance == "moderate"


# ---------------------------------------------------------------------------
# 3. Missing Capital Test
# ---------------------------------------------------------------------------


def test_missing_capital_triggers_clarification():
    """Verify missing capital triggers clarification and asks for capital amount."""
    conv_out = ConversationOutput(
        normalized_query="Should I invest in TCS for 5 years with moderate risk?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=None,
        time_horizon="5 years",
        risk_tolerance="moderate",
    )

    mock_llm_json = json.dumps(
        {"questions": ["How much capital are you planning to invest in TCS?"]}
    )
    provider = MockClarificationProvider(responses=[mock_llm_json])
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is True
    assert provider.call_count == 1
    assert "capital_amount" in provider.prompts_received[0]

    output = result.data
    assert output.clarification_needed is True
    assert output.missing_parameters == ["capital_amount"]
    assert len(output.clarification_questions) == 1
    assert "capital" in output.clarification_questions[0].lower()
    assert output.capital_amount is None


# ---------------------------------------------------------------------------
# 4. Missing Risk Tolerance Test
# ---------------------------------------------------------------------------


def test_missing_risk_tolerance_triggers_clarification():
    """Verify missing risk tolerance triggers clarification and asks for risk."""
    conv_out = ConversationOutput(
        normalized_query="Should I invest 100000 in TCS for 5 years?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=100000.0,
        time_horizon="5 years",
        risk_tolerance=None,
    )

    mock_llm_json = json.dumps(
        {
            "questions": [
                "What is your risk tolerance (conservative, moderate, or aggressive)?"
            ]
        }
    )
    provider = MockClarificationProvider(responses=[mock_llm_json])
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is True
    assert provider.call_count == 1
    assert "risk_tolerance" in provider.prompts_received[0]

    output = result.data
    assert output.clarification_needed is True
    assert output.missing_parameters == ["risk_tolerance"]
    assert len(output.clarification_questions) == 1
    assert "risk" in output.clarification_questions[0].lower()
    assert output.risk_tolerance is None


# ---------------------------------------------------------------------------
# 5. Multiple Missing Parameters & No-Invention Test
# ---------------------------------------------------------------------------


def test_multiple_missing_parameters_and_no_invention():
    """Verify multiple missing parameters are identified and values are NOT invented."""
    conv_out = ConversationOutput(
        normalized_query="Should I invest in TCS?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=None,
        time_horizon=None,
        risk_tolerance=None,
    )

    mock_llm_json = json.dumps(
        {
            "questions": [
                "How much capital do you plan to invest?",
                "What is your investment duration?",
                "What is your risk tolerance (conservative, moderate, aggressive)?",
            ]
        }
    )
    provider = MockClarificationProvider(responses=[mock_llm_json])
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is True
    output = result.data
    assert output.clarification_needed is True
    assert output.missing_parameters == [
        "capital_amount",
        "time_horizon",
        "risk_tolerance",
    ]
    assert len(output.clarification_questions) == 3

    # NO-INVENTION RULE: absent parameters MUST remain None
    assert output.company == "TCS"
    assert output.capital_amount is None
    assert output.time_horizon is None
    assert output.risk_tolerance is None


# ---------------------------------------------------------------------------
# 6. Conversation Context & Multi-Turn Preservation Test
# ---------------------------------------------------------------------------


def test_conversation_context_preserves_prior_turn_information():
    """Verify that parameters captured in prior turns are not re-prompted."""
    # Existing session already captured company and capital
    existing_profile = {
        "target_company": "TCS",
        "capital_amount": 50000.0,
    }

    # Turn 2 provides the duration and risk tolerance
    turn2_output = ConversationOutput(
        normalized_query="I want to invest for 3 years with aggressive risk.",
        intent_type="investment_analysis",
        company=None,
        capital_amount=None,
        time_horizon="3 years",
        risk_tolerance="aggressive",
    )

    clarification_input = ClarificationInput(
        conversation_output=turn2_output,
        existing_profile=existing_profile,
    )

    provider = MockClarificationProvider()
    agent = ClarificationAgent(provider=provider)

    result = agent.run(clarification_input)

    assert result.success is True
    assert provider.call_count == 0  # Complete, no LLM call needed

    output = result.data
    assert output.clarification_needed is False
    assert output.missing_parameters == []
    assert output.clarification_questions == []

    # Merged profile contains all 4 fields
    assert output.company == "TCS"
    assert output.capital_amount == 50000.0
    assert output.time_horizon == "3 years"
    assert output.risk_tolerance == "aggressive"


def test_conversation_context_partial_turn_prompts_only_remaining_missing():
    """Verify that only the remaining missing parameters are asked in multi-turn."""
    existing_profile = {
        "target_company": "Infosys",
        "capital_amount": 75000.0,
    }

    # Turn 2 provides only time horizon, risk tolerance is still missing
    turn2_output = ConversationOutput(
        normalized_query="For a 2 year horizon.",
        intent_type="investment_analysis",
        company=None,
        capital_amount=None,
        time_horizon="2 years",
        risk_tolerance=None,
    )

    clarification_input = ClarificationInput(
        conversation_output=turn2_output,
        existing_profile=existing_profile,
    )

    mock_llm_json = json.dumps({"questions": ["What is your risk tolerance?"]})
    provider = MockClarificationProvider(responses=[mock_llm_json])
    agent = ClarificationAgent(provider=provider)

    result = agent.run(clarification_input)

    assert result.success is True
    output = result.data
    assert output.clarification_needed is True
    # Only risk_tolerance is missing; company, capital, and horizon are known
    assert output.missing_parameters == ["risk_tolerance"]
    assert len(output.clarification_questions) == 1
    assert output.company == "Infosys"
    assert output.capital_amount == 75000.0
    assert output.time_horizon == "2 years"
    assert output.risk_tolerance is None


# ---------------------------------------------------------------------------
# 7. Non-Investment / Factual Request Tests
# ---------------------------------------------------------------------------


def test_factual_company_metric_query_no_investor_profile_required():
    """Verify factual requests like P/E ratio do not require capital/horizon/risk."""
    conv_out = ConversationOutput(
        normalized_query="What is the P/E ratio of TCS?",
        intent_type="stock_research",
        company="TCS",
        capital_amount=None,
        time_horizon=None,
        risk_tolerance=None,
    )

    provider = MockClarificationProvider()
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is True
    assert provider.call_count == 0  # No LLM call

    output = result.data
    assert output.clarification_needed is False
    assert output.missing_parameters == []
    assert output.clarification_questions == []
    assert output.company == "TCS"


def test_general_factual_query_without_company_complete():
    """Verify concepts like 'What is a P/E ratio?' require no questions."""
    conv_out = ConversationOutput(
        normalized_query="What is a P/E ratio?",
        intent_type="general_inquiry",
        company=None,
        capital_amount=None,
        time_horizon=None,
        risk_tolerance=None,
    )

    provider = MockClarificationProvider()
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is True
    output = result.data
    assert output.clarification_needed is False
    assert output.missing_parameters == []
    assert output.clarification_questions == []


def test_factual_query_missing_company_asks_only_for_company():
    """Verify metric queries lacking a company ask ONLY for the company."""
    conv_out = ConversationOutput(
        normalized_query="What is the P/E ratio?",
        intent_type="stock_research",
        company=None,
        capital_amount=None,
        time_horizon=None,
        risk_tolerance=None,
    )

    mock_llm_json = json.dumps(
        {"questions": ["Which company's P/E ratio would you like to check?"]}
    )
    provider = MockClarificationProvider(responses=[mock_llm_json])
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is True
    output = result.data
    assert output.clarification_needed is True
    assert output.missing_parameters == ["company"]
    assert len(output.clarification_questions) == 1
    assert "company" in output.clarification_questions[0].lower()


# ---------------------------------------------------------------------------
# 8. Duplicate Question Prevention Test
# ---------------------------------------------------------------------------


def test_duplicate_question_prevention():
    """Verify that duplicate questions returned by the model are deduplicated."""
    raw_questions = [
        "What is your investment duration?",
        "  what is your investment duration?  ",
        "What is your risk tolerance?",
        "what is your risk tolerance?",
    ]
    deduped = deduplicate_questions(raw_questions)
    assert len(deduped) == 2
    assert deduped[0] == "What is your investment duration?"
    assert deduped[1] == "What is your risk tolerance?"

    # Test deduplication inside agent execution
    conv_out = ConversationOutput(
        normalized_query="Should I invest in TCS?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=None,
        time_horizon=None,
        risk_tolerance="moderate",
    )

    mock_llm_json = json.dumps(
        {
            "questions": [
                "How much capital do you plan to invest?",
                "How much capital do you plan to invest?",
                "What is your investment time horizon?",
            ]
        }
    )
    provider = MockClarificationProvider(responses=[mock_llm_json])
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)
    assert result.success is True
    output = result.data
    assert len(output.clarification_questions) == 2
    assert (
        output.clarification_questions[0] == "How much capital do you plan to invest?"
    )
    assert output.clarification_questions[1] == "What is your investment time horizon?"


# ---------------------------------------------------------------------------
# 9. LLM / Provider Failure Test
# ---------------------------------------------------------------------------


def test_provider_failure_returns_failed_agent_result():
    """Verify provider exceptions are caught and returned as failed AgentResult."""
    conv_out = ConversationOutput(
        normalized_query="Should I invest in TCS?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=None,
        time_horizon=None,
        risk_tolerance=None,
    )

    auth_err = LLMAuthenticationError(
        message="Invalid API credentials",
        provider="mock_clarification_llm",
    )
    provider = MockClarificationProvider(fail_with=auth_err)
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is False
    assert result.data is None
    assert "Invalid API credentials" in str(result.error)
    assert result.confidence is None


# ---------------------------------------------------------------------------
# 10. Structured Output Validation Error Test
# ---------------------------------------------------------------------------


def test_structured_output_retry_exhaustion_returns_failed_agent_result():
    """Verify malformed JSON exhausting 2 retries returns failed AgentResult."""
    conv_out = ConversationOutput(
        normalized_query="Should I invest in TCS?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=None,
        time_horizon=None,
        risk_tolerance=None,
    )

    # Returns malformed JSON both times
    provider = MockClarificationProvider(
        responses=["not valid json at all", "still not json"]
    )
    agent = ClarificationAgent(provider=provider)

    result = agent.run(conv_out)

    assert result.success is False
    assert result.data is None
    assert "Structured output validation failed" in str(result.error)
    assert provider.call_count == 2  # Exactly 2 attempts per Phase 2.2 contract


# ---------------------------------------------------------------------------
# 11. Deterministic Mode (use_llm=False) Test
# ---------------------------------------------------------------------------


def test_deterministic_question_generation_without_llm():
    """Verify questions can be generated deterministically without calling LLM."""
    conv_out = ConversationOutput(
        normalized_query="Should I invest in TCS?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=None,
        time_horizon=None,
        risk_tolerance=None,
    )

    provider = MockClarificationProvider()
    agent = ClarificationAgent(provider=provider, use_llm=False)

    result = agent.run(conv_out)

    assert result.success is True
    assert provider.call_count == 0  # LLM never called
    output = result.data
    assert output.clarification_needed is True
    assert len(output.clarification_questions) == 3
    assert any("capital" in q.lower() for q in output.clarification_questions)
    assert any("duration" in q.lower() for q in output.clarification_questions)
    assert any("risk" in q.lower() for q in output.clarification_questions)


# ---------------------------------------------------------------------------
# 12. Input Adaptability & Validation Error Handling
# ---------------------------------------------------------------------------


def test_invalid_input_type_returns_failure():
    """Verify unsupported input type returns a clean AgentResult failure."""
    agent = ClarificationAgent(provider=MockClarificationProvider())
    result = agent.run(12345)  # Unsupported type
    assert result.success is False
    assert "Input must be an instance of ClarificationInput" in result.error


def test_run_with_raw_dict_input():
    """Verify dict with 'conversation_output' or 'normalized_query' works."""
    raw_dict = {
        "normalized_query": "Invest 100000 in TCS for 5 years with moderate risk.",
        "intent_type": "investment_analysis",
        "company": "TCS",
        "capital_amount": 100000.0,
        "time_horizon": "5 years",
        "risk_tolerance": "moderate",
    }
    agent = ClarificationAgent(provider=MockClarificationProvider())
    result = agent.run(raw_dict)
    assert result.success is True
    assert result.data.clarification_needed is False


# ---------------------------------------------------------------------------
# 13. State Conversion & Graph Adapter Test
# ---------------------------------------------------------------------------


def test_clarification_output_to_state_conversions():
    """Verify conversion to ClarifiedRequest and InvestorProfile."""
    output = ClarificationOutput(
        clarification_needed=True,
        missing_parameters=["risk_tolerance"],
        clarification_questions=["What is your risk tolerance?"],
        normalized_query="Should I invest 100000 in TCS for 5 years?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=100000.0,
        time_horizon="5 years",
        risk_tolerance=None,
    )

    clarified_request = output.to_clarified_request()
    assert (
        clarified_request["normalized_query"]
        == "Should I invest 100000 in TCS for 5 years?"
    )
    assert clarified_request["intent_type"] == "investment_analysis"
    assert clarified_request["clarification_needed"] is True
    assert clarified_request["clarification_questions"] == [
        "What is your risk tolerance?"
    ]
    assert clarified_request["entities"]["company"] == "TCS"
    assert clarified_request["entities"]["capital_amount"] == 100000.0
    assert clarified_request["entities"]["time_horizon"] == "5 years"
    assert "risk_tolerance" not in clarified_request["entities"]

    investor_profile = output.to_investor_profile()
    assert investor_profile["target_company"] == "TCS"
    assert investor_profile["capital_amount"] == 100000.0
    assert investor_profile["time_horizon"] == "5 years"
    assert investor_profile["risk_tolerance"] is None
    assert investor_profile["profile_complete"] is False


def test_graph_node_adapter_execution():
    """Verify how a LangGraph workflow node adapts GraphState."""
    # 1. Simulate GraphState initialized with ConversationAgent's clarified_request
    state: GraphState = create_initial_state(
        user_query="Should I invest 100000 in TCS with moderate risk?"
    )
    state["clarified_request"] = {
        "normalized_query": "Should I invest 100000 in TCS with moderate risk?",
        "intent_type": "investment_analysis",
        "entities": {
            "company": "TCS",
            "capital_amount": 100000.0,
            "risk_tolerance": "moderate",
        },
        "clarification_needed": None,
        "clarification_questions": None,
    }

    # 2. Node adapter function
    def clarification_node(current_state: GraphState) -> dict:
        mock_resp = json.dumps({"questions": ["What is your investment duration?"]})
        agent = ClarificationAgent(
            provider=MockClarificationProvider(responses=[mock_resp])
        )
        res = agent.run(current_state)
        if not res.success:
            raise RuntimeError(res.error)

        clarification_out: ClarificationOutput = res.data
        return {
            "clarified_request": clarification_out.to_clarified_request(),
            "investor_profile": clarification_out.to_investor_profile(),
        }

    # 3. Execute adapter node
    node_update = clarification_node(state)

    # 4. Verify node returns conforming dictionary updates
    assert "clarified_request" in node_update
    assert "investor_profile" in node_update

    cr = node_update["clarified_request"]
    assert cr["clarification_needed"] is True
    assert cr["clarification_questions"] == ["What is your investment duration?"]

    ip = node_update["investor_profile"]
    assert ip["profile_complete"] is False
    assert ip["target_company"] == "TCS"
    assert ip["capital_amount"] == 100000.0
    assert ip["time_horizon"] is None
    assert ip["risk_tolerance"] == "moderate"


# ---------------------------------------------------------------------------
# 14. Helper Unit Tests
# ---------------------------------------------------------------------------


def test_is_investment_intent_detection():
    """Verify intent and keyword recognition for investment requests."""
    assert is_investment_intent("investment_analysis") is True
    assert is_investment_intent("stock_recommendation") is True
    assert is_investment_intent("stock_research", "Should I invest in TCS?") is True
    assert (
        is_investment_intent("stock_research", "What is the P/E ratio of TCS?") is False
    )
    assert is_investment_intent("general_inquiry", "Tell me what an ETF is.") is False


def test_format_clarification_prompt_structure():
    """Verify formatted prompt includes required sections."""
    prompt = format_clarification_prompt(
        normalized_query="Should I invest in TCS?",
        missing_parameters=["capital_amount", "time_horizon"],
        provided_parameters={"company": "TCS"},
    )
    assert 'User query:\n"Should I invest in TCS?"' in prompt
    assert (
        "Missing required parameters that must be collected: "
        "capital_amount, time_horizon"
    ) in prompt
    assert "Parameters already provided (DO NOT ASK FOR THESE): company: TCS" in prompt
