"""Phase 3.4 Integration and Edge-Case Tests.

Comprehensive offline test suite for:
- 3.4.1: Varied phrasings (Indian numbering, colloquial units, varied horizons)
- 3.4.2: Missing/ambiguous entities and out-of-scope non-financial queries
- 3.4.3: Structured output schema validation and edge cases
- End-to-end chained pipeline: ConversationAgent -> ClarificationAgent
- Multi-turn clarification loop
- Factual query handling without investor constraint requirements
- Failure isolation and error handling

All tests run completely offline with MockLLMProvider. No real Gemini/network calls.
"""

import json
from typing import Any, List, Optional

import pytest
from pydantic import ValidationError

from app.agents import (
    ClarificationAgent,
    ClarificationInput,
    ClarificationOutput,
    ConversationAgent,
    ConversationOutput,
    GraphState,
    create_initial_state,
)
from app.core.llm import (
    LLMAuthenticationError,
    LLMProvider,
    LLMResponse,
)


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM provider for Phase 3 integration testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_phase3_provider",
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
            content = "{}"
        else:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            content = self.responses[idx]

        return LLMResponse(
            content=content,
            model="mock-phase3-v1",
            provider=self.provider_name,
        )


# ===========================================================================
# A. Varied Phrasings (Roadmap 3.4.1)
# ===========================================================================


def test_varied_phrasing_indian_numbering_format():
    """Verify Indian numbering format (₹1,00,000) extraction in ConversationAgent."""
    mock_json = json.dumps(
        {
            "normalized_query": ("Should I invest 100000 in Infosys for 5 years?"),
            "intent_type": "investment_analysis",
            "company": "Infosys",
            "capital_amount": 100000.0,
            "time_horizon": "5 years",
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    result = agent.run("Should I invest ₹1,00,000 in Infosys for 5 years?")

    assert result.success is True
    assert result.confidence is None
    output: ConversationOutput = result.data
    assert output.company == "Infosys"
    assert output.capital_amount == 100000.0
    assert output.time_horizon == "5 years"
    assert output.risk_tolerance is None


def test_varied_phrasing_colloquial_k_format():
    """Verify colloquial '50k' and risk phrasing extraction in ConversationAgent."""
    mock_json = json.dumps(
        {
            "normalized_query": (
                "Can I invest 50000 in Tata Motors with moderate risk for 2 years?"
            ),
            "intent_type": "investment_analysis",
            "company": "Tata Motors",
            "capital_amount": 50000.0,
            "time_horizon": "2 years",
            "risk_tolerance": "moderate",
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    result = agent.run("Can I put 50k into Tata Motors with moderate risk for 2 years?")

    assert result.success is True
    output: ConversationOutput = result.data
    assert output.company == "Tata Motors"
    assert output.capital_amount == 50000.0
    assert output.time_horizon == "2 years"
    assert output.risk_tolerance == "moderate"


def test_varied_phrasing_lakhs_and_growth_horizon():
    """Verify '10 lakhs' and 'long-term growth' extraction in ConversationAgent."""
    mock_json = json.dumps(
        {
            "normalized_query": (
                "I want to allocate 1000000 in TCS for long-term growth"
            ),
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": 1000000.0,
            "time_horizon": "long-term",
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    result = agent.run("I want to allocate 10 lakhs in TCS for long-term growth")

    assert result.success is True
    output: ConversationOutput = result.data
    assert output.company == "TCS"
    assert output.capital_amount == 1000000.0
    assert output.time_horizon == "long-term"


# ===========================================================================
# B. Missing / Ambiguous Entities (Roadmap 3.4.2)
# ===========================================================================


def test_missing_company_not_inferred():
    """Verify missing company is NOT fabricated and remains None."""
    mock_json = json.dumps(
        {
            "normalized_query": "Should I invest 50000 for 3 years?",
            "intent_type": "investment_analysis",
            "company": None,
            "capital_amount": 50000.0,
            "time_horizon": "3 years",
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    result = agent.run("Should I invest 50000 for 3 years?")

    assert result.success is True
    output: ConversationOutput = result.data
    assert output.company is None
    assert output.capital_amount == 50000.0
    assert output.time_horizon == "3 years"


def test_company_ticker_and_name_variations():
    """Verify company/ticker variations like TCS vs Tata Consultancy Services."""
    mock_json_ticker = json.dumps(
        {
            "normalized_query": "Analyze TCS stock",
            "intent_type": "stock_research",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json_ticker])
    agent = ConversationAgent(provider=provider)
    result_ticker = agent.run("Analyze TCS stock")
    assert result_ticker.data.company == "TCS"

    mock_json_full = json.dumps(
        {
            "normalized_query": "Analyze Tata Consultancy Services stock",
            "intent_type": "stock_research",
            "company": "Tata Consultancy Services",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    provider_full = MockLLMProvider(responses=[mock_json_full])
    agent_full = ConversationAgent(provider=provider_full)
    result_full = agent_full.run("Analyze Tata Consultancy Services stock")
    assert result_full.data.company == "Tata Consultancy Services"


def test_ambiguous_company_mention_does_not_fabricate():
    """Verify ambiguous references do not fabricate arbitrary companies."""
    mock_json = json.dumps(
        {
            "normalized_query": "Should I invest in that popular tech company?",
            "intent_type": "investment_analysis",
            "company": None,
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    provider = MockLLMProvider(responses=[mock_json])
    agent = ConversationAgent(provider=provider)

    result = agent.run("Should I invest in that popular tech company?")

    assert result.success is True
    output: ConversationOutput = result.data
    assert output.company is None


# ===========================================================================
# C. Out-of-Scope / Irrelevant Queries (Roadmap 3.3.1 & 3.4.2)
# ===========================================================================


def test_out_of_scope_weather_query_no_financial_fabrication():
    """Verify non-financial weather query produces out_of_scope intent and no data."""
    mock_conv_json = json.dumps(
        {
            "normalized_query": "What is the weather today?",
            "intent_type": "out_of_scope",
            "company": None,
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    conv_res = conv_agent.run("What is the weather today?")

    assert conv_res.success is True
    conv_out: ConversationOutput = conv_res.data
    assert conv_out.intent_type == "out_of_scope"
    assert conv_out.company is None
    assert conv_out.capital_amount is None
    assert conv_out.time_horizon is None
    assert conv_out.risk_tolerance is None

    # ClarificationAgent should also recognize out-of-scope and not ask
    # investor questions
    clar_agent = ClarificationAgent(provider=MockLLMProvider())
    clar_res = clar_agent.run(conv_out)

    assert clar_res.success is True
    clar_out: ClarificationOutput = clar_res.data
    assert clar_out.clarification_needed is False
    assert clar_out.missing_parameters == []
    assert clar_out.clarification_questions == []


def test_out_of_scope_coding_query_graceful_path():
    """Verify coding instruction produces out_of_scope intent with zero fabrication."""
    mock_conv_json = json.dumps(
        {
            "normalized_query": "Write a Python script for sorting.",
            "intent_type": "out_of_scope",
            "company": None,
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    conv_res = conv_agent.run("Write a Python script for sorting.")

    assert conv_res.success is True
    assert conv_res.data.intent_type == "out_of_scope"
    assert conv_res.data.company is None

    clar_agent = ClarificationAgent(provider=MockLLMProvider())
    clar_res = clar_agent.run(conv_res.data)
    assert clar_res.data.clarification_needed is False


def test_out_of_scope_generic_conversation_no_fabrication():
    """Verify general conversation produces general_inquiry with no financial values."""
    mock_conv_json = json.dumps(
        {
            "normalized_query": "Hello, how are you today?",
            "intent_type": "general_inquiry",
            "company": None,
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    conv_res = conv_agent.run("Hello, how are you today?")

    assert conv_res.success is True
    assert conv_res.data.intent_type == "general_inquiry"
    assert conv_res.data.company is None


# ===========================================================================
# D. Complete Investment Request Pipeline
# ===========================================================================


def test_complete_investment_request_pipeline():
    """Verify complete request flows through Conversation and Clarification agents."""
    # 1. Mock ConversationAgent response
    mock_conv_json = json.dumps(
        {
            "normalized_query": (
                "Should I invest 100000 in TCS for 5 years with moderate risk?"
            ),
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": 100000.0,
            "time_horizon": "5 years",
            "risk_tolerance": "moderate",
        }
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))

    # 2. ClarificationAgent provider (no calls expected because input is complete)
    clar_provider = MockLLMProvider()
    clar_agent = ClarificationAgent(provider=clar_provider)

    # 3. Execute pipeline
    raw_query = "Should I invest ₹100000 in TCS for 5 years with moderate risk?"
    conv_res = conv_agent.run(raw_query)
    assert conv_res.success is True

    clar_res = clar_agent.run(conv_res.data)
    assert clar_res.success is True
    assert clar_provider.call_count == 0  # Zero LLM calls when complete

    clar_out: ClarificationOutput = clar_res.data
    assert clar_out.clarification_needed is False
    assert clar_out.missing_parameters == []
    assert clar_out.clarification_questions == []
    assert clar_out.company == "TCS"
    assert clar_out.capital_amount == 100000.0
    assert clar_out.time_horizon == "5 years"
    assert clar_out.risk_tolerance == "moderate"

    # 4. State adapters
    cr = clar_out.to_clarified_request()
    assert cr["clarification_needed"] is False
    assert cr["clarification_questions"] == []
    assert cr["entities"]["company"] == "TCS"

    ip = clar_out.to_investor_profile()
    assert ip["profile_complete"] is True
    assert ip["target_company"] == "TCS"
    assert ip["capital_amount"] == 100000.0
    assert ip["time_horizon"] == "5 years"
    assert ip["risk_tolerance"] == "moderate"


# ===========================================================================
# E. Partially Specified Investment Request Pipeline
# ===========================================================================


def test_partially_specified_investment_request_pipeline():
    """Verify partial request triggers clarification and identifies missing fields."""
    mock_conv_json = json.dumps(
        {
            "normalized_query": "Should I invest in TCS?",
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))

    mock_questions_json = json.dumps(
        {
            "questions": [
                "How much capital do you plan to invest?",
                "What is your intended investment horizon (e.g. 3 years)?",
                "What is your risk tolerance (conservative, moderate, aggressive)?",
            ]
        }
    )
    clar_agent = ClarificationAgent(
        provider=MockLLMProvider(responses=[mock_questions_json])
    )

    # Execute pipeline
    conv_res = conv_agent.run("Should I invest in TCS?")
    assert conv_res.success is True

    clar_res = clar_agent.run(conv_res.data)
    assert clar_res.success is True

    clar_out: ClarificationOutput = clar_res.data
    assert clar_out.clarification_needed is True
    assert clar_out.missing_parameters == [
        "capital_amount",
        "time_horizon",
        "risk_tolerance",
    ]
    assert len(clar_out.clarification_questions) == 3

    # NO-INVENTION RULE: missing values remain None
    assert clar_out.company == "TCS"
    assert clar_out.capital_amount is None
    assert clar_out.time_horizon is None
    assert clar_out.risk_tolerance is None

    ip = clar_out.to_investor_profile()
    assert ip["profile_complete"] is False


# ===========================================================================
# F. Multi-Turn Clarification Simulation
# ===========================================================================


def test_multi_turn_clarification_pipeline():
    """Simulate a 3-turn clarification loop accumulating parameters to completion."""
    # --- Turn 1: "Should I invest in TCS?" ---
    mock_turn1_conv = json.dumps(
        {
            "normalized_query": "Should I invest in TCS?",
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    conv_agent_t1 = ConversationAgent(
        provider=MockLLMProvider(responses=[mock_turn1_conv])
    )
    t1_conv_res = conv_agent_t1.run("Should I invest in TCS?")

    clar_agent = ClarificationAgent(
        provider=MockLLMProvider(
            responses=[
                json.dumps(
                    {
                        "questions": [
                            "How much capital do you plan to invest?",
                            "What is your investment duration?",
                            "What is your risk tolerance?",
                        ]
                    }
                ),
                json.dumps(
                    {
                        "questions": [
                            "What is your risk tolerance (conservative or aggressive)?"
                        ]
                    }
                ),
            ]
        )
    )

    t1_clar_res = clar_agent.run(t1_conv_res.data)
    assert t1_clar_res.data.clarification_needed is True
    assert t1_clar_res.data.missing_parameters == [
        "capital_amount",
        "time_horizon",
        "risk_tolerance",
    ]

    # Session profile updated after Turn 1
    session_profile = {
        "target_company": t1_clar_res.data.company,
    }

    # --- Turn 2: User provides capital and horizon ---
    mock_turn2_conv = json.dumps(
        {
            "normalized_query": "I have 1 lakh to invest for 5 years.",
            "intent_type": "investment_analysis",
            "company": None,
            "capital_amount": 100000.0,
            "time_horizon": "5 years",
            "risk_tolerance": None,
        }
    )
    conv_agent_t2 = ConversationAgent(
        provider=MockLLMProvider(responses=[mock_turn2_conv])
    )
    t2_conv_res = conv_agent_t2.run("I have 1 lakh to invest for 5 years.")

    t2_input = ClarificationInput(
        conversation_output=t2_conv_res.data,
        existing_profile=session_profile,
    )
    t2_clar_res = clar_agent.run(t2_input)

    assert t2_clar_res.data.clarification_needed is True
    # Company, capital, and horizon are now known; only risk_tolerance is missing
    assert t2_clar_res.data.missing_parameters == ["risk_tolerance"]
    assert len(t2_clar_res.data.clarification_questions) == 1
    assert "risk" in t2_clar_res.data.clarification_questions[0].lower()

    # Session profile updated after Turn 2
    session_profile.update(
        {
            "capital_amount": t2_clar_res.data.capital_amount,
            "time_horizon": t2_clar_res.data.time_horizon,
        }
    )

    # --- Turn 3: User provides risk tolerance ---
    mock_turn3_conv = json.dumps(
        {
            "normalized_query": "My risk tolerance is aggressive.",
            "intent_type": "investment_analysis",
            "company": None,
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": "aggressive",
        }
    )
    conv_agent_t3 = ConversationAgent(
        provider=MockLLMProvider(responses=[mock_turn3_conv])
    )
    t3_conv_res = conv_agent_t3.run("My risk tolerance is aggressive.")

    t3_input = ClarificationInput(
        conversation_output=t3_conv_res.data,
        existing_profile=session_profile,
    )
    t3_clar_res = clar_agent.run(t3_input)

    # All 4 parameters are now resolved!
    assert t3_clar_res.data.clarification_needed is False
    assert t3_clar_res.data.missing_parameters == []
    assert t3_clar_res.data.clarification_questions == []
    assert t3_clar_res.data.company == "TCS"
    assert t3_clar_res.data.capital_amount == 100000.0
    assert t3_clar_res.data.time_horizon == "5 years"
    assert t3_clar_res.data.risk_tolerance == "aggressive"

    final_profile = t3_clar_res.data.to_investor_profile()
    assert final_profile["profile_complete"] is True


# ===========================================================================
# G. Factual Financial Query
# ===========================================================================


def test_factual_financial_query_pipeline():
    """Verify factual inquiry does NOT trigger investor constraints questions."""
    mock_conv_json = json.dumps(
        {
            "normalized_query": "What is the P/E ratio of TCS?",
            "intent_type": "stock_research",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    conv_res = conv_agent.run("What is the P/E ratio of TCS?")
    assert conv_res.success is True

    clar_provider = MockLLMProvider()
    clar_agent = ClarificationAgent(provider=clar_provider)

    clar_res = clar_agent.run(conv_res.data)
    assert clar_res.success is True
    assert clar_provider.call_count == 0  # No questions, no LLM call

    clar_out: ClarificationOutput = clar_res.data
    assert clar_out.clarification_needed is False
    assert clar_out.missing_parameters == []
    assert clar_out.clarification_questions == []
    assert clar_out.company == "TCS"


# ===========================================================================
# H. Structured Output Schema Validation (Roadmap 3.4.3)
# ===========================================================================


def test_conversation_output_schema_validation_rules():
    """Verify ConversationOutput schema validation and constraint rules."""
    # 1. Valid full output
    valid_out = ConversationOutput(
        normalized_query="Analyze Infosys",
        intent_type="stock_research",
        company="Infosys",
        capital_amount=50000.0,
        time_horizon="1 year",
        risk_tolerance="moderate",
    )
    assert valid_out.company == "Infosys"

    # 2. Empty normalized_query rejected
    with pytest.raises(ValidationError):
        ConversationOutput(
            normalized_query="",
            intent_type="stock_research",
        )

    # 3. Empty intent_type rejected
    with pytest.raises(ValidationError):
        ConversationOutput(
            normalized_query="Analyze Infosys",
            intent_type="   ",
        )

    # 4. Negative capital rejected
    with pytest.raises(ValidationError):
        ConversationOutput(
            normalized_query="Analyze Infosys",
            intent_type="stock_research",
            capital_amount=-100.0,
        )

    # 5. Whitespace optional strings stripped to None
    whitespace_out = ConversationOutput(
        normalized_query="Analyze Infosys",
        intent_type="stock_research",
        company="   ",
        time_horizon="",
        risk_tolerance="  ",
    )
    assert whitespace_out.company is None
    assert whitespace_out.time_horizon is None
    assert whitespace_out.risk_tolerance is None


def test_clarification_output_schema_validation_rules():
    """Verify ClarificationOutput schema validation and constraint rules."""
    # 1. Valid output
    valid_out = ClarificationOutput(
        clarification_needed=False,
        missing_parameters=[],
        clarification_questions=[],
        normalized_query="Should I invest in TCS?",
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=10000.0,
        time_horizon="3 years",
        risk_tolerance="conservative",
    )
    assert valid_out.clarification_needed is False

    # 2. Empty normalized query rejected
    with pytest.raises(ValidationError):
        ClarificationOutput(
            clarification_needed=False,
            normalized_query="   ",
            intent_type="investment_analysis",
        )

    # 3. Empty intent type rejected
    with pytest.raises(ValidationError):
        ClarificationOutput(
            clarification_needed=False,
            normalized_query="Invest in TCS",
            intent_type="",
        )

    # 4. Negative capital rejected
    with pytest.raises(ValidationError):
        ClarificationOutput(
            clarification_needed=False,
            normalized_query="Invest in TCS",
            intent_type="investment_analysis",
            capital_amount=-50.0,
        )


# ===========================================================================
# I. Failure Isolation
# ===========================================================================


def test_conversation_agent_provider_failure_isolation():
    """Verify ConversationAgent provider error fails cleanly without downstream call."""
    auth_err = LLMAuthenticationError(
        message="Authentication failed for Gemini",
        provider="mock_phase3_provider",
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(fail_with=auth_err))

    conv_res = conv_agent.run("Should I invest in TCS?")

    assert conv_res.success is False
    assert conv_res.data is None
    assert "Authentication failed for Gemini" in str(conv_res.error)
    assert conv_res.confidence is None


def test_clarification_agent_provider_failure_isolation():
    """Verify ClarificationAgent provider error fails cleanly and isolates state."""
    mock_conv_json = json.dumps(
        {
            "normalized_query": "Should I invest in TCS?",
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    conv_res = conv_agent.run("Should I invest in TCS?")
    assert conv_res.success is True

    # ClarificationAgent fails during question generation
    fail_err = RuntimeError("Clarification model endpoint timed out")
    clar_agent = ClarificationAgent(provider=MockLLMProvider(fail_with=fail_err))

    clar_res = clar_agent.run(conv_res.data)

    assert clar_res.success is False
    assert clar_res.data is None
    assert "Clarification model endpoint timed out" in str(clar_res.error)
    assert clar_res.confidence is None


def test_graph_state_isolation_on_pipeline_failure():
    """Verify workflow state remains uncorrupted if an agent fails."""
    state: GraphState = create_initial_state(user_query="Analyze TCS")

    auth_err = LLMAuthenticationError(
        message="Provider key invalid",
        provider="mock_phase3_provider",
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(fail_with=auth_err))
    conv_res = conv_agent.run(state["user_query"])

    assert conv_res.success is False

    # Pipeline node checks success before updating state
    if conv_res.success:
        state["clarified_request"] = conv_res.data.to_clarified_request()

    # State remains pristine
    assert state["clarified_request"] is None
    assert state["user_query"] == "Analyze TCS"
