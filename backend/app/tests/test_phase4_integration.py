"""Integration tests for Phase 4.4 Testing: Clarification Flow in LangGraph.

Requirements from plan.md:
4.4.1 Unit tests: fully specified query (no clarification needed)
4.4.2 Unit tests: partially specified query (some fields missing)
4.4.3 Unit tests: multi-turn clarification resolves to a complete profile

All tests run 100% offline using deterministic mock providers.
ZERO real Gemini API or network calls are made.
"""

import json
from typing import Any, List, Optional

from app.agents import (
    ROUTE_CLARIFICATION_REQUIRED,
    ROUTE_READY_FOR_ANALYSIS,
    ClarificationAgent,
    ClarificationInput,
    ConversationAgent,
    GraphState,
    InvestorProfile,
    create_conversation_graph,
    create_initial_state,
    run_conversation_graph,
    should_continue_after_clarification,
)
from app.core.llm import (
    LLMAuthenticationError,
    LLMProvider,
    LLMResponse,
)


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for Phase 4.4 integration testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_phase4_provider",
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
            model="mock-phase4-v1",
            provider=self.provider_name,
        )


# ===========================================================================
# 4.4.1 Fully Specified Query Tests
# ===========================================================================


def test_4_4_1_fully_specified_investment_query_direct_workflow():
    """Verify complete investment query executes graph to ready_for_analysis.

    Roadmap 4.4.1:
    - ConversationAgent extracts all parameters.
    - ClarificationAgent executes without generating questions.
    - clarification_needed == False.
    - profile_complete == True.
    - clarification_questions is empty.
    - Routing reaches ready_for_analysis.
    - Zero clarification LLM calls needed.
    """
    mock_conv_json = json.dumps(
        {
            "normalized_query": (
                "Should I invest ₹1,00,000 in TCS for 5 years with moderate risk?"
            ),
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": 100000.0,
            "time_horizon": "5 years",
            "risk_tolerance": "moderate",
        }
    )

    conv_provider = MockLLMProvider(responses=[mock_conv_json])
    clar_provider = MockLLMProvider()

    conv_agent = ConversationAgent(provider=conv_provider)
    clar_agent = ClarificationAgent(provider=clar_provider)

    graph = create_conversation_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    initial_state = create_initial_state(
        user_query="Should I invest ₹1,00,000 in TCS for 5 years with moderate risk?"
    )

    final_state: GraphState = graph.invoke(initial_state)

    # 1. ConversationAgent executed
    assert conv_provider.call_count == 1
    cr = final_state["clarified_request"]
    assert cr is not None
    assert cr["intent_type"] == "investment_analysis"
    entities = cr.get("entities") or {}
    assert entities.get("company") == "TCS"
    assert entities.get("capital_amount") == 100000.0
    assert entities.get("time_horizon") == "5 years"
    assert entities.get("risk_tolerance") == "moderate"

    # 2. ClarificationAgent executed without needing questions
    assert clar_provider.call_count == 0  # Zero LLM calls when complete
    assert cr["clarification_needed"] is False
    assert cr["clarification_questions"] == []

    # 3. Investor profile marked complete
    ip = final_state["investor_profile"]
    assert ip is not None
    assert ip["target_company"] == "TCS"
    assert ip["capital_amount"] == 100000.0
    assert ip["time_horizon"] == "5 years"
    assert ip["risk_tolerance"] == "moderate"
    assert ip["profile_complete"] is True

    # 4. Routing condition reaches ready_for_analysis
    route = should_continue_after_clarification(final_state)
    assert route == ROUTE_READY_FOR_ANALYSIS


def test_4_4_1_fully_specified_query_via_preexisting_investor_profile():
    """Verify query resolves to complete when existing profile provides missing fields.

    Roadmap 4.4.1:
    - Incoming query provides only company.
    - Pre-existing investor_profile provides capital, horizon, and risk.
    - Profile resolves to complete without clarification.
    """
    mock_conv_json = json.dumps(
        {
            "normalized_query": "Should I invest in Reliance Industries?",
            "intent_type": "investment_analysis",
            "company": "Reliance Industries",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )

    conv_provider = MockLLMProvider(responses=[mock_conv_json])
    clar_provider = MockLLMProvider()

    conv_agent = ConversationAgent(provider=conv_provider)
    clar_agent = ClarificationAgent(provider=clar_provider)

    preexisting_profile: InvestorProfile = {
        "target_company": None,
        "capital_amount": 500000.0,
        "time_horizon": "10 years",
        "risk_tolerance": "aggressive",
        "profile_complete": False,
    }

    final_state = run_conversation_graph(
        query="Should I invest in Reliance Industries?",
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        investor_profile=preexisting_profile,
    )

    # Merged result is complete
    cr = final_state["clarified_request"]
    assert cr["clarification_needed"] is False
    assert cr["clarification_questions"] == []

    ip = final_state["investor_profile"]
    assert ip["target_company"] == "Reliance Industries"
    assert ip["capital_amount"] == 500000.0
    assert ip["time_horizon"] == "10 years"
    assert ip["risk_tolerance"] == "aggressive"
    assert ip["profile_complete"] is True

    # Zero clarification LLM calls needed
    assert clar_provider.call_count == 0
    assert should_continue_after_clarification(final_state) == ROUTE_READY_FOR_ANALYSIS


# ===========================================================================
# 4.4.2 Partially Specified Query Tests
# ===========================================================================


def test_4_4_2_partially_specified_query_missing_company_only():
    """Verify query with parameters but missing company triggers clarification.

    Roadmap 4.4.2:
    - Query: 'I want to invest ₹50,000 for 3 years with conservative risk.'
    - company remains None.
    - missing_parameters contains only 'company'.
    - profile_complete == False.
    - clarification_needed == True.
    - Clarification question targets company.
    - Routing reaches clarification_required.
    """
    mock_conv_json = json.dumps(
        {
            "normalized_query": (
                "I want to invest ₹50,000 for 3 years with conservative risk."
            ),
            "intent_type": "investment_analysis",
            "company": None,
            "capital_amount": 50000.0,
            "time_horizon": "3 years",
            "risk_tolerance": "conservative",
        }
    )
    mock_clar_json = json.dumps(
        {
            "questions": [
                "Which company or stock ticker are you considering investing in?"
            ]
        }
    )

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    clar_agent = ClarificationAgent(
        provider=MockLLMProvider(responses=[mock_clar_json])
    )

    # 1. Direct unit agent verification of missing_parameters
    query_text = "I want to invest ₹50,000 for 3 years with conservative risk."
    conv_res = conv_agent.run(query_text)
    clar_res = clar_agent.run(conv_res.data)
    assert clar_res.data.missing_parameters == ["company"]
    assert clar_res.data.clarification_needed is True

    # 2. Graph execution verification
    final_state = run_conversation_graph(
        query="I want to invest ₹50,000 for 3 years with conservative risk.",
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    cr = final_state["clarified_request"]
    entities = cr.get("entities") or {}
    assert entities.get("company") is None
    assert entities.get("capital_amount") == 50000.0
    assert entities.get("time_horizon") == "3 years"
    assert entities.get("risk_tolerance") == "conservative"

    assert cr["clarification_needed"] is True
    assert len(cr["clarification_questions"]) == 1
    assert "company" in cr["clarification_questions"][0].lower()

    # Profile is incomplete
    ip = final_state["investor_profile"]
    assert ip["profile_complete"] is False
    assert ip["target_company"] is None
    assert ip["capital_amount"] == 50000.0
    assert ip["time_horizon"] == "3 years"
    assert ip["risk_tolerance"] == "conservative"

    # Route reaches clarification_required
    assert (
        should_continue_after_clarification(final_state) == ROUTE_CLARIFICATION_REQUIRED
    )


def test_4_4_2_partially_specified_query_multiple_missing_fields():
    """Verify query with company and capital triggers questions for remaining fields.

    Roadmap 4.4.2:
    - Query: 'Should I invest in TCS with ₹50,000?'
    - company preserved (TCS).
    - capital preserved (50000).
    - time_horizon and risk_tolerance missing.
    - Exactly those missing parameters identified.
    - Profile is incomplete and routes to clarification_required.
    """
    mock_conv_json = json.dumps(
        {
            "normalized_query": "Should I invest in TCS with ₹50,000?",
            "intent_type": "investment_analysis",
            "company": "TCS",
            "capital_amount": 50000.0,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    mock_clar_json = json.dumps(
        {
            "questions": [
                "What is your target investment horizon (e.g. 1 year, 5 years)?",
                (
                    "What is your risk tolerance "
                    "(e.g. conservative, moderate, aggressive)?"
                ),
            ]
        }
    )

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    clar_agent = ClarificationAgent(
        provider=MockLLMProvider(responses=[mock_clar_json])
    )

    # 1. Direct unit agent verification of missing parameters
    conv_res = conv_agent.run("Should I invest in TCS with ₹50,000?")
    clar_res = clar_agent.run(conv_res.data)
    assert set(clar_res.data.missing_parameters) == {
        "time_horizon",
        "risk_tolerance",
    }
    assert clar_res.data.clarification_needed is True

    # 2. Graph execution verification
    final_state = run_conversation_graph(
        query="Should I invest in TCS with ₹50,000?",
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    cr = final_state["clarified_request"]
    entities = cr.get("entities") or {}
    assert entities.get("company") == "TCS"
    assert entities.get("capital_amount") == 50000.0
    assert entities.get("time_horizon") is None
    assert entities.get("risk_tolerance") is None

    assert cr["clarification_needed"] is True
    assert len(cr["clarification_questions"]) == 2

    ip = final_state["investor_profile"]
    assert ip["target_company"] == "TCS"
    assert ip["capital_amount"] == 50000.0
    assert ip["time_horizon"] is None
    assert ip["risk_tolerance"] is None
    assert ip["profile_complete"] is False

    assert (
        should_continue_after_clarification(final_state) == ROUTE_CLARIFICATION_REQUIRED
    )


# ===========================================================================
# 4.4.3 Multi-Turn Clarification Tests
# ===========================================================================


def test_4_4_3_three_turn_clarification_graph_workflow_to_completion():
    """Verify complete 3-turn graph workflow accumulating parameters to completion.

    Roadmap 4.4.3:
    - Turn 1: 'Should I invest in Infosys?'
      -> company=Infosys, missing capital/horizon/risk, route=clarification_required
    - Turn 2: 'I have ₹2,00,000 for 5 years.'
      -> company preserved, capital=200000, horizon=5 years, missing risk,
         route=clarification_required
    - Turn 3: 'My risk tolerance is moderate.'
      -> company, capital, horizon preserved, risk=moderate, profile_complete=True,
         clarification_needed=False, route=ready_for_analysis
    """
    # --- Turn 1 ---
    mock_t1_conv = json.dumps(
        {
            "normalized_query": "Should I invest in Infosys?",
            "intent_type": "investment_analysis",
            "company": "Infosys",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    mock_t1_clar = json.dumps(
        {
            "questions": [
                "How much capital do you plan to invest?",
                "What is your target investment time horizon?",
                "What is your risk tolerance?",
            ]
        }
    )

    conv_agent_t1 = ConversationAgent(
        provider=MockLLMProvider(responses=[mock_t1_conv])
    )
    clar_agent_t1 = ClarificationAgent(
        provider=MockLLMProvider(responses=[mock_t1_clar])
    )

    # Direct unit verification of Turn 1 missing parameters
    t1_conv_res = conv_agent_t1.run("Should I invest in Infosys?")
    t1_clar_res = clar_agent_t1.run(t1_conv_res.data)
    assert t1_clar_res.data.missing_parameters == [
        "capital_amount",
        "time_horizon",
        "risk_tolerance",
    ]

    state_t1 = run_conversation_graph(
        query="Should I invest in Infosys?",
        conversation_agent=conv_agent_t1,
        clarification_agent=clar_agent_t1,
    )

    # Verify Turn 1 state
    entities_t1 = state_t1["clarified_request"].get("entities") or {}
    assert entities_t1.get("company") == "Infosys"
    assert state_t1["clarified_request"]["clarification_needed"] is True
    assert state_t1["investor_profile"]["profile_complete"] is False
    assert state_t1["investor_profile"]["target_company"] == "Infosys"
    assert should_continue_after_clarification(state_t1) == ROUTE_CLARIFICATION_REQUIRED

    # --- Turn 2: User answers with capital and horizon ---
    mock_t2_conv = json.dumps(
        {
            "normalized_query": "I have ₹2,00,000 for 5 years.",
            "intent_type": "investment_analysis",
            "company": None,
            "capital_amount": 200000.0,
            "time_horizon": "5 years",
            "risk_tolerance": None,
        }
    )
    mock_t2_clar = json.dumps(
        {
            "questions": [
                "What is your risk tolerance (e.g. conservative, moderate, aggressive)?"
            ]
        }
    )

    conv_agent_t2 = ConversationAgent(
        provider=MockLLMProvider(responses=[mock_t2_conv])
    )
    clar_agent_t2 = ClarificationAgent(
        provider=MockLLMProvider(responses=[mock_t2_clar])
    )

    # Direct unit verification of Turn 2 missing parameters with prior profile
    t2_conv_res = conv_agent_t2.run("I have ₹2,00,000 for 5 years.")
    t2_clar_res = clar_agent_t2.run(
        ClarificationInput(
            conversation_output=t2_conv_res.data,
            existing_profile=state_t1["investor_profile"],
        )
    )
    assert t2_clar_res.data.missing_parameters == ["risk_tolerance"]

    # Pass Turn 1 investor_profile into Turn 2
    state_t2 = run_conversation_graph(
        query="I have ₹2,00,000 for 5 years.",
        conversation_agent=conv_agent_t2,
        clarification_agent=clar_agent_t2,
        investor_profile=state_t1["investor_profile"],
    )

    # Verify Turn 2 state: company preserved, capital and horizon added
    assert state_t2["investor_profile"]["target_company"] == "Infosys"
    assert state_t2["investor_profile"]["capital_amount"] == 200000.0
    assert state_t2["investor_profile"]["time_horizon"] == "5 years"
    assert state_t2["investor_profile"]["risk_tolerance"] is None
    assert state_t2["investor_profile"]["profile_complete"] is False

    assert state_t2["clarified_request"]["clarification_needed"] is True
    assert len(state_t2["clarified_request"]["clarification_questions"]) == 1
    assert should_continue_after_clarification(state_t2) == ROUTE_CLARIFICATION_REQUIRED

    # --- Turn 3: User answers with risk tolerance ---
    mock_t3_conv = json.dumps(
        {
            "normalized_query": "My risk tolerance is moderate.",
            "intent_type": "investment_analysis",
            "company": None,
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": "moderate",
        }
    )

    conv_agent_t3 = ConversationAgent(
        provider=MockLLMProvider(responses=[mock_t3_conv])
    )
    # Turn 3 Clarification provider should not be called because profile is complete!
    clar_provider_t3 = MockLLMProvider()
    clar_agent_t3 = ClarificationAgent(provider=clar_provider_t3)

    # Direct unit verification of Turn 3 resolution
    t3_conv_res = conv_agent_t3.run("My risk tolerance is moderate.")
    t3_clar_res = clar_agent_t3.run(
        ClarificationInput(
            conversation_output=t3_conv_res.data,
            existing_profile=state_t2["investor_profile"],
        )
    )
    assert t3_clar_res.data.missing_parameters == []
    assert t3_clar_res.data.clarification_needed is False

    # Pass Turn 2 investor_profile into Turn 3
    state_t3 = run_conversation_graph(
        query="My risk tolerance is moderate.",
        conversation_agent=conv_agent_t3,
        clarification_agent=clar_agent_t3,
        investor_profile=state_t2["investor_profile"],
    )

    # Verify Turn 3 state: all parameters resolved, profile marked complete
    ip3 = state_t3["investor_profile"]
    assert ip3["target_company"] == "Infosys"
    assert ip3["capital_amount"] == 200000.0
    assert ip3["time_horizon"] == "5 years"
    assert ip3["risk_tolerance"] == "moderate"
    assert ip3["profile_complete"] is True

    cr3 = state_t3["clarified_request"]
    assert cr3["clarification_needed"] is False
    assert cr3["clarification_questions"] == []

    # Verify zero LLM calls on Turn 3 for clarification
    assert clar_provider_t3.call_count == 0

    # Route reaches ready_for_analysis
    assert should_continue_after_clarification(state_t3) == ROUTE_READY_FOR_ANALYSIS


# ===========================================================================
# Additional Coverage: Factual Query Multi-Turn & Failure Isolation
# ===========================================================================


def test_4_4_additional_two_turn_factual_query_missing_company_resolves():
    """Verify factual metrics query missing company clarifies then resolves.

    Turn 1: 'What is the P/E ratio and dividend yield?' -> asks for company
    Turn 2: 'For TCS' -> resolves company without asking for capital/horizon/risk.
    """
    mock_t1_conv = json.dumps(
        {
            "normalized_query": "What is the P/E ratio and dividend yield?",
            "intent_type": "stock_research",
            "company": None,
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    mock_t1_clar = json.dumps(
        {
            "questions": [
                "Which company or stock ticker would you like the P/E ratio for?"
            ]
        }
    )

    conv_agent_t1 = ConversationAgent(
        provider=MockLLMProvider(responses=[mock_t1_conv])
    )
    clar_agent_t1 = ClarificationAgent(
        provider=MockLLMProvider(responses=[mock_t1_clar])
    )

    # Unit verification of Turn 1
    t1_conv_res = conv_agent_t1.run("What is the P/E ratio and dividend yield?")
    t1_clar_res = clar_agent_t1.run(t1_conv_res.data)
    assert t1_clar_res.data.missing_parameters == ["company"]
    assert t1_clar_res.data.clarification_needed is True

    state_t1 = run_conversation_graph(
        query="What is the P/E ratio and dividend yield?",
        conversation_agent=conv_agent_t1,
        clarification_agent=clar_agent_t1,
    )

    assert state_t1["clarified_request"]["clarification_needed"] is True
    assert state_t1["investor_profile"]["profile_complete"] is False
    assert should_continue_after_clarification(state_t1) == ROUTE_CLARIFICATION_REQUIRED

    # Turn 2: User provides company
    mock_t2_conv = json.dumps(
        {
            "normalized_query": "For TCS",
            "intent_type": "stock_research",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )

    conv_agent_t2 = ConversationAgent(
        provider=MockLLMProvider(responses=[mock_t2_conv])
    )
    clar_provider_t2 = MockLLMProvider()
    clar_agent_t2 = ClarificationAgent(provider=clar_provider_t2)

    # Unit verification of Turn 2
    t2_conv_res = conv_agent_t2.run("For TCS")
    t2_clar_res = clar_agent_t2.run(
        ClarificationInput(
            conversation_output=t2_conv_res.data,
            existing_profile=state_t1["investor_profile"],
        )
    )
    assert t2_clar_res.data.missing_parameters == []
    assert t2_clar_res.data.clarification_needed is False

    state_t2 = run_conversation_graph(
        query="For TCS",
        conversation_agent=conv_agent_t2,
        clarification_agent=clar_agent_t2,
        investor_profile=state_t1["investor_profile"],
    )

    # Factual query resolves without requiring capital, horizon, or risk
    entities_t2 = state_t2["clarified_request"].get("entities") or {}
    assert entities_t2.get("company") == "TCS"
    assert state_t2["clarified_request"]["clarification_needed"] is False
    assert state_t2["investor_profile"]["profile_complete"] is True
    assert clar_provider_t2.call_count == 0
    assert should_continue_after_clarification(state_t2) == ROUTE_READY_FOR_ANALYSIS


def test_4_4_additional_failure_isolation_in_later_clarification_turn():
    """Verify provider failure during later turn preserves state.

    Ensures profile_complete remains False and records error.
    """
    # Turn 1 partial state
    existing_profile: InvestorProfile = {
        "target_company": "INFY",
        "capital_amount": 100000.0,
        "time_horizon": None,
        "risk_tolerance": None,
        "profile_complete": False,
    }

    mock_conv_json = json.dumps(
        {
            "normalized_query": "For 3 years",
            "intent_type": "investment_analysis",
            "company": None,
            "capital_amount": None,
            "time_horizon": "3 years",
            "risk_tolerance": None,
        }
    )

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    # Clarification provider fails on Turn 2
    failing_clar_agent = ClarificationAgent(
        provider=MockLLMProvider(
            fail_with=LLMAuthenticationError("Simulated LLM auth error")
        )
    )

    state = run_conversation_graph(
        query="For 3 years",
        conversation_agent=conv_agent,
        clarification_agent=failing_clar_agent,
        investor_profile=existing_profile,
    )

    # Profile complete MUST be False
    assert state["investor_profile"]["profile_complete"] is False
    # Prior attributes preserved
    assert state["investor_profile"]["target_company"] == "INFY"
    assert state["investor_profile"]["capital_amount"] == 100000.0

    # Error recorded in clarified_request
    cr = state["clarified_request"]
    assert cr["clarification_needed"] is True
    assert any("Simulated LLM auth error" in q for q in cr["clarification_questions"])

    # Routing stays at clarification_required
    assert should_continue_after_clarification(state) == ROUTE_CLARIFICATION_REQUIRED


def test_4_4_investor_profile_field_preservation_across_graph_executions():
    """Verify investor profile fields survive multi-turn graph execution."""
    custom_profile: InvestorProfile = {
        "target_company": "HDFCBANK",
        "capital_amount": 250000.0,
        "time_horizon": "3 years",
        "risk_tolerance": "moderate",
        "profile_complete": False,
    }

    mock_conv = json.dumps(
        {
            "normalized_query": "Proceed with analysis for HDFCBANK",
            "intent_type": "investment_analysis",
            "company": "HDFCBANK",
            "capital_amount": 250000.0,
            "time_horizon": "3 years",
            "risk_tolerance": "moderate",
        }
    )

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider())

    final_state = run_conversation_graph(
        query="Proceed with analysis for HDFCBANK",
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        investor_profile=custom_profile,
    )

    ip = final_state["investor_profile"]
    assert ip["profile_complete"] is True
    assert ip["target_company"] == "HDFCBANK"
    assert ip["capital_amount"] == 250000.0
    assert ip["time_horizon"] == "3 years"
    assert ip["risk_tolerance"] == "moderate"
    assert should_continue_after_clarification(final_state) == ROUTE_READY_FOR_ANALYSIS
