"""Unit and integration tests for Phase 4.3 Conversation Flow in LangGraph.

All tests run completely offline using deterministic MockLLMProvider instances.
Zero real Gemini API or network calls are made.

Tests cover:
1. Phase 2 baseline graph stability (create_graph remains START -> passthrough -> END).
2. Conversation flow graph compilation and topology.
3. Standalone execution of conversation_node and clarification_node.
4. Fully specified investment request (profile_complete=True,
   ready_for_analysis -> END).
5. Partially specified investment request (profile_complete=False,
   questions present -> END).
6. Preservation of initial GraphState fields across nodes.
7. ConversationAgent failure isolation (halts early, ClarificationAgent not called).
8. ClarificationAgent failure isolation (does not mark profile complete, handles error).
9. Dependency injection of custom/mock agents.
10. Independent unit testing of should_continue_after_clarification routing.
11. Factual query execution without requiring investor profile.
12. run_conversation_graph helper execution.
"""

import json
from typing import Any, List, Optional

from langgraph.graph import START
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    CLARIFICATION_NODE_NAME,
    CONVERSATION_NODE_NAME,
    PASSTHROUGH_NODE_NAME,
    ROUTE_CLARIFICATION_REQUIRED,
    ROUTE_READY_FOR_ANALYSIS,
    ClarificationAgent,
    ConversationAgent,
    GraphState,
    clarification_node,
    conversation_node,
    create_conversation_graph,
    create_graph,
    create_initial_state,
    run_conversation_graph,
    should_continue_after_clarification,
    should_continue_after_conversation,
)
from app.core.llm import (
    LLMAuthenticationError,
    LLMProvider,
    LLMResponse,
)


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for Phase 4.3 graph testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_flow_provider",
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
            model="mock-flow-v1",
            provider=self.provider_name,
        )


# ===========================================================================
# 1. Phase 2 Baseline Regression Protection
# ===========================================================================


def test_phase2_baseline_graph_unchanged():
    """Verify create_graph() retains Phase 2 pass-through baseline behavior."""
    graph = create_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert PASSTHROUGH_NODE_NAME in graph.nodes

    state = create_initial_state(user_query="Regression baseline check")
    result = graph.invoke(state)

    assert result["user_query"] == "Regression baseline check"
    assert result["clarified_request"] is None
    assert result["technical_result"] is None


# ===========================================================================
# 2. Phase 4.3 Graph Compilation & Topology
# ===========================================================================


def test_conversation_graph_compilation_and_nodes():
    """Verify create_conversation_graph compiles with conversation & clarification."""
    graph = create_conversation_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert hasattr(graph, "invoke")

    assert CONVERSATION_NODE_NAME in graph.nodes
    assert CLARIFICATION_NODE_NAME in graph.nodes
    assert START in graph.nodes or "__start__" in graph.nodes


# ===========================================================================
# 3. Direct Node Adapter Execution
# ===========================================================================


def test_conversation_node_direct_execution():
    """Verify conversation_node adapts user_query and returns clarified_request."""
    mock_json = json.dumps(
        {
            "normalized_query": "Analyze TCS stock",
            "intent_type": "stock_research",
            "company": "TCS",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_json]))
    state = create_initial_state(user_query="Analyze TCS stock")

    update = conversation_node(state, agent=agent)

    assert "clarified_request" in update
    cr = update["clarified_request"]
    assert cr["normalized_query"] == "Analyze TCS stock"
    assert cr["intent_type"] == "stock_research"
    assert cr["entities"]["company"] == "TCS"


def test_clarification_node_direct_execution():
    """Verify clarification_node consumes clarified_request and returns updates."""
    state = create_initial_state(
        user_query="Should I invest 100000 in TCS for 5 years with moderate risk?"
    )
    state["clarified_request"] = {
        "normalized_query": (
            "Should I invest 100000 in TCS for 5 years with moderate risk?"
        ),
        "intent_type": "investment_analysis",
        "entities": {
            "company": "TCS",
            "capital_amount": 100000.0,
            "time_horizon": "5 years",
            "risk_tolerance": "moderate",
        },
        "clarification_needed": None,
        "clarification_questions": None,
    }

    agent = ClarificationAgent(provider=MockLLMProvider())
    update = clarification_node(state, agent=agent)

    assert "clarified_request" in update
    assert "investor_profile" in update

    cr = update["clarified_request"]
    assert cr["clarification_needed"] is False
    assert cr["clarification_questions"] == []

    ip = update["investor_profile"]
    assert ip["profile_complete"] is True
    assert ip["target_company"] == "TCS"
    assert ip["capital_amount"] == 100000.0


# ===========================================================================
# 4. End-to-End Workflow Execution (Complete Request)
# ===========================================================================


def test_fully_specified_investment_request_workflow():
    """Verify complete request sets profile_complete=True and ready_for_analysis."""
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

    clar_provider = MockLLMProvider()
    clar_agent = ClarificationAgent(provider=clar_provider)

    graph = create_conversation_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    initial_state = create_initial_state(
        user_query="Should I invest ₹1,00,000 in TCS for 5 years with moderate risk?"
    )
    final_state = graph.invoke(initial_state)

    # 1. State integrity
    assert final_state["user_query"] == (
        "Should I invest ₹1,00,000 in TCS for 5 years with moderate risk?"
    )
    assert final_state["technical_result"] is None
    assert final_state["fundamental_result"] is None

    # 2. Clarified request state
    cr = final_state["clarified_request"]
    assert cr["clarification_needed"] is False
    assert cr["clarification_questions"] == []
    assert cr["entities"]["company"] == "TCS"

    # 3. Investor profile state
    ip = final_state["investor_profile"]
    assert ip["profile_complete"] is True
    assert ip["target_company"] == "TCS"
    assert ip["capital_amount"] == 100000.0
    assert ip["time_horizon"] == "5 years"
    assert ip["risk_tolerance"] == "moderate"

    # 4. Routing condition
    route = should_continue_after_clarification(final_state)
    assert route == ROUTE_READY_FOR_ANALYSIS


# ===========================================================================
# 5. End-to-End Workflow Execution (Partially Specified Request)
# ===========================================================================


def test_partially_specified_investment_request_workflow():
    """Verify partial request sets profile_complete=False and clarification_required."""
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
                "What is your investment duration?",
                "What is your risk tolerance?",
            ]
        }
    )
    clar_agent = ClarificationAgent(
        provider=MockLLMProvider(responses=[mock_questions_json])
    )

    graph = create_conversation_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    initial_state = create_initial_state(user_query="Should I invest in TCS?")
    final_state = graph.invoke(initial_state)

    # 1. Clarified request state
    cr = final_state["clarified_request"]
    assert cr["clarification_needed"] is True
    assert len(cr["clarification_questions"]) == 3

    # 2. Investor profile state
    ip = final_state["investor_profile"]
    assert ip["profile_complete"] is False
    assert ip["target_company"] == "TCS"
    assert ip["capital_amount"] is None
    assert ip["time_horizon"] is None
    assert ip["risk_tolerance"] is None

    # 3. Routing condition
    route = should_continue_after_clarification(final_state)
    assert route == ROUTE_CLARIFICATION_REQUIRED


# ===========================================================================
# 6. Failure Handling and Isolation
# ===========================================================================


def test_conversation_failure_halts_early_without_calling_clarification():
    """Verify ConversationAgent failure routes to END without invoking Clarification."""
    auth_err = LLMAuthenticationError(
        message="Invalid Gemini API key",
        provider="mock_flow_provider",
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(fail_with=auth_err))

    clar_provider = MockLLMProvider()
    clar_agent = ClarificationAgent(provider=clar_provider)

    graph = create_conversation_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    initial_state = create_initial_state(user_query="Analyze TCS")
    final_state = graph.invoke(initial_state)

    # ClarificationAgent was NEVER called
    assert clar_provider.call_count == 0

    # Conversation error captured safely
    cr = final_state["clarified_request"]
    assert cr["intent_type"] == "error"
    assert cr["clarification_needed"] is True
    assert "Invalid Gemini API key" in cr["clarification_questions"][0]

    # Workflow halted cleanly
    assert final_state["investor_profile"] is None
    assert should_continue_after_conversation(final_state) == "error"


def test_clarification_failure_does_not_mark_profile_complete():
    """Verify ClarificationAgent failure marks profile_complete=False and halts."""
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

    fail_err = RuntimeError("Clarification model connection error")
    clar_agent = ClarificationAgent(provider=MockLLMProvider(fail_with=fail_err))

    graph = create_conversation_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    initial_state = create_initial_state(user_query="Should I invest in TCS?")
    final_state = graph.invoke(initial_state)

    # Must NOT be marked complete
    ip = final_state["investor_profile"]
    assert ip["profile_complete"] is False

    # Clarification error recorded
    cr = final_state["clarified_request"]
    assert cr["clarification_needed"] is True
    assert "Clarification model connection error" in cr["clarification_questions"][0]

    # Routing halts at clarification_required
    assert (
        should_continue_after_clarification(final_state) == ROUTE_CLARIFICATION_REQUIRED
    )


# ===========================================================================
# 7. Independent Unit Testing of Conditional Routers
# ===========================================================================


def test_should_continue_after_clarification_independent_cases():
    """Verify router logic across diverse complete, partial, and empty states."""
    # 1. Complete profile -> ready_for_analysis
    complete_state: GraphState = {
        "user_query": "Test",
        "investor_profile": {"profile_complete": True},
        "clarified_request": {"clarification_needed": False},
    }
    assert (
        should_continue_after_clarification(complete_state) == ROUTE_READY_FOR_ANALYSIS
    )

    # 2. Missing profile_complete -> clarification_required
    incomplete_state_1: GraphState = {
        "user_query": "Test",
        "investor_profile": {"profile_complete": False},
        "clarified_request": {"clarification_needed": False},
    }
    assert (
        should_continue_after_clarification(incomplete_state_1)
        == ROUTE_CLARIFICATION_REQUIRED
    )

    # 3. clarification_needed is True -> clarification_required
    incomplete_state_2: GraphState = {
        "user_query": "Test",
        "investor_profile": {"profile_complete": True},
        "clarified_request": {"clarification_needed": True},
    }
    assert (
        should_continue_after_clarification(incomplete_state_2)
        == ROUTE_CLARIFICATION_REQUIRED
    )

    # 4. Completely empty state -> clarification_required
    empty_state: GraphState = {"user_query": "Test"}
    assert (
        should_continue_after_clarification(empty_state) == ROUTE_CLARIFICATION_REQUIRED
    )


def test_should_continue_after_conversation_independent_cases():
    """Verify conversation routing on success and error states."""
    success_state: GraphState = {
        "user_query": "Test",
        "clarified_request": {"intent_type": "investment_analysis"},
    }
    assert should_continue_after_conversation(success_state) == "continue"

    error_state: GraphState = {
        "user_query": "Test",
        "clarified_request": {"intent_type": "error"},
    }
    assert should_continue_after_conversation(error_state) == "error"

    missing_state: GraphState = {"user_query": "Test"}
    assert should_continue_after_conversation(missing_state) == "error"


# ===========================================================================
# 8. Factual Query Workflow & Helper Execution
# ===========================================================================


def test_factual_query_workflow_ready_for_analysis():
    """Verify factual query needs no constraints and reaches ready_for_analysis."""
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
    clar_agent = ClarificationAgent(provider=MockLLMProvider())

    graph = create_conversation_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    initial_state = create_initial_state(user_query="What is the P/E ratio of TCS?")
    final_state = graph.invoke(initial_state)

    assert final_state["clarified_request"]["clarification_needed"] is False
    assert final_state["investor_profile"]["profile_complete"] is True
    assert should_continue_after_clarification(final_state) == ROUTE_READY_FOR_ANALYSIS


def test_run_conversation_graph_helper():
    """Verify run_conversation_graph helper executes properly."""
    mock_conv_json = json.dumps(
        {
            "normalized_query": "Analyze Infosys",
            "intent_type": "stock_research",
            "company": "Infosys",
            "capital_amount": None,
            "time_horizon": None,
            "risk_tolerance": None,
        }
    )
    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[mock_conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider())

    final_state = run_conversation_graph(
        query="Analyze Infosys",
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
    )

    assert final_state["user_query"] == "Analyze Infosys"
    assert final_state["clarified_request"]["entities"]["company"] == "Infosys"
