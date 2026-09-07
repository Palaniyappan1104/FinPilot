"""Unit tests for the minimal LangGraph workflow.

Phase 2.5 tests:
- Graph construction and compilation succeeds
- Graph topology contains the expected pass-through node and edges
- START -> passthrough -> END execution succeeds
- Initial user_query and investor_profile survive execution
- Resulting state matches the required GraphState schema
- Standalone execution helpers (run_graph, passthrough_node)
"""

from langgraph.graph import START
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    PASSTHROUGH_NODE_NAME,
    create_graph,
    create_initial_state,
    get_graph,
    passthrough_node,
    run_graph,
)


def test_graph_construction_succeeds():
    """Verify create_graph compiles into a valid CompiledStateGraph instance."""
    graph = create_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert hasattr(graph, "invoke")


def test_get_graph_alias():
    """Verify get_graph returns a compiled StateGraph."""
    graph = get_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_graph_contains_expected_nodes():
    """Verify the compiled graph contains the passthrough node."""
    graph = create_graph()
    # LangGraph exposes graph nodes through the underlying graph definition
    assert PASSTHROUGH_NODE_NAME in graph.nodes
    assert START in graph.nodes or "__start__" in graph.nodes


def test_graph_execution_start_passthrough_end():
    """Verify end-to-end execution of START -> passthrough -> END."""
    graph = create_graph()
    initial_state = create_initial_state(user_query="Analyze Reliance Industries")

    result = graph.invoke(initial_state)

    assert result is not None
    assert isinstance(result, dict)
    assert result["user_query"] == "Analyze Reliance Industries"


def test_initial_user_query_survives_execution():
    """Verify user_query is preserved exactly through graph execution."""
    custom_query = "What is the fair value of HDFC Bank given a 10-year horizon?"
    initial_state = create_initial_state(user_query=custom_query)
    graph = create_graph()

    result = graph.invoke(initial_state)
    assert result["user_query"] == custom_query


def test_investor_profile_survives_execution():
    """Verify pre-existing investor_profile is preserved across graph execution."""
    profile = {
        "target_company": "Infosys",
        "ticker": "INFY",
        "investment_goal": "Capital Appreciation",
        "time_horizon": "5 years",
        "capital_amount": 500000.0,
        "risk_tolerance": "moderate",
        "profile_complete": True,
    }
    initial_state = create_initial_state(
        user_query="Should I invest 5L in INFY?",
        investor_profile=profile,
    )
    graph = create_graph()

    result = graph.invoke(initial_state)
    assert result["investor_profile"] == profile
    assert result["investor_profile"]["ticker"] == "INFY"


def test_resulting_state_structure():
    """Verify the output state contains all required GraphState fields."""
    graph = create_graph()
    initial_state = create_initial_state(user_query="Analyze TCS")

    result = graph.invoke(initial_state)

    expected_keys = {
        "user_query",
        "investor_profile",
        "clarified_request",
        "technical_result",
        "fundamental_result",
        "news_result",
        "research_result",
        "risk_result",
        "aggregated_result",
        "report",
    }
    assert expected_keys.issubset(result.keys())

    # Ensure uninvoked agent results remain None
    assert result["clarified_request"] is None
    assert result["technical_result"] is None
    assert result["fundamental_result"] is None
    assert result["news_result"] is None
    assert result["research_result"] is None
    assert result["risk_result"] is None
    assert result["aggregated_result"] is None
    assert result["report"] is None


def test_passthrough_node_direct():
    """Verify passthrough_node function directly without graph execution."""
    state = create_initial_state(user_query="Direct test")
    update = passthrough_node(state)

    assert isinstance(update, dict)
    assert update == {}


def test_run_graph_helper():
    """Verify run_graph helper executes with default and custom queries."""
    result_default = run_graph()
    assert result_default["user_query"] == "Analyze TCS for long-term investment"

    custom_query = "Assess risk for Tata Motors"
    result_custom = run_graph(custom_query)
    assert result_custom["user_query"] == custom_query
