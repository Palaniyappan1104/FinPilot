"""Minimal LangGraph workflow for FinPilot.

Phase 2.5 establishes the baseline LangGraph workflow wiring:
    START -> passthrough -> END

This verifies that the typed GraphState correctly flows through a compiled
LangGraph StateGraph without requiring LLM calls, external APIs, or complex routing.
"""

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.state import GraphState, create_initial_state
from app.core.logging import get_logger

logger = get_logger("app.agents.graph")

PASSTHROUGH_NODE_NAME = "passthrough"


def passthrough_node(state: GraphState) -> Dict[str, Any]:
    """Trivial pass-through node for validating graph flow.

    Accepts the current GraphState, logs the incoming user query,
    and returns an empty dictionary (no-op state update).

    Args:
        state: Current GraphState dictionary.

    Returns:
        Dict[str, Any]: State update dictionary (empty for pass-through).
    """
    logger.debug(
        "Executing pass-through node for user query: %s",
        state.get("user_query"),
    )
    return {}


def create_graph() -> CompiledStateGraph:
    """Build and compile the minimal LangGraph workflow.

    Workflow topology:
        START -> passthrough -> END

    Returns:
        CompiledStateGraph: The compiled, executable LangGraph instance.
    """
    builder = StateGraph(GraphState)
    builder.add_node(PASSTHROUGH_NODE_NAME, passthrough_node)
    builder.add_edge(START, PASSTHROUGH_NODE_NAME)
    builder.add_edge(PASSTHROUGH_NODE_NAME, END)
    return builder.compile()


get_graph = create_graph


def run_graph(
    query: str = "Analyze TCS for long-term investment",
) -> GraphState:
    """Convenience helper running the minimal graph with a user query.

    Args:
        query: User prompt to initialize the graph state with.

    Returns:
        GraphState: The resulting graph state dictionary after execution.
    """
    initial_state = create_initial_state(user_query=query)
    graph = get_graph()
    result: GraphState = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    sample_query = "Analyze TCS for long-term investment"
    print(f"--- Entering Graph with query: '{sample_query}' ---")
    result_state = run_graph(sample_query)
    print("--- Graph Execution Completed Successfully ---")
    print(f"Resulting user_query: {result_state.get('user_query')}")
    print(f"Resulting GraphState keys: {list(result_state.keys())}")
