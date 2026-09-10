"""LangGraph workflows for FinPilot.

Phase 2.5 established the baseline LangGraph workflow wiring:
    START -> passthrough -> END

Phase 4.3 implements the conversation flow integration:
    START -> conversation_node -> clarification_node -> conditional_routing -> END
"""

from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.clarification import ClarificationAgent
from app.agents.clarification_schema import (
    ClarificationInput,
    ClarificationOutput,
)
from app.agents.conversation import ConversationAgent
from app.agents.conversation_schema import ConversationOutput
from app.agents.state import (
    GraphState,
    InvestorProfile,
    create_initial_state,
)
from app.core.logging import get_logger

logger = get_logger("app.agents.graph")

PASSTHROUGH_NODE_NAME = "passthrough"
CONVERSATION_NODE_NAME = "conversation"
CLARIFICATION_NODE_NAME = "clarification"

ROUTE_CLARIFICATION_REQUIRED = "clarification_required"
ROUTE_READY_FOR_ANALYSIS = "ready_for_analysis"


# ---------------------------------------------------------------------------
# 1. Phase 2.5 Baseline Pass-Through Components (Preserved for backward compatibility)
# ---------------------------------------------------------------------------


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
    """Convenience helper running the baseline graph with a user query.

    Args:
        query: User prompt to initialize the graph state with.

    Returns:
        GraphState: The resulting graph state dictionary after execution.
    """
    initial_state = create_initial_state(user_query=query)
    graph = get_graph()
    result: GraphState = graph.invoke(initial_state)
    return result


# ---------------------------------------------------------------------------
# 2. Phase 4.3 Conversation Flow Adapter Nodes
# ---------------------------------------------------------------------------


def conversation_node(
    state: GraphState,
    agent: Optional[ConversationAgent] = None,
) -> Dict[str, Any]:
    """Execute ConversationAgent to interpret query and extract parameters.

    Adapts GraphState['user_query'] into a ConversationInput, invokes the agent,
    and converts the resulting ConversationOutput into a GraphState update.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured ConversationAgent instance.

    Returns:
        Dict[str, Any]: State update dictionary containing 'clarified_request'.
    """
    active_agent = agent or ConversationAgent()
    user_query = state.get("user_query", "")

    res = active_agent.run(user_query)
    if not res.success:
        logger.error("Conversation node execution failed: %s", res.error)
        return {
            "clarified_request": {
                "normalized_query": user_query,
                "intent_type": "error",
                "entities": None,
                "clarification_needed": True,
                "clarification_questions": [
                    f"Conversation understanding error: {res.error}"
                ],
            }
        }

    conv_out: ConversationOutput = res.data
    return {
        "clarified_request": conv_out.to_clarified_request(),
    }


def clarification_node(
    state: GraphState,
    agent: Optional[ClarificationAgent] = None,
) -> Dict[str, Any]:
    """Execute ClarificationAgent to evaluate completeness and generate questions.

    Consumes GraphState['clarified_request'] and optional
    GraphState['investor_profile'], invokes the agent, and updates
    'clarified_request' and 'investor_profile'.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured ClarificationAgent instance.

    Returns:
        Dict[str, Any]: State update dictionary containing 'clarified_request'
        and 'investor_profile'.
    """
    active_agent = agent or ClarificationAgent()
    clarified_req = state.get("clarified_request")
    if not clarified_req or clarified_req.get("intent_type") == "error":
        logger.warning(
            "Clarification node skipped: invalid or error clarified_request."
        )
        return {}

    existing_profile = state.get("investor_profile")
    entities = clarified_req.get("entities") or {}

    conv_out = ConversationOutput(
        normalized_query=clarified_req.get(
            "normalized_query", state.get("user_query", "")
        ),
        intent_type=clarified_req.get("intent_type", "investment_analysis"),
        company=entities.get("company"),
        capital_amount=entities.get("capital_amount"),
        time_horizon=entities.get("time_horizon"),
        risk_tolerance=entities.get("risk_tolerance"),
    )

    clar_input = ClarificationInput(
        conversation_output=conv_out,
        existing_profile=existing_profile,
    )

    res = active_agent.run(clar_input)
    if not res.success:
        logger.error("Clarification node execution failed: %s", res.error)
        profile_fallback: Dict[str, Any] = dict(existing_profile or {})
        profile_fallback["profile_complete"] = False
        return {
            "investor_profile": profile_fallback,
            "clarified_request": {
                **clarified_req,
                "clarification_needed": True,
                "clarification_questions": [
                    f"Clarification processing error: {res.error}"
                ],
            },
        }

    clar_out: ClarificationOutput = res.data
    return {
        "clarified_request": clar_out.to_clarified_request(),
        "investor_profile": clar_out.to_investor_profile(),
    }


# ---------------------------------------------------------------------------
# 3. Phase 4.3 Conditional Routing Functions
# ---------------------------------------------------------------------------


def should_continue_after_conversation(state: GraphState) -> str:
    """Evaluate whether to proceed to clarification or halt on error.

    Args:
        state: Current GraphState dictionary.

    Returns:
        str: 'continue' if conversation succeeded, otherwise 'error'.
    """
    clarified = state.get("clarified_request") or {}
    if not clarified or clarified.get("intent_type") == "error":
        return "error"
    return "continue"


def should_continue_after_clarification(state: GraphState) -> str:
    """Evaluate whether investor profile is complete to route workflow.

    Roadmap 4.3.2 conditional routing:
    - If profile_complete is True and clarification_needed is False ->
      'ready_for_analysis'
    - Otherwise -> 'clarification_required'

    Args:
        state: Current GraphState dictionary.

    Returns:
        str: 'ready_for_analysis' or 'clarification_required'.
    """
    profile = state.get("investor_profile") or {}
    clarified = state.get("clarified_request") or {}

    is_complete = bool(profile.get("profile_complete", False))
    needs_clarification = bool(clarified.get("clarification_needed", True))

    if is_complete and not needs_clarification:
        return ROUTE_READY_FOR_ANALYSIS
    return ROUTE_CLARIFICATION_REQUIRED


# ---------------------------------------------------------------------------
# 4. Phase 4.3 Conversation Flow Graph Builder
# ---------------------------------------------------------------------------


def create_conversation_graph(
    conversation_agent: Optional[ConversationAgent] = None,
    clarification_agent: Optional[ClarificationAgent] = None,
) -> CompiledStateGraph:
    """Build and compile the Phase 4 integrated conversation workflow.

    Workflow topology:
        START
          ↓
        conversation_node
          ↓ (should_continue_after_conversation)
          ├── error → END
          └── continue → clarification_node
                            ↓ (should_continue_after_clarification)
                            ├── clarification_required → END
                            └── ready_for_analysis → END

    Args:
        conversation_agent: Optional pre-configured or mocked ConversationAgent.
        clarification_agent: Optional pre-configured or mocked ClarificationAgent.

    Returns:
        CompiledStateGraph: The compiled conversation workflow.
    """
    conv_agent = conversation_agent or ConversationAgent()
    clar_agent = clarification_agent or ClarificationAgent()

    def _conv_step(state: GraphState) -> Dict[str, Any]:
        return conversation_node(state, agent=conv_agent)

    def _clar_step(state: GraphState) -> Dict[str, Any]:
        return clarification_node(state, agent=clar_agent)

    builder = StateGraph(GraphState)
    builder.add_node(CONVERSATION_NODE_NAME, _conv_step)
    builder.add_node(CLARIFICATION_NODE_NAME, _clar_step)

    builder.add_edge(START, CONVERSATION_NODE_NAME)

    builder.add_conditional_edges(
        CONVERSATION_NODE_NAME,
        should_continue_after_conversation,
        {
            "continue": CLARIFICATION_NODE_NAME,
            "error": END,
        },
    )

    builder.add_conditional_edges(
        CLARIFICATION_NODE_NAME,
        should_continue_after_clarification,
        {
            ROUTE_CLARIFICATION_REQUIRED: END,
            ROUTE_READY_FOR_ANALYSIS: END,
        },
    )

    return builder.compile()


def run_conversation_graph(
    query: str,
    conversation_agent: Optional[ConversationAgent] = None,
    clarification_agent: Optional[ClarificationAgent] = None,
    investor_profile: Optional[InvestorProfile] = None,
) -> GraphState:
    """Convenience helper to run the conversation workflow with an initial query.

    Args:
        query: User natural-language prompt.
        conversation_agent: Optional ConversationAgent instance.
        clarification_agent: Optional ClarificationAgent instance.
        investor_profile: Optional pre-existing investor profile.

    Returns:
        GraphState: The resulting state after workflow execution.
    """
    initial_state = create_initial_state(
        user_query=query,
        investor_profile=investor_profile,
    )
    graph = create_conversation_graph(
        conversation_agent=conversation_agent,
        clarification_agent=clarification_agent,
    )
    result: GraphState = graph.invoke(initial_state)
    return result
