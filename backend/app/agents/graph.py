"""LangGraph workflows for FinPilot.

Phase 2.5 established the baseline LangGraph workflow wiring:
    START -> passthrough -> END

Phase 4.3 implements the conversation flow integration:
    START -> conversation_node -> clarification_node -> conditional_routing -> END
"""

from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.cio import CIOAgent
from app.agents.cio_schema import (
    CIORoutingDecision,
    SpecialistName,
)
from app.agents.clarification import ClarificationAgent
from app.agents.clarification_schema import (
    ClarificationInput,
    ClarificationOutput,
)
from app.agents.conversation import ConversationAgent
from app.agents.conversation_schema import ConversationOutput
from app.agents.specialist_stubs import create_specialist_stub_node
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
CIO_NODE_NAME = "cio"
FAN_IN_NODE_NAME = "fan_in"

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


# ---------------------------------------------------------------------------
# 5. Phase 5.3 CIO Node, Fan-Out Router, Fan-In Node & Graph Builder
# ---------------------------------------------------------------------------


def cio_node(
    state: GraphState,
    agent: Optional[CIOAgent] = None,
) -> Dict[str, Any]:
    """Execute CIOAgent to determine specialist routing and tasks.

    Adapts GraphState['clarified_request'], GraphState['investor_profile'],
    and GraphState['documents_available'] into CIOAgent input, invokes the agent,
    and returns a state update containing 'cio_decision'.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured CIOAgent instance.

    Returns:
        Dict[str, Any]: State update dictionary containing 'cio_decision'.
    """
    active_agent = agent or CIOAgent()
    res = active_agent.run(state)
    if not res.success or not res.data:
        logger.error("CIO node execution failed: %s", res.error)
        return {}

    decision: CIORoutingDecision = res.data
    return {
        "cio_decision": decision.to_state(),
    }


def route_to_specialists(state: GraphState) -> List[str]:
    """Evaluate CIO routing decision to fan-out to selected specialists.

    Extracts selected specialists from GraphState['cio_decision'].
    If empty or no specialists match registered nodes, routes directly to 'fan_in'.

    Args:
        state: Current GraphState dictionary.

    Returns:
        List[str]: Node names of selected specialists, or ['fan_in'].
    """
    decision = state.get("cio_decision") or {}
    selected = decision.get("selected_specialists") or []

    valid_specialist_names = {s.value for s in SpecialistName}
    matched = [s for s in selected if s in valid_specialist_names]

    if not matched:
        logger.warning(
            "No valid specialists found in CIO decision; routing directly to fan_in."
        )
        return [FAN_IN_NODE_NAME]

    return matched


def fan_in_node(state: GraphState) -> Dict[str, Any]:
    """Fan-in collection node synchronizing parallel specialist outputs.

    Collects results from all executed specialist branches.
    Returns an empty state update (no-op), preserving all specialist results
    accumulated in the shared GraphState.

    Args:
        state: Current GraphState dictionary containing parallel specialist results.

    Returns:
        Dict[str, Any]: Empty update dict (preserves state unchanged).
    """
    logger.debug(
        "Fan-in collection completed. Results present: technical=%s, fundamental=%s, "
        "news=%s, research=%s, risk=%s",
        state.get("technical_result") is not None,
        state.get("fundamental_result") is not None,
        state.get("news_result") is not None,
        state.get("research_result") is not None,
        state.get("risk_result") is not None,
    )
    return {}


def create_orchestration_graph(
    conversation_agent: Optional[ConversationAgent] = None,
    clarification_agent: Optional[ClarificationAgent] = None,
    cio_agent: Optional[CIOAgent] = None,
    specialist_stubs: Optional[
        Dict[SpecialistName, Callable[[GraphState], Dict[str, Any]]]
    ] = None,
    specialist_timeout_seconds: Optional[float] = 30.0,
) -> CompiledStateGraph:
    """Build and compile the Phase 5 CIO-orchestrated analysis workflow.

    Workflow topology:
        START
          ↓
        conversation_node
          ↓ (should_continue_after_conversation)
          ├── error → END
          └── continue → clarification_node
                            ↓ (should_continue_after_clarification)
                            ├── clarification_required → END
                            └── ready_for_analysis → cio_node
                                                       ↓ (route_to_specialists)
                                                       ├── technical  ─┐
                                                       ├── fundamental ┼
                                                       ├── news ───────┼─> fan_in -> END
                                                       ├── research ───┤
                                                       ├── risk ───────┘
                                                       └── fan_in ───────> END

    Args:
        conversation_agent: Optional ConversationAgent instance.
        clarification_agent: Optional ClarificationAgent instance.
        cio_agent: Optional CIOAgent instance.
        specialist_stubs: Optional mapping of SpecialistName to custom node callable.
        specialist_timeout_seconds: Execution timeout per specialist node in seconds.

    Returns:
        CompiledStateGraph: The compiled orchestration workflow.
    """
    conv_agent = conversation_agent or ConversationAgent()
    clar_agent = clarification_agent or ClarificationAgent()
    active_cio = cio_agent or CIOAgent()

    def _conv_step(state: GraphState) -> Dict[str, Any]:
        return conversation_node(state, agent=conv_agent)

    def _clar_step(state: GraphState) -> Dict[str, Any]:
        return clarification_node(state, agent=clar_agent)

    def _cio_step(state: GraphState) -> Dict[str, Any]:
        return cio_node(state, agent=active_cio)

    builder = StateGraph(GraphState)

    # 1. Add conversation and clarification nodes
    builder.add_node(CONVERSATION_NODE_NAME, _conv_step)
    builder.add_node(CLARIFICATION_NODE_NAME, _clar_step)

    # 2. Add CIO router node
    builder.add_node(CIO_NODE_NAME, _cio_step)

    # 3. Add specialist stub nodes (Phase 5 placeholders)
    stubs = specialist_stubs or {}
    for spec in SpecialistName:
        custom_fn = stubs.get(spec)
        node_fn = create_specialist_stub_node(
            specialist_name=spec,
            timeout_seconds=specialist_timeout_seconds,
            custom_handler=custom_fn,
        )
        builder.add_node(spec.value, node_fn)

    # 4. Add fan-in collection node
    builder.add_node(FAN_IN_NODE_NAME, fan_in_node)

    # 5. Edges: START -> Conversation
    builder.add_edge(START, CONVERSATION_NODE_NAME)

    # 6. Edges: Conversation -> Clarification or END
    builder.add_conditional_edges(
        CONVERSATION_NODE_NAME,
        should_continue_after_conversation,
        {
            "continue": CLARIFICATION_NODE_NAME,
            "error": END,
        },
    )

    # 7. Edges: Clarification -> CIO or END
    builder.add_conditional_edges(
        CLARIFICATION_NODE_NAME,
        should_continue_after_clarification,
        {
            ROUTE_CLARIFICATION_REQUIRED: END,
            ROUTE_READY_FOR_ANALYSIS: CIO_NODE_NAME,
        },
    )

    # 8. Edges: CIO -> Fan-Out to Specialists (or Fan-In directly if none)
    specialist_targets = [s.value for s in SpecialistName] + [FAN_IN_NODE_NAME]
    builder.add_conditional_edges(
        CIO_NODE_NAME,
        route_to_specialists,
        specialist_targets,
    )

    # 9. Edges: Each specialist -> Fan-In (Fan-In collection)
    for spec in SpecialistName:
        builder.add_edge(spec.value, FAN_IN_NODE_NAME)

    # 10. Edges: Fan-In -> END
    builder.add_edge(FAN_IN_NODE_NAME, END)

    return builder.compile()


def run_orchestration_graph(
    query: str,
    conversation_agent: Optional[ConversationAgent] = None,
    clarification_agent: Optional[ClarificationAgent] = None,
    cio_agent: Optional[CIOAgent] = None,
    investor_profile: Optional[InvestorProfile] = None,
    documents_available: bool = False,
    specialist_stubs: Optional[
        Dict[SpecialistName, Callable[[GraphState], Dict[str, Any]]]
    ] = None,
    specialist_timeout_seconds: Optional[float] = 30.0,
) -> GraphState:
    """Convenience helper to run the complete Phase 5 orchestration workflow.

    Args:
        query: User natural language prompt.
        conversation_agent: Optional ConversationAgent instance.
        clarification_agent: Optional ClarificationAgent instance.
        cio_agent: Optional CIOAgent instance.
        investor_profile: Optional pre-existing investor profile.
        documents_available: Whether documents are attached.
        specialist_stubs: Optional custom specialist stub overrides.
        specialist_timeout_seconds: Per-specialist timeout in seconds.

    Returns:
        GraphState: The resulting state after complete workflow execution.
    """
    initial_state = create_initial_state(
        user_query=query,
        investor_profile=investor_profile,
        documents_available=documents_available,
    )
    graph = create_orchestration_graph(
        conversation_agent=conversation_agent,
        clarification_agent=clarification_agent,
        cio_agent=cio_agent,
        specialist_stubs=specialist_stubs,
        specialist_timeout_seconds=specialist_timeout_seconds,
    )
    result: GraphState = graph.invoke(initial_state)
    return result
