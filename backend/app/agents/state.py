"""Shared Graph State for FinPilot LangGraph workflows.

Phase 2.3 defines the typed workflow execution state that flows across LangGraph
nodes. Nodes read fields they need and return dictionary updates containing
only the fields they own.
"""

from typing import Any, Dict, List, Optional, TypedDict


class InvestorProfile(TypedDict, total=False):
    """Investor profile capturing user investment preferences and constraints.

    Provisional structure for Phase 2.3; detailed fields are elaborated in Phase 4.
    """

    target_company: Optional[str]
    ticker: Optional[str]
    investment_goal: Optional[str]
    time_horizon: Optional[str]
    capital_amount: Optional[float]
    risk_tolerance: Optional[str]
    profile_complete: Optional[bool]


class ClarifiedRequest(TypedDict, total=False):
    """Normalized and clarified request output.

    Provisional structure for Phase 2.3; elaborated in Phase 3 and Phase 4.
    """

    normalized_query: str
    intent_type: Optional[str]
    entities: Optional[Dict[str, Any]]
    clarification_needed: Optional[bool]
    clarification_questions: Optional[list[str]]


class CIORoutingDecisionState(TypedDict, total=False):
    """Structured state representation of the CIO routing decision (Phase 5)."""

    target_company: str
    ticker: Optional[str]
    selected_specialists: list[str]
    specialist_tasks: Dict[str, Any]
    reasoning: str
    fallback_applied: bool


class GraphState(TypedDict, total=False):
    """Shared execution state passed between LangGraph nodes in FinPilot workflows.

    Fields:
        user_query: Raw natural language prompt submitted by the user.
        investor_profile: Extracted/clarified investor constraints and profile.
        clarified_request: Normalized intent and structured entities from user query.
        technical_result: Structured findings from Technical Analyst (Phase 7+).
        fundamental_result: Structured findings from Fundamental Analyst (Phase 6+).
        news_result: Structured findings from News & Sentiment Analyst (Phase 8+).
        research_result: Structured findings from Document & SEC Analyst (Phase 9+).
        risk_result: Structured findings from Risk Assessment Analyst (Phase 10+).
        aggregated_result: Synthesis and multi-agent alignment result (Phase 11+).
        report: Final synthesized research report markdown/structure (Phase 12+).

    State Update Conventions:
        - Nodes receive the full GraphState as input.
        - Nodes MUST return a partial dictionary containing ONLY the keys they own.
        - Nodes MUST NOT mutate unrelated fields in the state.
        - LangGraph applies node return dictionaries to update the shared state.
    """

    # User Query & Session State
    user_query: str

    # Context & Request Normalization (Phase 3 & Phase 4)
    investor_profile: Optional[InvestorProfile]
    clarified_request: Optional[ClarifiedRequest]
    documents_available: Optional[bool]

    # CIO Orchestration & Routing (Phase 5)
    cio_decision: Optional[CIORoutingDecisionState]

    # Specialist Agent Outputs (Phase 6 - Phase 10)
    # Typed as Optional[Dict[str, Any]] placeholders until each specialist
    # Pydantic schema is defined in their respective phases.
    technical_result: Optional[Dict[str, Any]]
    fundamental_result: Optional[Dict[str, Any]]
    news_result: Optional[Dict[str, Any]]
    research_result: Optional[Dict[str, Any]]
    risk_result: Optional[Dict[str, Any]]

    # Aggregation & Synthesis (Phase 11 - Phase 12)
    aggregated_result: Optional[Dict[str, Any]]
    report: Optional[Dict[str, Any]]

    # Execution tracing & metadata (Phase 13)
    trace_id: Optional[str]

    # Target company & ticker resolution (Phase 13)
    target_company: Optional[str]
    ticker: Optional[str]

    # Specialist Input Feeds & Context (Phase 13)
    technical_metrics: Optional[Dict[str, Any]]
    fundamental_metrics: Optional[Dict[str, Any]]
    news_data: Optional[List[Dict[str, Any]]]
    research_context: Optional[Dict[str, Any]]
    research_query: Optional[str]

    # Clarification Feedback Loop (Phase 13.2.1)
    clarification_answers: Optional[Dict[str, Any]]

    # Workflow Status & Errors
    error: Optional[str]


def create_initial_state(
    user_query: str,
    investor_profile: Optional[InvestorProfile] = None,
    documents_available: bool = False,
    technical_metrics: Optional[Dict[str, Any]] = None,
    fundamental_metrics: Optional[Dict[str, Any]] = None,
    news_data: Optional[List[Dict[str, Any]]] = None,
    research_context: Optional[Dict[str, Any]] = None,
    clarification_answers: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    target_company: Optional[str] = None,
    ticker: Optional[str] = None,
) -> GraphState:
    """Initialize a clean GraphState with a user query.

    All downstream specialist, aggregation, and report fields start as None.

    Args:
        user_query: Raw natural language user query.
        investor_profile: Optional existing investor profile context.
        documents_available: Whether user uploaded documents are available.
        technical_metrics: Optional pre-loaded technical indicators data.
        fundamental_metrics: Optional pre-loaded fundamental financial metrics.
        news_data: Optional pre-loaded news articles.
        research_context: Optional pre-loaded document chunks / vault context.
        clarification_answers: Optional user answers to clarification questions.
        trace_id: Optional correlation identifier for structured logging.
        target_company: Optional explicit target company name.
        ticker: Optional explicit company ticker symbol.

    Returns:
        GraphState: An initial state dictionary ready for workflow execution.

    Raises:
        ValueError: If user_query is empty or whitespace-only.
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")

    return GraphState(
        user_query=user_query.strip(),
        investor_profile=investor_profile,
        clarified_request=None,
        documents_available=documents_available,
        cio_decision=None,
        technical_result=None,
        fundamental_result=None,
        news_result=None,
        research_result=None,
        risk_result=None,
        aggregated_result=None,
        report=None,
        trace_id=trace_id,
        target_company=target_company,
        ticker=ticker,
        technical_metrics=technical_metrics,
        fundamental_metrics=fundamental_metrics,
        news_data=news_data,
        research_context=research_context,
        clarification_answers=clarification_answers,
        error=None,
    )
