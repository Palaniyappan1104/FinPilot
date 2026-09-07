"""Unit tests for Phase 2.3 Shared Graph State."""

from typing import get_type_hints

import pytest

from app.agents.state import (
    ClarifiedRequest,
    GraphState,
    InvestorProfile,
    create_initial_state,
)

# ==============================================================================
# 1. State Initialization
# ==============================================================================


def test_create_initial_state_success():
    """Verify create_initial_state sets query and clean None defaults."""
    query = "Should I invest in Infosys for 5 years?"
    state = create_initial_state(user_query=query)

    assert state["user_query"] == query
    assert state["investor_profile"] is None
    assert state["clarified_request"] is None
    assert state["technical_result"] is None
    assert state["fundamental_result"] is None
    assert state["news_result"] is None
    assert state["research_result"] is None
    assert state["risk_result"] is None
    assert state["aggregated_result"] is None
    assert state["report"] is None


def test_create_initial_state_with_existing_profile():
    """Verify existing profile context can be provided at initial state creation."""
    query = "Analyze Apple"
    profile: InvestorProfile = {
        "ticker": "AAPL",
        "time_horizon": "long_term",
        "risk_tolerance": "moderate",
    }
    state = create_initial_state(user_query=query, investor_profile=profile)

    assert state["user_query"] == query
    assert state["investor_profile"] == profile
    assert state["investor_profile"]["ticker"] == "AAPL"


def test_create_initial_state_empty_query_raises_value_error():
    """Verify empty or whitespace query is rejected with ValueError."""
    with pytest.raises(ValueError) as exc_info:
        create_initial_state("")
    assert "user_query must be a non-empty string" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        create_initial_state("   ")
    assert "user_query must be a non-empty string" in str(exc_info.value)


# ==============================================================================
# 2. State Partial Updates & Isolation
# ==============================================================================


def test_state_partial_updates_preserve_unrelated_fields():
    """Verify updating one node's field does not alter or erase unrelated fields."""
    state = create_initial_state(user_query="Evaluate Reliance")

    # Simulate Conversation / Clarification Agent updating clarified_request
    clarified: ClarifiedRequest = {
        "normalized_query": "Evaluate Reliance Industries",
        "intent_type": "investment_analysis",
        "clarification_needed": False,
    }
    state["clarified_request"] = clarified

    assert state["user_query"] == "Evaluate Reliance"
    assert (
        state["clarified_request"]["normalized_query"] == "Evaluate Reliance Industries"
    )
    assert state["fundamental_result"] is None
    assert state["technical_result"] is None

    # Simulate Fundamental Analyst updating fundamental_result
    state["fundamental_result"] = {"pe_ratio": 24.5, "roe": 0.18}

    # Verify existing fields are completely preserved
    assert state["user_query"] == "Evaluate Reliance"
    assert state["clarified_request"] == clarified
    assert state["fundamental_result"] == {"pe_ratio": 24.5, "roe": 0.18}
    assert state["technical_result"] is None
    assert state["report"] is None


def test_state_aggregation_and_report_updates():
    """Verify aggregation and report fields can be populated sequentially."""
    state = create_initial_state(user_query="Analyze Tesla")

    # Simulate specialist outputs
    state["technical_result"] = {"trend": "bullish"}
    state["fundamental_result"] = {"valuation": "fair"}

    # Simulate Aggregator node update
    state["aggregated_result"] = {
        "consensus": "moderate_buy",
        "confidence": 0.85,
    }

    # Simulate Report Generator node update
    state["report"] = {
        "title": "FinPilot Research Report: Tesla",
        "summary": "Synthesized findings from specialists.",
    }

    assert state["technical_result"] == {"trend": "bullish"}
    assert state["fundamental_result"] == {"valuation": "fair"}
    assert state["aggregated_result"]["consensus"] == "moderate_buy"
    assert state["report"]["title"] == "FinPilot Research Report: Tesla"


# ==============================================================================
# 3. Schema & Type Hint Verification
# ==============================================================================


def test_graph_state_required_fields_and_annotations():
    """Verify GraphState annotations define all 10 required roadmap fields."""
    hints = get_type_hints(GraphState)

    expected_fields = [
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
    ]

    for field in expected_fields:
        assert field in hints, f"Field '{field}' missing from GraphState type hints."
