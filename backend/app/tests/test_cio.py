"""Unit tests for Phase 5.1 and 5.2 CIO / Router Agent.

Roadmap Requirement 5.4.1: Routing decisions across different query types.
All tests run 100% offline using deterministic mock providers.
ZERO real Gemini API or network calls are made.
"""

import json
from typing import Any, List, Optional

from app.agents.cio import (
    CIOAgent,
    create_fallback_routing_decision,
)
from app.agents.cio_schema import (
    CORE_SPECIALISTS,
    CIOInput,
    CIORoutingDecision,
    SpecialistName,
)
from app.agents.state import (
    ClarifiedRequest,
    InvestorProfile,
)
from app.core.llm import (
    LLMAuthenticationError,
    LLMProvider,
    LLMResponse,
)


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for Phase 5 CIO unit testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_cio_provider",
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
            model="mock-cio-v1",
            provider=self.provider_name,
        )


# ===========================================================================
# Helper Fixtures
# ===========================================================================


def make_sample_input(
    query: str = "Should I invest in TCS for 5 years?",
    intent_type: str = "investment_analysis",
    company: str = "TCS",
    ticker: Optional[str] = "TCS",
    capital_amount: Optional[float] = 100000.0,
    time_horizon: Optional[str] = "5 years",
    risk_tolerance: Optional[str] = "moderate",
    documents_available: bool = False,
) -> CIOInput:
    """Construct a well-formed CIOInput for testing."""
    entities = {
        "company": company,
        "ticker": ticker,
        "capital_amount": capital_amount,
        "time_horizon": time_horizon,
        "risk_tolerance": risk_tolerance,
    }
    clarified_request: ClarifiedRequest = {
        "normalized_query": query,
        "intent_type": intent_type,
        "entities": entities,
        "clarification_needed": False,
        "clarification_questions": [],
    }
    investor_profile: InvestorProfile = {
        "target_company": company,
        "ticker": ticker,
        "capital_amount": capital_amount,
        "time_horizon": time_horizon,
        "risk_tolerance": risk_tolerance,
        "profile_complete": True,
    }
    return CIOInput(
        clarified_request=clarified_request,
        investor_profile=investor_profile,
        documents_available=documents_available,
    )


# ===========================================================================
# 1. Full Investment Query Routing
# ===========================================================================


def test_full_investment_query_routes_to_core_specialists():
    """Verify full investment query routes to technical, fundamental, news, risk."""
    mock_decision_json = json.dumps(
        {
            "target_company": "TCS",
            "ticker": "TCS",
            "selected_specialists": ["technical", "fundamental", "news", "risk"],
            "specialist_tasks": {
                "technical": {
                    "specialist": "technical",
                    "task_description": "Analyze technical indicators for TCS.",
                    "parameters": {"company": "TCS", "ticker": "TCS"},
                },
                "fundamental": {
                    "specialist": "fundamental",
                    "task_description": "Evaluate P/E and financial health for TCS.",
                    "parameters": {"company": "TCS", "ticker": "TCS"},
                },
                "news": {
                    "specialist": "news",
                    "task_description": "Assess media sentiment and earnings reports.",
                    "parameters": {"company": "TCS", "ticker": "TCS"},
                },
                "risk": {
                    "specialist": "risk",
                    "task_description": "Evaluate downside volatility for TCS.",
                    "parameters": {"company": "TCS", "ticker": "TCS"},
                },
            },
            "reasoning": "Full analysis requires all core analytical dimensions.",
            "fallback_applied": False,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[mock_decision_json]))
    inp = make_sample_input(
        query="Should I invest ₹1,00,000 in TCS for 5 years?",
        documents_available=False,
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    assert decision.target_company == "TCS"
    assert set(decision.selected_specialists) == set(CORE_SPECIALISTS)
    assert SpecialistName.RESEARCH not in decision.selected_specialists
    assert decision.fallback_applied is False
    assert len(decision.specialist_tasks) == 4


# ===========================================================================
# 2. Technical-Specific Query Routing
# ===========================================================================


def test_technical_specific_query_routes_to_technical_and_risk():
    """Verify technical indicator query routes to technical and risk specialists."""
    mock_decision_json = json.dumps(
        {
            "target_company": "Tata Motors",
            "ticker": "TATAMOTORS",
            "selected_specialists": ["technical", "risk"],
            "specialist_tasks": {
                "technical": {
                    "specialist": "technical",
                    "task_description": "Analyze RSI, MACD, and 50/200 DMA trend.",
                    "parameters": {"indicators": ["RSI", "MACD"]},
                },
                "risk": {
                    "specialist": "risk",
                    "task_description": "Assess support levels and downside stop loss.",
                    "parameters": {},
                },
            },
            "reasoning": "User specifically requested chart indicators and momentum.",
            "fallback_applied": False,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[mock_decision_json]))
    inp = make_sample_input(
        query="Show RSI, MACD and technical chart trend for Tata Motors",
        intent_type="technical_analysis",
        company="Tata Motors",
        ticker="TATAMOTORS",
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    assert decision.selected_specialists == [
        SpecialistName.TECHNICAL,
        SpecialistName.RISK,
    ]
    assert SpecialistName.FUNDAMENTAL not in decision.selected_specialists
    assert SpecialistName.NEWS not in decision.selected_specialists
    assert SpecialistName.RESEARCH not in decision.selected_specialists


# ===========================================================================
# 3. Fundamental-Specific Query Routing
# ===========================================================================


def test_fundamental_specific_query_routes_to_fundamental_and_risk():
    """Verify valuation and balance sheet query routes to fundamental and risk."""
    mock_decision_json = json.dumps(
        {
            "target_company": "Infosys",
            "ticker": "INFY",
            "selected_specialists": ["fundamental", "risk"],
            "specialist_tasks": {
                "fundamental": {
                    "specialist": "fundamental",
                    "task_description": "Analyze P/E, ROE, revenue growth, and debt.",
                    "parameters": {"company": "Infosys"},
                },
                "risk": {
                    "specialist": "risk",
                    "task_description": "Evaluate earnings stability and margin risk.",
                    "parameters": {"company": "Infosys"},
                },
            },
            "reasoning": "Query is focused on valuation and accounting fundamentals.",
            "fallback_applied": False,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[mock_decision_json]))
    inp = make_sample_input(
        query="What is the P/E ratio, balance sheet health, and valuation of Infosys?",
        intent_type="fundamental_analysis",
        company="Infosys",
        ticker="INFY",
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    assert decision.selected_specialists == [
        SpecialistName.FUNDAMENTAL,
        SpecialistName.RISK,
    ]
    assert SpecialistName.TECHNICAL not in decision.selected_specialists
    assert SpecialistName.NEWS not in decision.selected_specialists


# ===========================================================================
# 4. News / Sentiment-Specific Query Routing
# ===========================================================================


def test_news_specific_query_routes_to_news_and_risk():
    """Verify news sentiment query routes to news and risk specialists."""
    mock_decision_json = json.dumps(
        {
            "target_company": "Reliance Industries",
            "ticker": "RELIANCE",
            "selected_specialists": ["news", "risk"],
            "specialist_tasks": {
                "news": {
                    "specialist": "news",
                    "task_description": "Analyze latest news headlines and sentiment.",
                    "parameters": {"company": "Reliance Industries"},
                },
                "risk": {
                    "specialist": "risk",
                    "task_description": "Assess headline risk and news exposure.",
                    "parameters": {"company": "Reliance Industries"},
                },
            },
            "reasoning": "Query asks about recent market news and sentiment.",
            "fallback_applied": False,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[mock_decision_json]))
    inp = make_sample_input(
        query="What is the latest news and market sentiment on Reliance?",
        intent_type="news_sentiment",
        company="Reliance Industries",
        ticker="RELIANCE",
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    assert decision.selected_specialists == [
        SpecialistName.NEWS,
        SpecialistName.RISK,
    ]
    assert SpecialistName.TECHNICAL not in decision.selected_specialists
    assert SpecialistName.FUNDAMENTAL not in decision.selected_specialists


# ===========================================================================
# 5. Research Query With Documents Available
# ===========================================================================


def test_research_query_with_documents_selects_research_specialist():
    """Verify query with uploaded documents selects research specialist."""
    mock_decision_json = json.dumps(
        {
            "target_company": "TCS",
            "ticker": "TCS",
            "selected_specialists": ["research", "fundamental"],
            "specialist_tasks": {
                "research": {
                    "specialist": "research",
                    "task_description": "Extract guidance from 10-K report.",
                    "parameters": {"document_type": "10-K"},
                },
                "fundamental": {
                    "specialist": "fundamental",
                    "task_description": "Verify metrics against reported data.",
                    "parameters": {},
                },
            },
            "reasoning": "User uploaded annual report; research specialist needed.",
            "fallback_applied": False,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[mock_decision_json]))
    inp = make_sample_input(
        query="Summarize management guidance in the uploaded 10-K report for TCS.",
        intent_type="document_research",
        documents_available=True,
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    assert SpecialistName.RESEARCH in decision.selected_specialists
    assert SpecialistName.FUNDAMENTAL in decision.selected_specialists


# ===========================================================================
# 6. Research Query Without Documents Available
# ===========================================================================


def test_research_query_without_documents_omits_research_analyst():
    """Verify research analyst is omitted when documents_available=False."""
    # Mock LLM outputs fundamental and news instead since no documents are available
    mock_decision_json = json.dumps(
        {
            "target_company": "TCS",
            "ticker": "TCS",
            "selected_specialists": ["fundamental", "news"],
            "specialist_tasks": {
                "fundamental": {
                    "specialist": "fundamental",
                    "task_description": "Analyze available public filing metrics.",
                    "parameters": {},
                },
                "news": {
                    "specialist": "news",
                    "task_description": "Check public press releases for guidance.",
                    "parameters": {},
                },
            },
            "reasoning": "Documents are unavailable, so Research Analyst was skipped.",
            "fallback_applied": False,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[mock_decision_json]))
    inp = make_sample_input(
        query="What does the 10-K say about TCS?",
        intent_type="document_research",
        documents_available=False,
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    assert SpecialistName.RESEARCH not in decision.selected_specialists


# ===========================================================================
# 7. Uncertain Query Defaults to Fallback
# ===========================================================================


def test_uncertain_query_routes_to_all_core_specialists():
    """Verify uncertain or broad intent triggers core specialist routing."""
    mock_decision_json = json.dumps(
        {
            "target_company": "Infosys",
            "ticker": "INFY",
            "selected_specialists": ["technical", "fundamental", "news", "risk"],
            "specialist_tasks": {
                "technical": {
                    "specialist": "technical",
                    "task_description": "Analyze chart.",
                },
                "fundamental": {
                    "specialist": "fundamental",
                    "task_description": "Analyze balance sheet.",
                },
                "news": {"specialist": "news", "task_description": "Analyze news."},
                "risk": {"specialist": "risk", "task_description": "Analyze risk."},
            },
            "reasoning": "Uncertain user intent; defaulting to all 4 core specialists.",
            "fallback_applied": True,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[mock_decision_json]))
    inp = make_sample_input(
        query="Give me an overview of Infosys.",
        intent_type="general_inquiry",
        company="Infosys",
        ticker="INFY",
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    assert set(decision.selected_specialists) == set(CORE_SPECIALISTS)
    assert decision.fallback_applied is True


# ===========================================================================
# 8. LLM Provider Failure Triggers Deterministic Fallback
# ===========================================================================


def test_provider_failure_triggers_deterministic_fallback():
    """Verify provider exception triggers fallback to core specialists."""
    failing_provider = MockLLMProvider(
        fail_with=LLMAuthenticationError("Simulated LLM authentication error")
    )
    cio_agent = CIOAgent(provider=failing_provider)

    inp = make_sample_input(
        query="Should I invest in TCS for 5 years?",
        company="TCS",
        ticker="TCS",
        capital_amount=250000.0,
        time_horizon="3 years",
        risk_tolerance="conservative",
        documents_available=False,
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    # Fallback applied is True
    assert decision.fallback_applied is True
    assert decision.target_company == "TCS"
    assert decision.ticker == "TCS"
    assert set(decision.selected_specialists) == set(CORE_SPECIALISTS)
    assert SpecialistName.RESEARCH not in decision.selected_specialists

    # All 4 core tasks are generated deterministically
    assert len(decision.specialist_tasks) == 4
    tech_task = decision.specialist_tasks[SpecialistName.TECHNICAL]
    assert tech_task.parameters["capital_amount"] == 250000.0
    assert tech_task.parameters["time_horizon"] == "3 years"
    assert tech_task.parameters["risk_tolerance"] == "conservative"


# ===========================================================================
# 9. Invalid Structured Output Triggers Deterministic Fallback
# ===========================================================================


def test_invalid_structured_output_triggers_deterministic_fallback():
    """Verify malformed JSON triggering retry exhaustion falls back cleanly."""
    malformed_provider = MockLLMProvider(
        responses=["This is not JSON at all!", "{invalid_json: 123}"]
    )
    cio_agent = CIOAgent(provider=malformed_provider)

    inp = make_sample_input(
        query="Analyze HDFC Bank for me.",
        company="HDFC Bank",
        ticker="HDFCBANK",
        documents_available=True,
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    assert decision.fallback_applied is True
    assert decision.target_company == "HDFC Bank"
    # When documents_available=True, research is included in fallback
    assert SpecialistName.RESEARCH in decision.selected_specialists
    assert set(CORE_SPECIALISTS).issubset(set(decision.selected_specialists))


# ===========================================================================
# 10. Deduplication of Selected Specialists
# ===========================================================================


def test_specialist_deduplication_in_schema():
    """Verify selected_specialists validator strips duplicate entries."""
    decision = CIORoutingDecision(
        target_company="Wipro",
        selected_specialists=[
            SpecialistName.TECHNICAL,
            SpecialistName.FUNDAMENTAL,
            SpecialistName.TECHNICAL,  # Duplicate
            SpecialistName.RISK,
            SpecialistName.FUNDAMENTAL,  # Duplicate
        ],
        specialist_tasks={},
        reasoning="Testing deduplication of specialists.",
    )

    assert decision.selected_specialists == [
        SpecialistName.TECHNICAL,
        SpecialistName.FUNDAMENTAL,
        SpecialistName.RISK,
    ]
    # Tasks auto-generated for deduplicated specialists
    assert len(decision.specialist_tasks) == 3


# ===========================================================================
# 11. Enforcement: Research Never Selected When Documents Unavailable
# ===========================================================================


def test_research_enforcement_filtered_if_llm_violates_rule():
    """Verify CIOAgent strips research if LLM outputs it without documents."""
    # LLM improperly included 'research' even though documents_available is False
    violating_decision_json = json.dumps(
        {
            "target_company": "Bharti Airtel",
            "ticker": "AIRTEL",
            "selected_specialists": ["fundamental", "research", "risk"],
            "specialist_tasks": {
                "fundamental": {
                    "specialist": "fundamental",
                    "task_description": "Analyze ROE.",
                },
                "research": {
                    "specialist": "research",
                    "task_description": "Analyze 10-K.",
                },
                "risk": {"specialist": "risk", "task_description": "Assess risk."},
            },
            "reasoning": "Selected research despite no documents.",
            "fallback_applied": False,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[violating_decision_json]))
    inp = make_sample_input(
        query="Analyze Bharti Airtel.",
        company="Bharti Airtel",
        ticker="AIRTEL",
        documents_available=False,
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    # Research MUST be filtered out
    assert SpecialistName.RESEARCH not in decision.selected_specialists
    assert SpecialistName.RESEARCH not in decision.specialist_tasks
    assert SpecialistName.FUNDAMENTAL in decision.selected_specialists
    assert SpecialistName.RISK in decision.selected_specialists
    assert "Research Analyst omitted" in decision.reasoning


# ===========================================================================
# 12. Correct Task Parameters Propagated From State
# ===========================================================================


def test_task_parameters_propagated_from_state():
    """Verify parameters from clarified_request and profile propagate into tasks."""
    mock_decision_json = json.dumps(
        {
            "target_company": "Reliance",
            "ticker": "RELIANCE",
            "selected_specialists": ["technical", "risk"],
            "specialist_tasks": {
                "technical": {
                    "specialist": "technical",
                    "task_description": "Analyze technical setup.",
                    "parameters": {},  # Agent should populate missing state parameters
                },
                "risk": {
                    "specialist": "risk",
                    "task_description": "Analyze downside.",
                    "parameters": {},
                },
            },
            "reasoning": "Technical swing setup requested.",
            "fallback_applied": False,
        }
    )

    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[mock_decision_json]))
    inp = make_sample_input(
        query="Technical outlook for Reliance for 6 months with 50k",
        company="Reliance",
        ticker="RELIANCE",
        capital_amount=50000.0,
        time_horizon="6 months",
        risk_tolerance="high",
    )

    res = cio_agent.run(inp)
    assert res.success is True
    decision: CIORoutingDecision = res.data

    for spec in [SpecialistName.TECHNICAL, SpecialistName.RISK]:
        task = decision.specialist_tasks[spec]
        assert task.parameters["company"] == "Reliance"
        assert task.parameters["ticker"] == "RELIANCE"
        assert task.parameters["capital_amount"] == 50000.0
        assert task.parameters["time_horizon"] == "6 months"
        assert task.parameters["risk_tolerance"] == "high"


# ===========================================================================
# 13. State Conversion and GraphState Compatibility
# ===========================================================================


def test_cio_decision_to_state_conversion():
    """Verify to_state() produces valid CIORoutingDecisionState for GraphState."""
    decision = create_fallback_routing_decision(
        target_company="TCS",
        ticker="TCS",
        documents_available=False,
        time_horizon="5 years",
        capital_amount=100000.0,
        risk_tolerance="moderate",
    )

    state_dict = decision.to_state()

    assert state_dict["target_company"] == "TCS"
    assert state_dict["ticker"] == "TCS"
    assert state_dict["fallback_applied"] is True
    assert set(state_dict["selected_specialists"]) == {
        "technical",
        "fundamental",
        "news",
        "risk",
    }
    assert isinstance(state_dict["specialist_tasks"], dict)
    assert "technical" in state_dict["specialist_tasks"]
    assert (
        state_dict["specialist_tasks"]["technical"]["parameters"]["capital_amount"]
        == 100000.0
    )
