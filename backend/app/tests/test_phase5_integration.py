"""Integration tests for Phase 5.4: CIO Agent & LangGraph Orchestration.

Roadmap Requirements:
5.4.1 Unit tests: routing decisions across different query types (in test_cio.py).
5.4.2 Integration test: CIO output correctly drives LangGraph branching with stub
      specialist agents.
5.4.3 Failure-injection test: one specialist fails, graph still completes with partial
      results.

Additionally tests:
- Timeout isolation: timed-out specialist does not block others or crash graph.
- True concurrent fan-out: verify parallel execution with synchronization barrier.
- Selective routing: ensures only CIO-selected specialist branches are executed.
- Unselected specialists remain None in GraphState.
- Fan-in collections preserve all branch outputs cleanly.

All tests run 100% offline using deterministic mock providers and stubs.
ZERO real Gemini API or network calls are made.
"""

import json
import threading
import time
from typing import Any, Dict, List, Optional

from app.agents import (
    CIO_NODE_NAME,
    FAN_IN_NODE_NAME,
    CIOAgent,
    ClarificationAgent,
    ConversationAgent,
    GraphState,
    SpecialistName,
    create_initial_state,
    create_orchestration_graph,
    run_orchestration_graph,
)
from app.core.llm import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for Phase 5.4 integration testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_phase5_provider",
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
            model="mock-phase5-v1",
            provider=self.provider_name,
        )


def _build_mock_conversation_json(
    query: str = "Analyze TCS for long-term growth",
    company: str = "TCS",
    capital_amount: Optional[float] = 100000.0,
    time_horizon: Optional[str] = "5 years",
    risk_tolerance: Optional[str] = "moderate",
) -> str:
    """Helper constructing validated ConversationOutput JSON."""
    return json.dumps(
        {
            "normalized_query": query,
            "intent_type": "investment_analysis",
            "company": company,
            "capital_amount": capital_amount,
            "time_horizon": time_horizon,
            "risk_tolerance": risk_tolerance,
        }
    )


def _build_mock_cio_json(
    target_company: str = "TCS",
    ticker: Optional[str] = "TCS",
    selected_specialists: Optional[List[str]] = None,
    reasoning: str = "Selected based on user investment query.",
) -> str:
    """Helper constructing validated CIORoutingDecision JSON."""
    specs = selected_specialists or ["technical", "fundamental", "news", "risk"]
    tasks = {
        s: {
            "specialist": s,
            "task_description": f"Analyze {s} aspects for {target_company}.",
            "parameters": {"company": target_company, "ticker": ticker},
        }
        for s in specs
    }
    return json.dumps(
        {
            "target_company": target_company,
            "ticker": ticker,
            "selected_specialists": specs,
            "specialist_tasks": tasks,
            "reasoning": reasoning,
            "fallback_applied": False,
        }
    )


# ===========================================================================
# 1. Phase 5.4.2 Integration Test: Full Core Fan-Out & Fan-In Orchestration
# ===========================================================================


def test_5_4_2_cio_routing_drives_full_core_fan_out_and_fan_in():
    """Verify CIO output routes to all 4 core specialists with results collected."""
    conv_json = _build_mock_conversation_json()
    cio_json = _build_mock_cio_json(
        selected_specialists=["technical", "fundamental", "news", "risk"]
    )

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider(responses=[]))
    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[cio_json]))

    graph = create_orchestration_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        cio_agent=cio_agent,
    )

    initial_state = create_initial_state(
        user_query="Should I invest 100000 in TCS for 5 years with moderate risk?"
    )

    final_state: GraphState = graph.invoke(initial_state)

    # 1. Clarification & Profile completed
    assert final_state["investor_profile"]["profile_complete"] is True
    assert final_state["clarified_request"]["clarification_needed"] is False

    # 2. CIO decision correctly saved in GraphState
    cio_dec = final_state["cio_decision"]
    assert cio_dec is not None
    assert cio_dec["target_company"] == "TCS"
    assert cio_dec["ticker"] == "TCS"
    assert set(cio_dec["selected_specialists"]) == {
        "technical",
        "fundamental",
        "news",
        "risk",
    }

    # 3. All 4 selected specialists executed and produced stub results
    for spec_key in [
        "technical_result",
        "fundamental_result",
        "news_result",
        "risk_result",
    ]:
        res = final_state.get(spec_key)
        assert res is not None
        assert res["status"] == "completed"
        assert res["success"] is True
        assert res["is_stub"] is True
        assert res["data"]["company"] == "TCS"

    # 4. Unselected specialist (research) is absent
    assert final_state.get("research_result") is None


# ===========================================================================
# 2. Phase 5.4.2 Integration Test: Selective Routing (Subsets & Research)
# ===========================================================================


def test_5_4_2_selective_routing_technical_and_fundamental_only():
    """Verify CIO selecting only a subset executes strictly those branches."""
    conv_json = _build_mock_conversation_json()
    cio_json = _build_mock_cio_json(selected_specialists=["technical", "fundamental"])

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider(responses=[]))
    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[cio_json]))

    graph = create_orchestration_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        cio_agent=cio_agent,
    )

    initial_state = create_initial_state(
        user_query="Technical chart and valuation for TCS for 5 years"
    )

    final_state = graph.invoke(initial_state)

    assert final_state["technical_result"] is not None
    assert final_state["technical_result"]["status"] == "completed"

    assert final_state["fundamental_result"] is not None
    assert final_state["fundamental_result"]["status"] == "completed"

    # Unselected branches MUST remain None
    assert final_state.get("news_result") is None
    assert final_state.get("research_result") is None
    assert final_state.get("risk_result") is None


def test_5_4_2_selective_routing_with_documents_available():
    """Verify research analyst executes when documents_available is True."""
    conv_json = _build_mock_conversation_json()
    cio_json = _build_mock_cio_json(selected_specialists=["fundamental", "research"])

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider(responses=[]))
    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[cio_json]))

    graph = create_orchestration_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        cio_agent=cio_agent,
    )

    initial_state = create_initial_state(
        user_query="Analyze 10-K filings and valuation for TCS",
        documents_available=True,
    )

    final_state = graph.invoke(initial_state)

    assert final_state["fundamental_result"] is not None
    assert final_state["fundamental_result"]["status"] == "completed"

    assert final_state["research_result"] is not None
    assert final_state["research_result"]["status"] == "completed"

    # Unselected branches remain None
    assert final_state.get("technical_result") is None
    assert final_state.get("news_result") is None
    assert final_state.get("risk_result") is None


# ===========================================================================
# 3. Phase 5.4.3 Failure Injection Tests: Single & Multiple Specialist Errors
# ===========================================================================


def test_5_4_3_single_specialist_failure_isolation():
    """Verify one specialist failing does not crash graph; partial results saved."""
    conv_json = _build_mock_conversation_json()
    cio_json = _build_mock_cio_json(
        selected_specialists=["technical", "fundamental", "news"]
    )

    def failing_technical_stub(state: GraphState) -> Dict[str, Any]:
        raise RuntimeError("Simulated connection timeout to market data service")

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider(responses=[]))
    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[cio_json]))

    graph = create_orchestration_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        cio_agent=cio_agent,
        specialist_stubs={SpecialistName.TECHNICAL: failing_technical_stub},
    )

    initial_state = create_initial_state(user_query="Analyze TCS for 5 years")
    final_state = graph.invoke(initial_state)

    # 1. Technical specialist recorded failure cleanly
    tech_res = final_state.get("technical_result")
    assert tech_res is not None
    assert tech_res["status"] == "failed"
    assert tech_res["success"] is False
    assert "Simulated connection timeout" in tech_res["error"]

    # 2. Other selected specialists completed normally
    fund_res = final_state.get("fundamental_result")
    assert fund_res is not None
    assert fund_res["status"] == "completed"
    assert fund_res["success"] is True

    news_res = final_state.get("news_result")
    assert news_res is not None
    assert news_res["status"] == "completed"
    assert news_res["success"] is True

    # 3. Graph reached fan-in and completed without unhandled exception
    assert final_state.get("research_result") is None


def test_5_4_3_multiple_specialists_failing_partial_results():
    """Verify multiple specialist failures are contained while healthy ones succeed."""
    conv_json = _build_mock_conversation_json()
    cio_json = _build_mock_cio_json(
        selected_specialists=["technical", "fundamental", "risk"]
    )

    def failing_tech(state: GraphState) -> Dict[str, Any]:
        raise ValueError("Invalid chart parameters")

    def failing_risk(state: GraphState) -> Dict[str, Any]:
        raise KeyError("Risk matrix unavailable")

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider(responses=[]))
    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[cio_json]))

    graph = create_orchestration_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        cio_agent=cio_agent,
        specialist_stubs={
            SpecialistName.TECHNICAL: failing_tech,
            SpecialistName.RISK: failing_risk,
        },
    )

    initial_state = create_initial_state(user_query="Analyze TCS for 5 years")
    final_state = graph.invoke(initial_state)

    assert final_state["technical_result"]["status"] == "failed"
    assert final_state["risk_result"]["status"] == "failed"
    assert final_state["fundamental_result"]["status"] == "completed"
    assert final_state["fundamental_result"]["success"] is True


# ===========================================================================
# 4. Timeout Isolation Tests
# ===========================================================================


def test_specialist_timeout_isolation_unblocks_immediately():
    """Verify specialist timing out produces timeout result without stalling graph."""
    conv_json = _build_mock_conversation_json()
    cio_json = _build_mock_cio_json(selected_specialists=["technical", "fundamental"])

    def slow_technical_stub(state: GraphState) -> Dict[str, Any]:
        time.sleep(1.5)
        return {"technical_result": {"status": "completed"}}

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider(responses=[]))
    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[cio_json]))

    graph = create_orchestration_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        cio_agent=cio_agent,
        specialist_stubs={SpecialistName.TECHNICAL: slow_technical_stub},
        specialist_timeout_seconds=0.15,
    )

    initial_state = create_initial_state(user_query="Analyze TCS for 5 years")

    t0 = time.perf_counter()
    final_state = graph.invoke(initial_state)
    elapsed = time.perf_counter() - t0

    # 1. Unblocked around 0.15s, well before the 1.5s sleep
    assert elapsed < 0.7, f"Elapsed time {elapsed:.2f}s indicates blocking timeout."

    # 2. Timeout recorded
    tech_res = final_state.get("technical_result")
    assert tech_res is not None
    assert tech_res["status"] == "timeout"
    assert tech_res["success"] is False
    assert "timed out after 0.15 seconds" in tech_res["error"]

    # 3. Fundamental completed normally
    assert final_state["fundamental_result"]["status"] == "completed"
    assert final_state["fundamental_result"]["success"] is True


# ===========================================================================
# 5. Parallel Concurrency Verification (Synchronization Barrier)
# ===========================================================================


def test_parallel_fan_out_concurrency_barrier():
    """Verify specialists execute concurrently using threading synchronization."""
    conv_json = _build_mock_conversation_json()
    cio_json = _build_mock_cio_json(
        selected_specialists=["technical", "fundamental", "news"]
    )

    barrier = threading.Barrier(3, timeout=3.0)
    entered_specialists = []
    lock = threading.Lock()

    def sync_stub(spec_name: str):
        def _stub(state: GraphState) -> Dict[str, Any]:
            with lock:
                entered_specialists.append(spec_name)
            # All 3 threads must reach this barrier concurrently to proceed
            barrier.wait()
            return {
                f"{spec_name}_result": {
                    "specialist": spec_name,
                    "status": "completed",
                    "success": True,
                }
            }

        return _stub

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider(responses=[]))
    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[cio_json]))

    graph = create_orchestration_graph(
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        cio_agent=cio_agent,
        specialist_stubs={
            SpecialistName.TECHNICAL: sync_stub("technical"),
            SpecialistName.FUNDAMENTAL: sync_stub("fundamental"),
            SpecialistName.NEWS: sync_stub("news"),
        },
    )

    initial_state = create_initial_state(user_query="Analyze TCS for 5 years")
    final_state = graph.invoke(initial_state)

    # All 3 reached the barrier concurrently
    assert set(entered_specialists) == {"technical", "fundamental", "news"}
    assert final_state["technical_result"]["status"] == "completed"
    assert final_state["fundamental_result"]["status"] == "completed"
    assert final_state["news_result"]["status"] == "completed"


# ===========================================================================
# 6. Convenience Runner & Graph Topology Tests
# ===========================================================================


def test_run_orchestration_graph_helper_execution():
    """Verify run_orchestration_graph convenience helper end-to-end execution."""
    conv_json = _build_mock_conversation_json()
    cio_json = _build_mock_cio_json(selected_specialists=["technical", "risk"])

    conv_agent = ConversationAgent(provider=MockLLMProvider(responses=[conv_json]))
    clar_agent = ClarificationAgent(provider=MockLLMProvider(responses=[]))
    cio_agent = CIOAgent(provider=MockLLMProvider(responses=[cio_json]))

    final_state = run_orchestration_graph(
        query="Quick technical and risk check on TCS",
        conversation_agent=conv_agent,
        clarification_agent=clar_agent,
        cio_agent=cio_agent,
    )

    assert final_state["technical_result"] is not None
    assert final_state["risk_result"] is not None
    assert final_state["fundamental_result"] is None
    assert final_state["news_result"] is None
    assert final_state["research_result"] is None


def test_orchestration_graph_nodes_and_topology():
    """Verify create_orchestration_graph compiles with all expected nodes."""
    graph = create_orchestration_graph()
    nodes = graph.nodes

    assert CIO_NODE_NAME in nodes
    assert FAN_IN_NODE_NAME in nodes
    for spec in SpecialistName:
        assert spec.value in nodes
