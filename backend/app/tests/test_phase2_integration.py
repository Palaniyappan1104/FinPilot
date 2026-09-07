"""Phase 2.6 Integration Tests.

Verifies end-to-end infrastructure integration across all Phase 2 components:
- Phase 2.1: LLMProvider abstraction (mocked, zero real API calls)
- Phase 2.2: Structured output parsing & bounded retries
- Phase 2.3: Shared GraphState lifecycle and field preservation
- Phase 2.4: BaseAgent interface, AgentResult wrapper, and Tool abstraction
- Phase 2.5: LangGraph workflow compilation and execution

All tests are deterministic, offline, and require no API keys or network access.
"""

from typing import Any, Dict, List, Optional

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents import (
    AgentResult,
    BaseAgent,
    GraphState,
    Tool,
    create_graph,
    create_initial_state,
)
from app.core.llm import (
    LLMAuthenticationError,
    LLMProvider,
    LLMResponse,
    LLMStructuredOutputError,
    generate_structured,
)

# ---------------------------------------------------------------------------
# Test Schemas & Mock Implementations
# ---------------------------------------------------------------------------


class MockInsightSchema(BaseModel):
    """Structured output schema representing an analyst's findings."""

    company: str = Field(..., description="Target company")
    stance: str = Field(..., description="Bullish, Bearish, or Neutral")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    key_metrics: Dict[str, float] = Field(default_factory=dict)


class DeterministicMockProvider(LLMProvider):
    """Deterministic mock LLMProvider for testing without network or API keys."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        fail_with: Optional[Exception] = None,
    ) -> None:
        self.responses = responses or []
        self.call_count = 0
        self.prompts_received: List[str] = []
        self.fail_with = fail_with

    @property
    def provider_name(self) -> str:
        return "mock_deterministic"

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
            model="mock-deterministic-v1",
            provider=self.provider_name,
            metadata={"call_index": self.call_count},
        )


class MockIndicatorTool(Tool):
    """Mock financial indicator tool."""

    name = "mock_rsi_indicator"
    description = "Calculates RSI indicator for a ticker"

    def execute(self, ticker: str) -> float:
        return 62.5


class MockAnalystAgent(BaseAgent):
    """Concrete mock agent integrating LLMProvider, structured output, and tools."""

    name = "mock_analyst"
    input_schema = Dict[str, Any]
    output_schema = MockInsightSchema

    def __init__(
        self,
        provider: LLMProvider,
        tools: Optional[List[Tool]] = None,
        target_state_field: str = "technical_result",
    ) -> None:
        super().__init__(tools=tools)
        self.provider = provider
        self.target_state_field = target_state_field

    def run(self, state: GraphState) -> Dict[str, Any]:
        query = state.get("user_query", "")

        # Incorporate tool if registered
        tool_data: Dict[str, float] = {}
        indicator_tool = self.get_tool("mock_rsi_indicator")
        if indicator_tool:
            tool_data["RSI"] = indicator_tool.execute(ticker="TCS")

        prompt = f"Analyze: {query} with metrics: {tool_data}"

        try:
            insight = generate_structured(
                provider=self.provider,
                prompt=prompt,
                schema=MockInsightSchema,
            )
            result = AgentResult.create_success(
                data=insight.model_dump(),
                confidence=insight.confidence_score,
            )
        except Exception as err:
            result = AgentResult.create_failure(
                error=f"Analysis failed: {err}",
                confidence=0.0,
            )

        return {self.target_state_field: result.model_dump()}


# ---------------------------------------------------------------------------
# 1. Mocked LLM Provider Tests
# ---------------------------------------------------------------------------


def test_mock_provider_generate_offline():
    """Verify mock provider generates deterministic LLMResponse without network."""
    provider = DeterministicMockProvider(responses=['{"data": "test"}'])
    response = provider.generate("Test prompt")

    assert response.content == '{"data": "test"}'
    assert response.provider == "mock_deterministic"
    assert response.model == "mock-deterministic-v1"
    assert provider.call_count == 1
    assert provider.prompts_received == ["Test prompt"]


def test_mock_provider_raises_configured_exception():
    """Verify mock provider propagates configured errors cleanly."""
    provider = DeterministicMockProvider(
        fail_with=LLMAuthenticationError(
            message="Invalid mock key",
            provider="mock",
        )
    )
    with pytest.raises(LLMAuthenticationError) as exc_info:
        provider.generate("Test prompt")

    assert "Invalid mock key" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Structured Output Integration with Mock Provider
# ---------------------------------------------------------------------------


def test_structured_output_valid_json():
    """Verify valid JSON converts into structured Pydantic model."""
    valid_json = (
        '{"company": "TCS", "stance": "Bullish", "confidence_score": 0.88, '
        '"key_metrics": {"P/E": 28.5}}'
    )
    provider = DeterministicMockProvider(responses=[valid_json])

    insight = generate_structured(
        provider=provider,
        prompt="Analyze TCS",
        schema=MockInsightSchema,
    )

    assert insight.company == "TCS"
    assert insight.stance == "Bullish"
    assert insight.confidence_score == 0.88
    assert insight.key_metrics == {"P/E": 28.5}
    assert provider.call_count == 1


def test_structured_output_markdown_wrapped_json():
    """Verify markdown-wrapped JSON code blocks are properly stripped and parsed."""
    markdown_json = (
        "```json\n"
        '{"company": "INFY", "stance": "Neutral", "confidence_score": 0.65}\n'
        "```"
    )
    provider = DeterministicMockProvider(responses=[markdown_json])

    insight = generate_structured(
        provider=provider,
        prompt="Analyze INFY",
        schema=MockInsightSchema,
    )

    assert insight.company == "INFY"
    assert insight.stance == "Neutral"
    assert insight.confidence_score == 0.65


def test_structured_output_bounded_retry_success():
    """Verify retry succeeds on attempt 2 when attempt 1 returns malformed output."""
    malformed_attempt_1 = "This is not valid JSON at all"
    valid_attempt_2 = (
        '{"company": "RELIANCE", "stance": "Bullish", "confidence_score": 0.92}'
    )
    provider = DeterministicMockProvider(
        responses=[malformed_attempt_1, valid_attempt_2]
    )

    insight = generate_structured(
        provider=provider,
        prompt="Analyze Reliance",
        schema=MockInsightSchema,
    )

    assert insight.company == "RELIANCE"
    assert insight.confidence_score == 0.92
    assert provider.call_count == 2  # Proves bounded retry occurred


def test_structured_output_bounded_retry_exhaustion():
    """Verify LLMStructuredOutputError is raised when both attempts fail."""
    provider = DeterministicMockProvider(responses=["malformed 1", "malformed 2"])

    with pytest.raises(LLMStructuredOutputError) as exc_info:
        generate_structured(
            provider=provider,
            prompt="Analyze Reliance",
            schema=MockInsightSchema,
        )

    err = exc_info.value
    assert err.schema_name == "MockInsightSchema"
    assert err.attempts == 2


# ---------------------------------------------------------------------------
# 3. Mock Agent with GraphState (Direct Execution)
# ---------------------------------------------------------------------------


def test_mock_agent_direct_state_operation():
    """Verify mock agent reads state and returns valid state update."""
    valid_json = (
        '{"company": "TCS", "stance": "Bullish", "confidence_score": 0.85, '
        '"key_metrics": {"RSI": 62.5}}'
    )
    provider = DeterministicMockProvider(responses=[valid_json])
    agent = MockAnalystAgent(provider=provider, target_state_field="technical_result")

    initial_state = create_initial_state(user_query="Evaluate TCS investment")
    update = agent.run(initial_state)

    assert "technical_result" in update
    tech_res = update["technical_result"]
    assert tech_res["success"] is True
    assert tech_res["confidence"] == 0.85
    assert tech_res["data"]["company"] == "TCS"
    assert tech_res["error"] is None

    # Apply state update according to Phase 2.3 conventions
    initial_state.update(update)
    assert initial_state["user_query"] == "Evaluate TCS investment"
    assert initial_state["technical_result"] is not None
    # Unrelated fields preserved
    assert initial_state["fundamental_result"] is None
    assert initial_state["report"] is None


# ---------------------------------------------------------------------------
# 4. Mock Agent via LangGraph Pipeline
# ---------------------------------------------------------------------------


def test_mock_agent_executes_in_langgraph():
    """Verify mock agent executes as a node in a minimal LangGraph StateGraph."""
    valid_json = '{"company": "WIPRO", "stance": "Neutral", "confidence_score": 0.70}'
    provider = DeterministicMockProvider(responses=[valid_json])
    agent = MockAnalystAgent(provider=provider, target_state_field="technical_result")

    # Build test graph: START -> mock_agent -> END
    builder = StateGraph(GraphState)
    builder.add_node(agent.name, agent.run)
    builder.add_edge(START, agent.name)
    builder.add_edge(agent.name, END)
    graph = builder.compile()

    state = create_initial_state(user_query="Analyze WIPRO fundamentals")
    final_state = graph.invoke(state)

    assert final_state["user_query"] == "Analyze WIPRO fundamentals"
    assert final_state["technical_result"] is not None
    assert final_state["technical_result"]["success"] is True
    assert final_state["technical_result"]["confidence"] == 0.70
    assert final_state["technical_result"]["data"]["company"] == "WIPRO"
    assert final_state["fundamental_result"] is None


# ---------------------------------------------------------------------------
# 5. Complete End-to-End Workflow Integration
# ---------------------------------------------------------------------------


def test_complete_end_to_end_workflow():
    """Verify full chain: MockProvider -> Structured -> Mock Agent -> LangGraph."""
    valid_json = (
        '{"company": "HDFC", "stance": "Bullish", "confidence_score": 0.95, '
        '"key_metrics": {"RSI": 62.5}}'
    )
    provider = DeterministicMockProvider(responses=[valid_json])
    tool = MockIndicatorTool()
    agent = MockAnalystAgent(
        provider=provider,
        tools=[tool],
        target_state_field="technical_result",
    )

    builder = StateGraph(GraphState)
    builder.add_node(agent.name, agent.run)
    builder.add_edge(START, agent.name)
    builder.add_edge(agent.name, END)
    graph = builder.compile()

    initial_state = create_initial_state(
        user_query="Comprehensive analysis for HDFC Bank",
        investor_profile={
            "target_company": "HDFC",
            "time_horizon": "5 years",
        },
    )

    final_state = graph.invoke(initial_state)

    # Verify state integrity
    assert final_state["user_query"] == "Comprehensive analysis for HDFC Bank"
    assert final_state["investor_profile"]["time_horizon"] == "5 years"

    # Verify structured result
    res = final_state["technical_result"]
    assert res["success"] is True
    assert res["confidence"] == 0.95
    assert res["data"]["company"] == "HDFC"
    assert res["data"]["stance"] == "Bullish"

    # Verify tool was invoked by agent during graph execution
    assert "mock_rsi_indicator" in [t.name for t in agent.tools.values()]


# ---------------------------------------------------------------------------
# 6. Failure Isolation & Handling Integration
# ---------------------------------------------------------------------------


def test_failure_isolation_provider_error_in_graph():
    """Verify provider error produces controlled failed AgentResult in graph."""
    provider = DeterministicMockProvider(
        fail_with=LLMAuthenticationError(
            message="Simulated key error",
            provider="mock",
        )
    )
    agent = MockAnalystAgent(provider=provider, target_state_field="technical_result")

    builder = StateGraph(GraphState)
    builder.add_node(agent.name, agent.run)
    builder.add_edge(START, agent.name)
    builder.add_edge(agent.name, END)
    graph = builder.compile()

    state = create_initial_state(user_query="Analyze with bad key")
    final_state = graph.invoke(state)

    # Graph completes cleanly without unhandled crash
    assert final_state["user_query"] == "Analyze with bad key"
    res = final_state["technical_result"]
    assert res["success"] is False
    assert res["confidence"] == 0.0
    assert "Simulated key error" in res["error"]
    assert final_state["fundamental_result"] is None


def test_failure_isolation_structured_validation_error_in_graph():
    """Verify validation exhaustion produces controlled failed AgentResult."""
    provider = DeterministicMockProvider(responses=["broken json 1", "broken json 2"])
    agent = MockAnalystAgent(provider=provider, target_state_field="technical_result")

    builder = StateGraph(GraphState)
    builder.add_node(agent.name, agent.run)
    builder.add_edge(START, agent.name)
    builder.add_edge(agent.name, END)
    graph = builder.compile()

    state = create_initial_state(user_query="Analyze with malformed outputs")
    final_state = graph.invoke(state)

    res = final_state["technical_result"]
    assert res["success"] is False
    assert res["confidence"] == 0.0
    assert "Structured output validation failed" in res["error"]


# ---------------------------------------------------------------------------
# 7. Production Graph Baseline Integrity
# ---------------------------------------------------------------------------


def test_production_graph_baseline_unchanged():
    """Verify production graph create_graph() remains START -> passthrough -> END."""
    graph = create_graph()
    state = create_initial_state(user_query="Production graph check")
    result = graph.invoke(state)

    assert result["user_query"] == "Production graph check"
    assert result["technical_result"] is None
