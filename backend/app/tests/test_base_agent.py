"""Unit tests for BaseAgent, AgentResult, and Tool abstractions.

Phase 2.4 tests:
- AgentResult success/failure states and dictionary conversion
- Confidence validation and boundary handling
- BaseAgent and Tool abstract behavior and interface enforcement
- Mock agent and mock tool implementations
- Agent/Tool integration (tool lookup, execution, error wrapping)
"""

from typing import Any, Dict

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.agents import (
    Agent,
    AgentResult,
    BaseAgent,
    BaseTool,
    GraphState,
    Tool,
    create_initial_state,
)

# ---------------------------------------------------------------------------
# Test Helpers / Mock Schemas
# ---------------------------------------------------------------------------


class MockInputSchema(BaseModel):
    query: str = Field(..., description="Test query")


class MockOutputSchema(BaseModel):
    score: float = Field(..., description="Test output score")


class MockCalculatorTool(Tool):
    """Mock tool that performs simple addition."""

    name = "mock_calculator"
    description = "Adds two numbers together"

    def execute(self, a: int, b: int) -> int:
        return a + b


class MockFailingTool(Tool):
    """Mock tool that simulates an external tool failure."""

    name = "mock_failing_tool"
    description = "Simulates an external API error"

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("External service unavailable")


class ConcreteMockAgent(BaseAgent):
    """Concrete agent implementing the BaseAgent interface."""

    name = "mock_analyst"
    input_schema = MockInputSchema
    output_schema = MockOutputSchema

    def run(self, state: GraphState) -> Dict[str, Any]:
        result = AgentResult.create_success(
            data={"score": 0.95, "query": state["user_query"]},
            confidence=0.95,
        )
        return {"technical_result": result.model_dump()}


class ToolIntegratedAgent(BaseAgent):
    """Concrete agent that executes registered tools and produces AgentResult."""

    name = "tool_integrated_agent"
    input_schema = MockInputSchema
    output_schema = MockOutputSchema

    def run(self, state: GraphState) -> Dict[str, Any]:
        calc = self.get_tool("mock_calculator")
        if not calc:
            result = AgentResult.create_failure("Calculator tool not configured")
            return {"technical_result": result.model_dump()}

        try:
            val = calc.execute(10, 20)
            result = AgentResult.create_success(
                data={"calc_result": val},
                confidence=0.99,
            )
        except Exception as e:
            result = AgentResult.create_failure(str(e))

        return {"technical_result": result.model_dump()}


# ---------------------------------------------------------------------------
# 1. AgentResult Tests
# ---------------------------------------------------------------------------


def test_agent_result_success_instantiation():
    """Verify successful AgentResult initialization and attributes."""
    result = AgentResult(
        success=True,
        data={"metric": 42},
        confidence=0.88,
    )
    assert result.success is True
    assert result.data == {"metric": 42}
    assert result.error is None
    assert result.confidence == 0.88


def test_agent_result_success_factory():
    """Verify AgentResult.create_success factory helper."""
    result = AgentResult.create_success(data="completed", confidence=1.0)
    assert result.success is True
    assert result.data == "completed"
    assert result.error is None
    assert result.confidence == 1.0


def test_agent_result_failure_instantiation():
    """Verify failed AgentResult initialization and attributes."""
    result = AgentResult(
        success=False,
        error="LLM quota exceeded",
        confidence=0.0,
    )
    assert result.success is False
    assert result.data is None
    assert result.error == "LLM quota exceeded"
    assert result.confidence == 0.0


def test_agent_result_failure_factory():
    """Verify AgentResult.create_failure factory helper."""
    result = AgentResult.create_failure(error="API timeout")
    assert result.success is False
    assert result.data is None
    assert result.error == "API timeout"
    assert result.confidence is None


def test_agent_result_to_state_update():
    """Verify to_state_update creates a valid dict for GraphState merging."""
    result = AgentResult.create_success(data={"rsi": 65.4}, confidence=0.85)
    update = result.to_state_update("technical_result")

    assert "technical_result" in update
    assert update["technical_result"]["success"] is True
    assert update["technical_result"]["data"] == {"rsi": 65.4}
    assert update["technical_result"]["confidence"] == 0.85
    assert update["technical_result"]["error"] is None


def test_agent_result_confidence_validation():
    """Verify confidence accepts valid bounds [0.0, 1.0] and rejects out-of-bounds."""
    # Valid values
    assert AgentResult(success=True, confidence=0.0).confidence == 0.0
    assert AgentResult(success=True, confidence=0.5).confidence == 0.5
    assert AgentResult(success=True, confidence=1.0).confidence == 1.0
    assert AgentResult(success=True, confidence=None).confidence is None

    # Invalid: below 0.0
    with pytest.raises(ValidationError):
        AgentResult(success=True, confidence=-0.01)

    # Invalid: above 1.0
    with pytest.raises(ValidationError):
        AgentResult(success=True, confidence=1.01)


def test_agent_result_arbitrary_types_allowed():
    """Verify arbitrary Python objects can be passed in data."""

    class CustomData:
        def __init__(self, val: int) -> None:
            self.val = val

    custom_obj = CustomData(99)
    result = AgentResult(success=True, data=custom_obj, confidence=0.7)
    assert result.data.val == 99


# ---------------------------------------------------------------------------
# 2. Tool Abstraction Tests
# ---------------------------------------------------------------------------


def test_tool_abstract_behavior():
    """Verify Tool and BaseTool cannot be instantiated directly or incompletely."""
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        BaseTool()  # type: ignore[abstract]

    # Incomplete tool missing execute
    class IncompleteToolNoExecute(Tool):
        name = "incomplete"
        description = "missing execute"

    with pytest.raises(TypeError):
        IncompleteToolNoExecute()  # type: ignore[abstract]

    # Incomplete tool missing description
    class IncompleteToolNoDescription(Tool):
        name = "incomplete"

        def execute(self, *args: Any, **kwargs: Any) -> Any:
            return None

    with pytest.raises(TypeError):
        IncompleteToolNoDescription()  # type: ignore[abstract]

    # Incomplete tool missing name
    class IncompleteToolNoName(Tool):
        description = "missing name"

        def execute(self, *args: Any, **kwargs: Any) -> Any:
            return None

    with pytest.raises(TypeError):
        IncompleteToolNoName()  # type: ignore[abstract]


def test_concrete_mock_tool_execution():
    """Verify mock tool executes correctly and supports __call__."""
    tool = MockCalculatorTool()
    assert tool.name == "mock_calculator"
    assert "Adds two numbers" in tool.description

    # Direct execute
    assert tool.execute(a=15, b=25) == 40
    # Callable protocol
    assert tool(a=15, b=25) == 40


# ---------------------------------------------------------------------------
# 3. Base Agent Tests
# ---------------------------------------------------------------------------


def test_base_agent_abstract_behavior():
    """Verify BaseAgent and Agent cannot be instantiated directly or incompletely."""
    with pytest.raises(TypeError):
        BaseAgent()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]

    # Missing run method
    class IncompleteAgentNoRun(BaseAgent):
        name = "no_run"
        input_schema = MockInputSchema
        output_schema = MockOutputSchema

    with pytest.raises(TypeError):
        IncompleteAgentNoRun()  # type: ignore[abstract]

    # Missing name
    class IncompleteAgentNoName(BaseAgent):
        input_schema = MockInputSchema
        output_schema = MockOutputSchema

        def run(self, state: GraphState) -> Dict[str, Any]:
            return {}

    with pytest.raises(TypeError):
        IncompleteAgentNoName()  # type: ignore[abstract]


def test_concrete_mock_agent_run():
    """Verify a concrete mock agent implements the interface and updates GraphState."""
    agent = ConcreteMockAgent()
    assert agent.name == "mock_analyst"
    assert agent.input_schema == MockInputSchema
    assert agent.output_schema == MockOutputSchema

    state = create_initial_state(user_query="Analyze TCS fundamentals")
    update = agent.run(state)

    assert "technical_result" in update
    assert update["technical_result"]["success"] is True
    assert update["technical_result"]["confidence"] == 0.95
    assert update["technical_result"]["data"]["query"] == "Analyze TCS fundamentals"

    # Merge into state according to Phase 2.3 conventions
    state.update(update)
    assert state["technical_result"] is not None
    assert state["technical_result"]["data"]["score"] == 0.95
    # Unrelated fields remain untouched
    assert state["fundamental_result"] is None


# ---------------------------------------------------------------------------
# 4. Agent / Tool Integration Tests
# ---------------------------------------------------------------------------


def test_agent_tool_registration_and_lookup():
    """Verify agent correctly registers tools and supports lookup."""
    tool1 = MockCalculatorTool()
    tool2 = MockFailingTool()

    agent = ToolIntegratedAgent(tools=[tool1, tool2])
    assert len(agent.tools) == 2
    assert agent.get_tool("mock_calculator") is tool1
    assert agent.get_tool("mock_failing_tool") is tool2
    assert agent.get_tool("nonexistent_tool") is None


def test_agent_tool_integration_success():
    """Verify agent uses registered tool during execution and produces valid result."""
    calc_tool = MockCalculatorTool()
    agent = ToolIntegratedAgent(tools=[calc_tool])

    state = create_initial_state(user_query="Compute indicators")
    update = agent.run(state)

    assert update["technical_result"]["success"] is True
    assert update["technical_result"]["data"] == {"calc_result": 30}
    assert update["technical_result"]["confidence"] == 0.99


def test_agent_tool_integration_missing_tool():
    """Verify agent returns failure AgentResult when required tool is missing."""
    agent = ToolIntegratedAgent(tools=[])  # No calculator tool

    state = create_initial_state(user_query="Compute indicators")
    update = agent.run(state)

    assert update["technical_result"]["success"] is False
    assert "not configured" in update["technical_result"]["error"]


def test_agent_tool_integration_failing_tool():
    """Verify agent catches tool failure and packages it into failed AgentResult."""

    class FailingToolAgent(BaseAgent):
        name = "failing_tool_agent"
        input_schema = MockInputSchema
        output_schema = MockOutputSchema

        def run(self, state: GraphState) -> Dict[str, Any]:
            tool = self.get_tool("mock_failing_tool")
            assert tool is not None
            try:
                tool.execute()
                result = AgentResult.create_success("ok")
            except Exception as e:
                result = AgentResult.create_failure(f"Tool error: {e}", confidence=0.0)
            return {"technical_result": result.model_dump()}

    failing_tool = MockFailingTool()
    agent = FailingToolAgent(tools=[failing_tool])

    state = create_initial_state(user_query="Run failing tool")
    update = agent.run(state)

    res = update["technical_result"]
    assert res["success"] is False
    assert "Tool error: External service unavailable" in res["error"]
    assert res["confidence"] == 0.0
