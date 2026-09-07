"""Base Agent interface and AgentResult wrapper for FinPilot.

Phase 2.4 establishes the core contracts for specialist agents:
- AgentResult: Common result wrapper with success, data, error, and confidence.
- BaseAgent (Agent): Abstract base class defining name, input/output schemas,
  tool integration, and execution against shared GraphState.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from app.agents.state import GraphState
from app.agents.tools import Tool


class AgentResult(BaseModel):
    """Standard execution result wrapper returned by FinPilot agents.

    Attributes:
        success: Whether the agent execution completed successfully.
        data: Execution data payload or structured model output.
        error: Error description if execution failed.
        confidence: Analytical confidence score between 0.0 and 1.0 (or None).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(
        ...,
        description="Whether the agent execution completed successfully.",
    )
    data: Optional[Any] = Field(
        default=None,
        description="Execution data payload or structured model output.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error description if execution failed.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0, if applicable.",
    )

    @classmethod
    def create_success(
        cls,
        data: Any = None,
        confidence: Optional[float] = None,
    ) -> "AgentResult":
        """Factory helper for creating a successful AgentResult.

        Args:
            data: Structured payload or execution result.
            confidence: Optional confidence score between 0.0 and 1.0.

        Returns:
            AgentResult: Successful result instance.
        """
        return cls(
            success=True,
            data=data,
            error=None,
            confidence=confidence,
        )

    @classmethod
    def create_failure(
        cls,
        error: str,
        confidence: Optional[float] = None,
    ) -> "AgentResult":
        """Factory helper for creating a failed AgentResult.

        Args:
            error: Descriptive error message.
            confidence: Optional confidence score (e.g. 0.0).

        Returns:
            AgentResult: Failed result instance.
        """
        return cls(
            success=False,
            data=None,
            error=error,
            confidence=confidence,
        )

    def to_state_update(self, key: str) -> Dict[str, Any]:
        """Convert this result into a GraphState update dictionary.

        Args:
            key: Target GraphState field name (e.g. 'technical_result').

        Returns:
            Dict[str, Any]: State update mapping {key: model_dump()}.
        """
        return {key: self.model_dump()}


class BaseAgent(ABC):
    """Abstract base class for all FinPilot agents.

    An agent encapsulates an analytical or conversational role in the workflow.
    It reads from GraphState, may use registered Tools, and produces results
    or state updates.
    """

    def __init__(self, tools: Optional[list[Tool]] = None) -> None:
        """Initialize the agent with optional tools.

        Args:
            tools: Sequence of Tool instances available to this agent.
        """
        self._tools: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self._tools[tool.name] = tool

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name of the agent."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Type[Any]:
        """Schema defining the input expected by this agent."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Type[Any]:
        """Schema defining the structured output produced by this agent."""
        pass

    @property
    def tools(self) -> dict[str, Tool]:
        """Return the dictionary of registered tools keyed by tool name."""
        if not hasattr(self, "_tools"):
            self._tools = {}
        return self._tools

    def get_tool(self, name: str) -> Optional[Tool]:
        """Retrieve a registered tool by its name.

        Args:
            name: Tool name identifier.

        Returns:
            Optional[Tool]: The tool instance if registered, otherwise None.
        """
        return self.tools.get(name)

    @abstractmethod
    def run(self, state: GraphState) -> Union[Dict[str, Any], AgentResult]:
        """Execute the agent logic against the shared graph state.

        Args:
            state: Current GraphState dictionary.

        Returns:
            Union[Dict[str, Any], AgentResult]: Either a partial state update
            dictionary conforming to LangGraph node conventions, or an AgentResult.
        """
        pass


Agent = BaseAgent
