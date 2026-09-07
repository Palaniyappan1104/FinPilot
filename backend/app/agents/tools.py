"""Tool abstraction for FinPilot agents.

Phase 2.4 defines the Tool abstraction so agents access external capabilities
and data sources (e.g. market data, financial metrics, web scraping, document
retrieval) through tools rather than embedding external API calls directly into
agent classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class Tool(ABC):
    """Abstract base class defining the contract for all FinPilot tools.

    Subclasses must provide:
        - name: Unique identifier for the tool.
        - description: Explanation of capabilities and parameters.
        - execute(*args, **kwargs): Method performing the tool's action.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initialize tool with optional name and description overrides."""
        if name is not None:
            self._name = name
        if description is not None:
            self._description = description

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of what the tool does."""
        pass

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool logic.

        Args:
            *args: Positional arguments for execution.
            **kwargs: Keyword arguments for execution.

        Returns:
            Any: The output of the tool execution.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Convenience caller delegating directly to execute()."""
        return self.execute(*args, **kwargs)


BaseTool = Tool
