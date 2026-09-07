"""Multi-agent package for FinPilot.

Phase 2.3 defines the shared graph state used across workflow nodes.
Phase 2.4 defines the BaseAgent interface, AgentResult wrapper, and Tool abstraction.
"""

from app.agents.base import (
    Agent,
    AgentResult,
    BaseAgent,
)
from app.agents.state import (
    ClarifiedRequest,
    GraphState,
    InvestorProfile,
    create_initial_state,
)
from app.agents.tools import (
    BaseTool,
    Tool,
)

__all__ = [
    "Agent",
    "BaseAgent",
    "AgentResult",
    "Tool",
    "BaseTool",
    "GraphState",
    "InvestorProfile",
    "ClarifiedRequest",
    "create_initial_state",
]
