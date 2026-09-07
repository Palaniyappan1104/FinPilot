"""Multi-agent package for FinPilot.

Phase 2.3 defines the shared graph state used across workflow nodes.
"""

from app.agents.state import (
    ClarifiedRequest,
    GraphState,
    InvestorProfile,
    create_initial_state,
)

__all__ = [
    "GraphState",
    "InvestorProfile",
    "ClarifiedRequest",
    "create_initial_state",
]
