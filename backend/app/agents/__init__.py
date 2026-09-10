"""Multi-agent package for FinPilot.

Phase 2.3 defines the shared graph state used across workflow nodes.
Phase 2.4 defines the BaseAgent interface, AgentResult wrapper, and Tool abstraction.
Phase 2.5 defines the minimal LangGraph workflow wiring.
Phase 3.1 defines the input/output schemas for the Conversation Agent.
Phase 3.2 implements the Conversation Agent.
Phase 3.3 implements the Clarification Agent and completeness evaluation.
"""

from app.agents.base import (
    Agent,
    AgentResult,
    BaseAgent,
)
from app.agents.clarification import ClarificationAgent
from app.agents.clarification_schema import (
    ClarificationInput,
    ClarificationOutput,
    ClarificationQuestionsModel,
)
from app.agents.conversation import ConversationAgent
from app.agents.conversation_schema import (
    ChatMessage,
    ConversationInput,
    ConversationMessage,
    ConversationOutput,
)
from app.agents.graph import (
    PASSTHROUGH_NODE_NAME,
    create_graph,
    get_graph,
    passthrough_node,
    run_graph,
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
    "PASSTHROUGH_NODE_NAME",
    "passthrough_node",
    "create_graph",
    "get_graph",
    "run_graph",
    "ConversationAgent",
    "ChatMessage",
    "ConversationMessage",
    "ConversationInput",
    "ConversationOutput",
    "ClarificationAgent",
    "ClarificationInput",
    "ClarificationOutput",
    "ClarificationQuestionsModel",
]
