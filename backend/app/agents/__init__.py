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
from app.agents.cio import (
    CIOAgent,
    create_fallback_routing_decision,
)
from app.agents.cio_schema import (
    CORE_SPECIALISTS,
    CIOInput,
    CIORoutingDecision,
    SpecialistName,
    SpecialistTask,
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
from app.agents.fundamental import (
    FundamentalAnalystAgent,
    fundamental_analyst_node,
)
from app.agents.fundamental_schema import (
    DimensionAssessment,
    DimensionRating,
    FundamentalAnalysisOutput,
    FundamentalAnalystInput,
    OverallAssessmentRating,
)
from app.agents.graph import (
    CIO_NODE_NAME,
    CLARIFICATION_NODE_NAME,
    CONVERSATION_NODE_NAME,
    FAN_IN_NODE_NAME,
    PASSTHROUGH_NODE_NAME,
    ROUTE_CLARIFICATION_REQUIRED,
    ROUTE_READY_FOR_ANALYSIS,
    cio_node,
    clarification_node,
    conversation_node,
    create_conversation_graph,
    create_graph,
    create_orchestration_graph,
    fan_in_node,
    get_graph,
    passthrough_node,
    route_to_specialists,
    run_conversation_graph,
    run_graph,
    run_orchestration_graph,
    should_continue_after_clarification,
    should_continue_after_conversation,
)
from app.agents.specialist_stubs import (
    SPECIALIST_STATE_KEY_MAP,
    create_specialist_stub_node,
    run_with_timeout_and_isolation,
)
from app.agents.state import (
    CIORoutingDecisionState,
    ClarifiedRequest,
    GraphState,
    InvestorProfile,
    create_initial_state,
)
from app.agents.technical import (
    TechnicalAnalystAgent,
    technical_analyst_node,
)
from app.agents.technical_schema import (
    SupportResistanceSummary,
    TechnicalAnalysisOutput,
    TechnicalAnalysisValidationError,
    TechnicalAnalystInput,
    TechnicalIndicatorsSummary,
    TechnicalInterpretation,
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
    "CIORoutingDecisionState",
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
    "CONVERSATION_NODE_NAME",
    "CLARIFICATION_NODE_NAME",
    "ROUTE_CLARIFICATION_REQUIRED",
    "ROUTE_READY_FOR_ANALYSIS",
    "conversation_node",
    "clarification_node",
    "should_continue_after_conversation",
    "should_continue_after_clarification",
    "create_conversation_graph",
    "run_conversation_graph",
    "CIOAgent",
    "CIOInput",
    "CIORoutingDecision",
    "SpecialistName",
    "SpecialistTask",
    "CORE_SPECIALISTS",
    "create_fallback_routing_decision",
    "CIO_NODE_NAME",
    "FAN_IN_NODE_NAME",
    "cio_node",
    "route_to_specialists",
    "fan_in_node",
    "create_orchestration_graph",
    "run_orchestration_graph",
    "SPECIALIST_STATE_KEY_MAP",
    "create_specialist_stub_node",
    "run_with_timeout_and_isolation",
    "DimensionAssessment",
    "DimensionRating",
    "FundamentalAnalysisOutput",
    "FundamentalAnalystAgent",
    "FundamentalAnalystInput",
    "OverallAssessmentRating",
    "fundamental_analyst_node",
    "SupportResistanceSummary",
    "TechnicalAnalysisOutput",
    "TechnicalAnalysisValidationError",
    "TechnicalAnalystAgent",
    "TechnicalAnalystInput",
    "TechnicalIndicatorsSummary",
    "TechnicalInterpretation",
    "technical_analyst_node",
]
