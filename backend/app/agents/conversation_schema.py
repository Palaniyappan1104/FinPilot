"""Conversation agent schemas for FinPilot.

Phase 3.1 defines the input and output contracts for the Conversation Agent:
- ChatMessage: Structured representation of a conversational turn.
- ConversationInput: Input contract containing user message and optional history.
- ConversationOutput: Output contract with normalized query and extracted entities.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.agents.state import ClarifiedRequest


class ChatMessage(BaseModel):
    """Represents a single conversational turn within conversation history."""

    role: str = Field(
        ...,
        description="Role of the speaker (e.g. 'user', 'assistant', 'system').",
    )
    content: str = Field(
        ...,
        description="Text content of the message.",
    )

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("role must be a non-empty string.")
        return v.strip().lower()

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("content must be a non-empty string.")
        return v.strip()


ConversationMessage = ChatMessage


class ConversationInput(BaseModel):
    """Input contract for the Conversation Agent.

    Attributes:
        user_message: Raw natural-language input submitted by the user.
        conversation_history: Optional sequence of prior messages in the conversation.
    """

    user_message: str = Field(
        ...,
        description="Raw natural language prompt or query submitted by the user.",
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional list of prior chat messages in the current session.",
    )

    @field_validator("user_message", mode="before")
    @classmethod
    def validate_user_message(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("user_message must be a non-empty string.")
        return v.strip()


class ConversationOutput(BaseModel):
    """Output contract for the Conversation Agent.

    Captures the normalized intent and extracted financial entities from the user's
    request. Information not present in the user input remains None without default
    inventions.

    Attributes:
        normalized_query: Cleaned and normalized natural language user query.
        intent_type: Classified intention of the request (e.g. 'research').
        company: Extracted target company name or ticker, if mentioned.
        capital_amount: Extracted capital investment amount (non-negative).
        time_horizon: Extracted investment duration or horizon, if mentioned.
        risk_tolerance: Extracted risk preference or tolerance, if mentioned.
    """

    normalized_query: str = Field(
        ...,
        description="Cleaned, unambiguous representation of the user's intent.",
    )
    intent_type: str = Field(
        ...,
        description="Categorized intent type (e.g. 'stock_research').",
    )
    company: Optional[str] = Field(
        default=None,
        description="Identified target company name or ticker symbol, if specified.",
    )
    capital_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Investment capital amount in base currency, if specified.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Stated investment horizon (e.g. '5 years'), if specified.",
    )
    risk_tolerance: Optional[str] = Field(
        default=None,
        description="Investor risk preference (e.g. 'conservative'), if specified.",
    )

    @field_validator("normalized_query", mode="before")
    @classmethod
    def validate_normalized_query(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("normalized_query must be a non-empty string.")
        return v.strip()

    @field_validator("intent_type", mode="before")
    @classmethod
    def validate_intent_type(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("intent_type must be a non-empty string.")
        return v.strip()

    @field_validator("company", "time_horizon", "risk_tolerance", mode="before")
    @classmethod
    def normalize_optional_string(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return v

    def to_clarified_request(self) -> ClarifiedRequest:
        """Convert this output into a GraphState ClarifiedRequest TypedDict.

        Returns:
            ClarifiedRequest: Dictionary matching GraphState['clarified_request'].
        """
        entities: Dict[str, Any] = {}
        if self.company is not None:
            entities["company"] = self.company
        if self.capital_amount is not None:
            entities["capital_amount"] = self.capital_amount
        if self.time_horizon is not None:
            entities["time_horizon"] = self.time_horizon
        if self.risk_tolerance is not None:
            entities["risk_tolerance"] = self.risk_tolerance

        return ClarifiedRequest(
            normalized_query=self.normalized_query,
            intent_type=self.intent_type,
            entities=entities if entities else None,
            clarification_needed=None,
            clarification_questions=None,
        )
