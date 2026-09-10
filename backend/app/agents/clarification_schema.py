"""Clarification agent schemas for FinPilot.

Phase 3.3 defines the input, output, and question-generation contracts:
- ClarificationQuestionsModel: Model for LLM-generated clarification questions.
- ClarificationInput: Input containing ConversationOutput and prior context.
- ClarificationOutput: Output detailing completeness, missing parameters, and questions.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.conversation_schema import ConversationOutput
from app.agents.state import ClarifiedRequest, InvestorProfile


class ClarificationQuestionsModel(BaseModel):
    """Structured response schema for LLM-generated clarification questions."""

    questions: List[str] = Field(
        ...,
        description="Targeted questions asking ONLY for missing parameters.",
    )


class ClarificationInput(BaseModel):
    """Input contract for the Clarification Agent.

    Attributes:
        conversation_output: Intent and entities extracted by ConversationAgent.
        existing_profile: Optional profile or entities from previous turns.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conversation_output: ConversationOutput = Field(
        ...,
        description="Structured intent and entities extracted by Conversation Agent.",
    )
    existing_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional investor profile from previous turns.",
    )


class ClarificationOutput(BaseModel):
    """Output contract for the Clarification Agent.

    Captures the completeness decision, identified missing parameters, generated
    clarification questions, and preserved extracted entities without defaults.

    Attributes:
        clarification_needed: Whether clarification is required before analysis.
        missing_parameters: List of parameter names identified as missing.
        clarification_questions: User-facing questions asking for missing fields.
        normalized_query: Preserved normalized query from upstream ConversationOutput.
        intent_type: Preserved intent classification.
        company: Extracted or merged target company name or ticker, if present.
        capital_amount: Extracted or merged investment capital amount, if present.
        time_horizon: Extracted or merged investment duration/horizon, if present.
        risk_tolerance: Extracted or merged investor risk tolerance, if present.
    """

    clarification_needed: bool = Field(
        ...,
        description="Whether clarification is required before proceeding to analysis.",
    )
    missing_parameters: List[str] = Field(
        default_factory=list,
        description="List of required parameters that are missing.",
    )
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="Targeted clarification questions asking for missing parameters.",
    )
    normalized_query: str = Field(
        ...,
        description="Preserved normalized user query.",
    )
    intent_type: str = Field(
        ...,
        description="Preserved classified intent type.",
    )
    company: Optional[str] = Field(
        default=None,
        description="Target company name or ticker symbol, if specified.",
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
            clarification_needed=self.clarification_needed,
            clarification_questions=list(self.clarification_questions),
        )

    def to_investor_profile(self) -> InvestorProfile:
        """Convert this output into a GraphState InvestorProfile TypedDict.

        Returns:
            InvestorProfile: Dictionary matching GraphState['investor_profile'].
        """
        return InvestorProfile(
            target_company=self.company,
            ticker=None,
            investment_goal=None,
            time_horizon=self.time_horizon,
            capital_amount=self.capital_amount,
            risk_tolerance=self.risk_tolerance,
            profile_complete=not self.clarification_needed,
        )
