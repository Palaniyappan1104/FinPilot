"""Unit tests for Phase 3.1 Conversation Agent schemas.

Tests cover:
- Valid ConversationInput with and without conversation history
- Rejection of empty and whitespace-only user messages
- ChatMessage validation and role/content normalization
- Valid ConversationOutput with full, partial, and minimal fields
- Preservation of None for absent optional entities
- Rejection of negative capital and acceptance of zero capital
- Proper normalization of blank/whitespace optional strings to None
- Rejection of empty normalized_query and intent_type
- Conversion from ConversationOutput to GraphState ClarifiedRequest
"""

import pytest
from pydantic import ValidationError

from app.agents.conversation_schema import (
    ChatMessage,
    ConversationInput,
    ConversationMessage,
    ConversationOutput,
)

# ---------------------------------------------------------------------------
# 1. ChatMessage Tests
# ---------------------------------------------------------------------------


def test_chat_message_valid():
    """Verify valid ChatMessage instantiation and attribute normalization."""
    msg = ChatMessage(role="User", content="  Hello, I want to invest.  ")
    assert msg.role == "user"
    assert msg.content == "Hello, I want to invest."


def test_chat_message_alias():
    """Verify ConversationMessage alias points to ChatMessage."""
    assert ConversationMessage is ChatMessage
    msg = ConversationMessage(role="assistant", content="How can I help?")
    assert msg.role == "assistant"
    assert msg.content == "How can I help?"


def test_chat_message_empty_role_rejected():
    """Verify empty or whitespace-only role raises ValidationError."""
    with pytest.raises(ValidationError):
        ChatMessage(role="", content="Valid content")

    with pytest.raises(ValidationError):
        ChatMessage(role="   ", content="Valid content")


def test_chat_message_empty_content_rejected():
    """Verify empty or whitespace-only content raises ValidationError."""
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   ")


# ---------------------------------------------------------------------------
# 2. ConversationInput Tests
# ---------------------------------------------------------------------------


def test_conversation_input_valid_without_history():
    """Verify valid ConversationInput without conversation history."""
    inp = ConversationInput(user_message="Should I invest in Infosys?")
    assert inp.user_message == "Should I invest in Infosys?"
    assert inp.conversation_history is None


def test_conversation_input_user_message_stripped():
    """Verify user_message has leading/trailing whitespace stripped."""
    inp = ConversationInput(user_message="   Analyze TCS stock   ")
    assert inp.user_message == "Analyze TCS stock"


def test_conversation_input_empty_user_message_rejected():
    """Verify empty user_message raises ValidationError."""
    with pytest.raises(ValidationError):
        ConversationInput(user_message="")


def test_conversation_input_whitespace_only_user_message_rejected():
    """Verify whitespace-only user_message raises ValidationError."""
    with pytest.raises(ValidationError):
        ConversationInput(user_message="   \n\t   ")


def test_conversation_input_with_history_accepted():
    """Verify ConversationInput accepts typed conversation history."""
    history = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello! How can I assist you?"),
    ]
    inp = ConversationInput(
        user_message="Analyze Reliance Industries",
        conversation_history=history,
    )
    assert inp.user_message == "Analyze Reliance Industries"
    assert inp.conversation_history is not None
    assert len(inp.conversation_history) == 2
    assert inp.conversation_history[0].role == "user"
    assert inp.conversation_history[1].content == "Hello! How can I assist you?"


def test_conversation_input_with_dict_history_accepted():
    """Verify ConversationInput validates raw dictionary history into ChatMessage."""
    raw_history = [
        {"role": "user", "content": "What is P/E ratio?"},
        {"role": "assistant", "content": "Price to Earnings ratio..."},
    ]
    inp = ConversationInput(
        user_message="What about TCS?",
        conversation_history=raw_history,  # type: ignore[arg-type]
    )
    assert inp.conversation_history is not None
    assert len(inp.conversation_history) == 2
    assert isinstance(inp.conversation_history[0], ChatMessage)


# ---------------------------------------------------------------------------
# 3. ConversationOutput Tests
# ---------------------------------------------------------------------------


def test_conversation_output_valid_all_fields():
    """Verify ConversationOutput with all entities populated."""
    output = ConversationOutput(
        normalized_query=(
            "Should I invest 100000 in TCS for 5 years with moderate risk?"
        ),
        intent_type="investment_analysis",
        company="TCS",
        capital_amount=100000.0,
        time_horizon="5 years",
        risk_tolerance="moderate",
    )
    assert output.normalized_query == (
        "Should I invest 100000 in TCS for 5 years with moderate risk?"
    )
    assert output.intent_type == "investment_analysis"
    assert output.company == "TCS"
    assert output.capital_amount == 100000.0
    assert output.time_horizon == "5 years"
    assert output.risk_tolerance == "moderate"


def test_conversation_output_minimal_valid():
    """Verify ConversationOutput with only required fields."""
    output = ConversationOutput(
        normalized_query="What is the stock price of Apple?",
        intent_type="stock_price_lookup",
    )
    assert output.normalized_query == "What is the stock price of Apple?"
    assert output.intent_type == "stock_price_lookup"
    assert output.company is None
    assert output.capital_amount is None
    assert output.time_horizon is None
    assert output.risk_tolerance is None


def test_conversation_output_missing_optional_fields_remain_none():
    """Verify missing optional fields strictly remain None without invented defaults."""
    output = ConversationOutput(
        normalized_query="Tell me about tech stocks",
        intent_type="sector_research",
    )
    assert output.company is None
    assert output.capital_amount is None
    assert output.time_horizon is None
    assert output.risk_tolerance is None


def test_conversation_output_empty_normalized_query_rejected():
    """Verify empty or whitespace-only normalized_query raises ValidationError."""
    with pytest.raises(ValidationError):
        ConversationOutput(normalized_query="", intent_type="research")

    with pytest.raises(ValidationError):
        ConversationOutput(normalized_query="   ", intent_type="research")


def test_conversation_output_empty_intent_type_rejected():
    """Verify empty or whitespace-only intent_type raises ValidationError."""
    with pytest.raises(ValidationError):
        ConversationOutput(normalized_query="Analyze NVDA", intent_type="")

    with pytest.raises(ValidationError):
        ConversationOutput(normalized_query="Analyze NVDA", intent_type="   ")


def test_conversation_output_blank_optional_strings_normalize_to_none():
    """Verify empty or whitespace-only optional strings normalize to None."""
    output = ConversationOutput(
        normalized_query="Analyze market trends",
        intent_type="general_research",
        company="   ",
        time_horizon="",
        risk_tolerance="  \t  ",
    )
    assert output.company is None
    assert output.time_horizon is None
    assert output.risk_tolerance is None


def test_conversation_output_optional_strings_stripped():
    """Verify non-blank optional strings have whitespace stripped properly."""
    output = ConversationOutput(
        normalized_query="Analyze TCS",
        intent_type="research",
        company="  Tata Consultancy Services  ",
        time_horizon="  3-5 years  ",
        risk_tolerance="  Conservative  ",
    )
    assert output.company == "Tata Consultancy Services"
    assert output.time_horizon == "3-5 years"
    assert output.risk_tolerance == "Conservative"


def test_conversation_output_negative_capital_rejected():
    """Verify negative capital amount raises ValidationError."""
    with pytest.raises(ValidationError):
        ConversationOutput(
            normalized_query="Invest -50000 in Reliance",
            intent_type="investment_analysis",
            capital_amount=-50000.0,
        )

    with pytest.raises(ValidationError):
        ConversationOutput(
            normalized_query="Invest -0.01 in Reliance",
            intent_type="investment_analysis",
            capital_amount=-0.01,
        )


def test_conversation_output_zero_capital_accepted():
    """Verify capital amount of zero is accepted (non-negative)."""
    output = ConversationOutput(
        normalized_query="Evaluate strategy with zero initial capital",
        intent_type="strategy_evaluation",
        capital_amount=0.0,
    )
    assert output.capital_amount == 0.0


# ---------------------------------------------------------------------------
# 4. ClarifiedRequest Conversion Tests
# ---------------------------------------------------------------------------


def test_conversation_output_to_clarified_request_all_entities():
    """Verify conversion to GraphState ClarifiedRequest with entities."""
    output = ConversationOutput(
        normalized_query="Should I invest 50000 in HDFC for 3 years?",
        intent_type="investment_analysis",
        company="HDFC",
        capital_amount=50000.0,
        time_horizon="3 years",
        risk_tolerance="moderate",
    )
    clarified = output.to_clarified_request()

    assert clarified["normalized_query"] == (
        "Should I invest 50000 in HDFC for 3 years?"
    )
    assert clarified["intent_type"] == "investment_analysis"
    assert clarified["entities"] is not None
    assert clarified["entities"]["company"] == "HDFC"
    assert clarified["entities"]["capital_amount"] == 50000.0
    assert clarified["entities"]["time_horizon"] == "3 years"
    assert clarified["entities"]["risk_tolerance"] == "moderate"
    assert clarified["clarification_needed"] is None
    assert clarified["clarification_questions"] is None


def test_conversation_output_to_clarified_request_no_entities():
    """Verify conversion to ClarifiedRequest when no optional entities are present."""
    output = ConversationOutput(
        normalized_query="Market overview",
        intent_type="market_summary",
    )
    clarified = output.to_clarified_request()

    assert clarified["normalized_query"] == "Market overview"
    assert clarified["intent_type"] == "market_summary"
    assert clarified["entities"] is None
    assert clarified["clarification_needed"] is None
    assert clarified["clarification_questions"] is None
