"""Unit tests for Phase 8.3 News Analyst Agent and synthesis logic.

All tests run 100% offline using deterministic mock LLM providers.
ZERO real Gemini API or Finnhub network calls are made.

Verifies:
- Valid structured news analysis output generation and parsing.
- Ticker validation and mismatch rejection.
- Evidence and recent-news article reference grounding (rejecting invented IDs).
- Graceful handling of zero articles / insufficient news.
- Graceful handling when all articles failed Phase 8.2 classification.
- Rejection of invalid overall sentiment values.
- Confidence ceiling and bounds enforcement.
- Rejection of prohibited recommendation and price target phrases.
- Conflicting signals preservation (positive and negative factors).
- Rejection of unsupported events not present in input articles.
- LangGraph node adapter contract.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import pytest
from pydantic import ValidationError

from app.agents.base import AgentResult
from app.agents.news import (
    NewsAnalystAgent,
    calculate_news_confidence,
    news_analyst_node,
)
from app.agents.news_schema import (
    NewsAnalysisOutput,
    NewsAnalystInput,
)
from app.agents.state import GraphState
from app.core.llm.base import LLMProvider, LLMResponse
from app.models.news_processing import (
    NewsEvent,
    ProcessedNewsArticle,
)

# ===========================================================================
# Mock LLM Provider
# ===========================================================================


class MockNewsSynthesisLLMProvider(LLMProvider):
    """Deterministic mock LLM provider for Phase 8.3 News Analyst tests."""

    def __init__(
        self,
        responses: Optional[List[Union[str, Dict[str, Any]]]] = None,
        responder: Optional[Callable[[str], str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_news_synthesis_llm",
    ) -> None:
        self.responses = responses or []
        self.responder = responder
        self.fail_with = fail_with
        self.call_count = 0
        self.prompts_received: List[str] = []
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        schema: Optional[Any] = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)

        if self.fail_with is not None:
            raise self.fail_with

        if self.responder is not None:
            content = self.responder(prompt)
            return LLMResponse(content=content, provider=self._name, model="mock-model")

        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            resp = self.responses[idx]
            if isinstance(resp, dict):
                content = json.dumps(resp)
            else:
                content = str(resp)
            return LLMResponse(content=content, provider=self._name, model="mock-model")

        raise RuntimeError("No mock response queued.")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_processed_articles() -> List[ProcessedNewsArticle]:
    """Sample collection of Phase 8.2 ProcessedNewsArticle items."""
    now_utc = datetime.now(timezone.utc) - timedelta(hours=2)
    a1 = ProcessedNewsArticle(
        article_id="art-001",
        headline="Nvidia Reports Record Revenue Driven by Enterprise AI Hardware",
        summary="Quarterly revenue surged 265% year-over-year to $22.1 billion.",
        sentiment="positive",
        sentiment_reasoning="Record revenue growth reported in quarterly results.",
        important_events=[
            NewsEvent(
                event_type="earnings",
                evidence="Revenue surged 265% YoY to $22.1B.",
            )
        ],
        source="Reuters",
        url="https://www.reuters.com/tech/nvda-q4-2025",
        published_at=now_utc,
        ticker="NVDA",
        related_tickers=["NVDA"],
        is_classified=True,
    )
    a2 = ProcessedNewsArticle(
        article_id="art-002",
        headline="Regulator Reviews Tech Supply Chains for Compliance",
        summary="Antitrust agency initiated an inquiry into cloud chip distribution.",
        sentiment="negative",
        sentiment_reasoning="Regulatory scrutiny on distribution agreements.",
        important_events=[
            NewsEvent(
                event_type="regulatory_action",
                evidence="Antitrust agency opened preliminary inquiry.",
            )
        ],
        source="Bloomberg",
        url="https://www.bloomberg.com/news/chips-probe",
        published_at=now_utc,
        ticker="NVDA",
        related_tickers=["NVDA"],
        is_classified=True,
    )
    a3 = ProcessedNewsArticle(
        article_id="art-003",
        headline="Industry Executives to Speak at Upcoming Silicon Expo",
        summary="Keynote speakers announced for the annual technology summit.",
        sentiment="neutral",
        sentiment_reasoning="Routine conference announcement.",
        important_events=[],
        source="PR Newswire",
        url="https://www.prnewswire.com/silicon-expo",
        published_at=now_utc,
        ticker="NVDA",
        related_tickers=["NVDA"],
        is_classified=True,
    )
    return [a1, a2, a3]


@pytest.fixture
def sample_valid_output_payload() -> Dict[str, Any]:
    """Valid dictionary matching NewsAnalysisOutput for NVDA."""
    now_str = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    return {
        "ticker": "NVDA",
        "overall_sentiment": "neutral",
        "sentiment_distribution": {"positive": 1, "negative": 1, "neutral": 1},
        "recent_news": [
            {
                "article_id": "art-001",
                "headline": (
                    "Nvidia Reports Record Revenue Driven by Enterprise AI Hardware"
                ),
                "summary": "Quarterly revenue surged 265% YoY.",
                "source": "Reuters",
                "url": "https://www.reuters.com/tech/nvda-q4-2025",
                "published_at": now_str,
                "sentiment": "positive",
            },
            {
                "article_id": "art-002",
                "headline": "Regulator Reviews Tech Supply Chains for Compliance",
                "summary": "Antitrust agency initiated an inquiry.",
                "source": "Bloomberg",
                "url": "https://www.bloomberg.com/news/chips-probe",
                "published_at": now_str,
                "sentiment": "negative",
            },
        ],
        "important_events": [
            {
                "event_type": "earnings",
                "description": "Revenue surged 265% YoY to $22.1B.",
                "article_ids": ["art-001"],
            },
            {
                "event_type": "regulatory_action",
                "description": "Antitrust agency opened preliminary inquiry.",
                "article_ids": ["art-002"],
            },
        ],
        "positive_factors": [
            {
                "text": "Exceptional quarterly revenue growth of 265% YoY.",
                "article_ids": ["art-001"],
            }
        ],
        "negative_factors": [
            {
                "text": "Regulatory antitrust probe into supply distribution.",
                "article_ids": ["art-002"],
            }
        ],
        "confidence": 0.70,
        "summary": (
            "Current news coverage reflects strong operational and earnings results "
            "counterbalanced by an emerging regulatory inquiry."
        ),
        "evidence": [
            "Reuters reported $22.1B quarterly revenue (art-001).",
            "Bloomberg reported antitrust inquiry initiation (art-002).",
        ],
    }


# ===========================================================================
# 1. Output Schema & Structured Synthesis Tests (8.3.1, 8.3.2, 8.3.4)
# ===========================================================================


def test_valid_structured_news_analysis_output(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify happy-path structured news synthesis execution."""
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result: AgentResult = agent.run(
        NewsAnalystInput(
            ticker="NVDA",
            articles=sample_processed_articles,
        )
    )

    assert result.success is True
    output: NewsAnalysisOutput = result.data
    assert isinstance(output, NewsAnalysisOutput)
    assert output.ticker == "NVDA"
    assert output.overall_sentiment == "neutral"
    assert len(output.recent_news) == 2
    assert len(output.important_events) == 2
    assert output.positive_factors[0].article_ids == ["art-001"]
    assert output.negative_factors[0].article_ids == ["art-002"]
    assert output.confidence == 0.70


def test_ticker_validation_mismatch_rejected(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify error when model output ticker mismatches the requested ticker."""
    sample_valid_output_payload["ticker"] = "AAPL"
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert "Ticker mismatch" in result.error


def test_evidence_and_recent_news_reference_validation(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify error when recent news cites an article_id not present in input."""
    # Introduce an invented article ID
    sample_valid_output_payload["recent_news"][0]["article_id"] = "art-fabricated"
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert "unknown article_id 'art-fabricated'" in result.error


def test_source_mismatch_in_recent_news_rejected(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify error when recent news alters the source organization."""
    sample_valid_output_payload["recent_news"][0]["source"] = "Fabricated Wire"
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert "Source mismatch for article 'art-001'" in result.error


# ===========================================================================
# 2. Insufficient News Handling Tests
# ===========================================================================


def test_insufficient_news_handling_empty_articles():
    """Verify zero articles produces deterministic insufficient_news without LLM."""
    provider = MockNewsSynthesisLLMProvider(responses=[])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="TSLA", articles=[]))

    assert result.success is True
    assert provider.call_count == 0  # Zero LLM quota consumed
    output: NewsAnalysisOutput = result.data
    assert output.overall_sentiment == "insufficient_news"
    assert output.confidence == 0.0
    assert output.recent_news == []
    assert output.important_events == []
    assert output.positive_factors == []
    assert output.negative_factors == []
    assert "Insufficient usable news data" in output.summary


def test_insufficient_news_handling_all_failed_classification():
    """Verify articles with is_classified=False are treated as insufficient news."""
    failed_article = ProcessedNewsArticle(
        article_id="art-fail",
        headline="Some Headline",
        sentiment="neutral",
        sentiment_reasoning="Processing failure",
        source="Wire",
        published_at=datetime.now(timezone.utc),
        is_classified=False,
        processing_error="Connection error",
    )
    provider = MockNewsSynthesisLLMProvider(responses=[])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="TSLA", articles=[failed_article]))

    assert result.success is True
    assert provider.call_count == 0
    output: NewsAnalysisOutput = result.data
    assert output.overall_sentiment == "insufficient_news"
    assert output.confidence == 0.0


# ===========================================================================
# 3. Deterministic Overall Sentiment Grounding (Fix 1)
# ===========================================================================


def test_unanimous_positive_cannot_produce_negative(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify unanimous positive articles cannot produce negative overall sentiment."""
    positive_only = [sample_processed_articles[0]]
    sample_valid_output_payload["overall_sentiment"] = "negative"
    sample_valid_output_payload["negative_factors"] = []
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="NVDA", articles=positive_only))

    assert result.success is False
    assert "Overall sentiment cannot be 'negative'" in result.error


def test_unanimous_negative_cannot_produce_positive(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify unanimous negative articles cannot produce positive overall sentiment."""
    negative_only = [sample_processed_articles[1]]
    sample_valid_output_payload["overall_sentiment"] = "positive"
    sample_valid_output_payload["positive_factors"] = []
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="NVDA", articles=negative_only))

    assert result.success is False
    assert "Overall sentiment cannot be 'positive'" in result.error


def test_failed_classifications_excluded_from_distribution(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify failed/unclassified articles are not counted in sentiment_distribution."""
    failed_article = ProcessedNewsArticle(
        article_id="art-fail",
        headline="Bad Data",
        sentiment="neutral",
        sentiment_reasoning="Processing failure occurred",
        published_at=datetime.now(timezone.utc),
        source="Unknown Source",
        is_classified=False,
        processing_error="Corrupt text",
    )
    articles = [sample_processed_articles[0], failed_article]
    sample_valid_output_payload["overall_sentiment"] = "positive"
    sample_valid_output_payload["recent_news"] = [
        sample_valid_output_payload["recent_news"][0]
    ]
    sample_valid_output_payload["important_events"] = [
        sample_valid_output_payload["important_events"][0]
    ]
    sample_valid_output_payload["negative_factors"] = []
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="NVDA", articles=articles))

    assert result.success is True
    output: NewsAnalysisOutput = result.data
    assert output.sentiment_distribution == {"positive": 1, "negative": 0, "neutral": 0}


def test_sentiment_distribution_is_deterministic(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify sentiment_distribution is strictly assigned from usable input articles."""
    sample_valid_output_payload["sentiment_distribution"] = {
        "positive": 99,
        "negative": 99,
        "neutral": 99,
    }
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is True
    output: NewsAnalysisOutput = result.data
    assert output.sentiment_distribution == {"positive": 1, "negative": 1, "neutral": 1}


# ===========================================================================
# 4. Structured Factor Provenance (Fix 2)
# ===========================================================================


def test_positive_factor_requires_valid_article_ids(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify positive factor with empty article_ids is rejected."""
    sample_valid_output_payload["positive_factors"] = [
        {"text": "Strong AI growth.", "article_ids": []}
    ]
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert (
        "Positive factor" in result.error
        or "at least one non-empty article_id" in result.error
        or "validation failed" in result.error
    )


def test_negative_factor_requires_valid_article_ids(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify negative factor with empty article_ids is rejected."""
    sample_valid_output_payload["negative_factors"] = [
        {"text": "Antitrust scrutiny.", "article_ids": []}
    ]
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert (
        "Negative factor" in result.error
        or "at least one non-empty article_id" in result.error
        or "validation failed" in result.error
    )


def test_unknown_factor_article_id_rejected(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify factor citing an unsupplied article_id is rejected."""
    sample_valid_output_payload["positive_factors"] = [
        {"text": "Strong earnings growth.", "article_ids": ["art-ghost-id"]}
    ]
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert "references unknown article_id 'art-ghost-id'" in result.error


# ===========================================================================
# 5. Structured Important Event Provenance (Fix 3)
# ===========================================================================


def test_important_event_requires_article_ids(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify important event with empty article_ids is rejected."""
    sample_valid_output_payload["important_events"] = [
        {
            "event_type": "earnings",
            "description": "Revenue beat estimates.",
            "article_ids": [],
        }
    ]
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert (
        "Important event" in result.error
        or "at least one non-empty article_id" in result.error
        or "validation failed" in result.error
    )


def test_event_article_id_must_actually_contain_event_type(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify event is rejected when cited article lacks that event type."""
    sample_valid_output_payload["important_events"] = [
        {
            "event_type": "regulatory_action",
            "description": "Regulatory inquiry launched.",
            "article_ids": ["art-001"],
        }
    ]
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert "was not detected in referenced article 'art-001'" in result.error


def test_fabricated_event_evidence_rejected(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify fabricated event description with no article grounding is rejected."""
    sample_valid_output_payload["important_events"] = [
        {
            "event_type": "earnings",
            "description": "Catastrophic liquidation crisis and massive losses.",
            "article_ids": ["art-001"],
        }
    ]
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert "lacks factual grounding" in result.error


# ===========================================================================
# 6. Deterministic Confidence Grounding (Fix 4)
# ===========================================================================


def test_confidence_is_deterministic_for_identical_input(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify identical input always results in the exact same confidence score."""
    conf1 = calculate_news_confidence(sample_processed_articles)
    conf2 = calculate_news_confidence(sample_processed_articles)
    assert conf1 == conf2 == 0.70

    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)
    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )
    assert result.success is True
    assert result.confidence == 0.70
    assert result.data.confidence == 0.70


def test_llm_confidence_cannot_override_deterministic_confidence(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify LLM attempt to output arbitrary confidence (e.g. 0.15) is overridden."""
    sample_valid_output_payload["confidence"] = 0.15
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is True
    assert result.data.confidence == 0.70
    assert result.confidence == 0.70


# ===========================================================================
# 7. Safety and Schema Boundary Tests
# ===========================================================================


def test_invalid_overall_sentiment_rejection():
    """Verify invalid overall_sentiment value is rejected by schema validator."""
    with pytest.raises(ValidationError, match="Input should be"):
        NewsAnalysisOutput(
            ticker="NVDA",
            overall_sentiment="bullish",  # Disallowed
            confidence=0.5,
            summary="Some summary.",
        )


def test_prohibited_recommendation_language_rejected(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify model output offering buy/sell ratings is rejected."""
    sample_valid_output_payload["summary"] = (
        "Strong quarterly results make this a strong buy recommendation for growth."
    )
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=sample_processed_articles)
    )

    assert result.success is False
    assert "Prohibited advice or recommendation phrase detected" in result.error


# ===========================================================================
# 8. LangGraph Node Adapter Contract Tests
# ===========================================================================


def test_news_analyst_node_adapter(
    sample_processed_articles: List[ProcessedNewsArticle],
    sample_valid_output_payload: Dict[str, Any],
):
    """Verify news_analyst_node executes agent and updates news_result."""
    provider = MockNewsSynthesisLLMProvider(responses=[sample_valid_output_payload])
    agent = NewsAnalystAgent(provider=provider)

    state: GraphState = {
        "target_company": "NVDA",
        "ticker": "NVDA",
        "processed_news": sample_processed_articles,
    }

    update = news_analyst_node(state=state, agent=agent)

    assert "news_result" in update
    assert update["news_result"]["ticker"] == "NVDA"
    assert update["news_result"]["overall_sentiment"] == "neutral"
    assert update["news_result"]["confidence"] == 0.70
