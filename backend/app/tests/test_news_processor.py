"""Unit tests for Phase 8.2 News Processing.

All tests run 100% offline using deterministic mock LLM providers.
ZERO real Gemini API or Finnhub network calls are made.

Verifies:
1. Valid positive article classification
2. Valid negative article classification
3. Valid neutral article classification
4. Ambiguous article -> neutral classification
5. Earnings event detection
6. Management-change event detection
7. Regulatory-action event detection
8. M&A event detection
9. Article with no important event (empty events list)
10. Multiple supported events on a single article
11. Article with missing summary (None preserved, prompt handled)
12. Missing/empty article title rejected
13. Malformed structured LLM output handling
14. Unsupported sentiment value rejected
15. Unsupported event type rejected
16. Evidence remains grounded in output
17. article_id strictly preserved
18. source, url, and published_at strictly preserved
19. Batch processing: one failed article does not corrupt the batch
20. Prohibited investment recommendation language is rejected
"""

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import pytest

from app.core.llm.base import LLMProvider, LLMResponse
from app.core.llm.exceptions import LLMStructuredOutputError
from app.models.news import NewsArticle
from app.models.news_processing import (
    ProcessedNewsArticle,
)
from app.services.news_processor import (
    NewsProcessor,
)

# ===========================================================================
# Mock LLM Provider
# ===========================================================================


class MockNewsLLMProvider(LLMProvider):
    """Deterministic mock LLM provider for Phase 8.2 News Processing tests."""

    def __init__(
        self,
        responses: Optional[List[Union[str, Dict[str, Any]]]] = None,
        responder: Optional[Callable[[str], str]] = None,
        fail_with: Optional[Exception] = None,
        name: str = "mock_news_llm",
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

        # Default fallback response
        default = {
            "sentiment": "neutral",
            "sentiment_reasoning": (
                "Standard business operations reported without material variance."
            ),
            "events": [],
        }
        return LLMResponse(
            content=json.dumps(default), provider=self._name, model="mock-model"
        )


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def base_article() -> NewsArticle:
    """Standard normalized NewsArticle fixture."""
    return NewsArticle(
        article_id="art-1001",
        ticker="NVDA",
        title=(
            "Nvidia Reports Record Fourth Quarter Revenue Beating Street " "Estimates"
        ),
        summary=(
            "Nvidia delivered quarterly revenue of $22.1 billion, surging 265% "
            "year-over-year driven by accelerating enterprise demand for AI compute."
        ),
        source="Reuters",
        url="https://www.reuters.com/technology/nvidia-record-q4-2025",
        published_at=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
    )


# ===========================================================================
# 1. Sentiment Classification Tests (8.2.2)
# ===========================================================================


def test_valid_positive_article(base_article: NewsArticle):
    """Verify strong operational/earnings beat classifies as positive."""
    mock_payload = {
        "sentiment": "positive",
        "sentiment_reasoning": (
            "Quarterly revenue surged 265% year-over-year to $22.1 billion, "
            "beating market expectations."
        ),
        "events": [
            {
                "event_type": "earnings",
                "evidence": (
                    "Nvidia delivered quarterly revenue of $22.1 billion, "
                    "surging 265% YoY."
                ),
            }
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(base_article)

    assert isinstance(processed, ProcessedNewsArticle)
    assert processed.article_id == "art-1001"
    assert processed.sentiment == "positive"
    assert "265%" in processed.sentiment_reasoning
    assert len(processed.important_events) == 1
    assert processed.important_events[0].event_type == "earnings"


def test_valid_negative_article():
    """Verify adverse corporate development classifies as negative."""
    article = NewsArticle(
        article_id="art-1002",
        ticker="XYZ",
        title="XYZ Corp Cuts Full-Year Revenue Guidance Amid Slumping Product Demand",
        summary=(
            "Company lowers annual forecast by 20% citing supply chain delays "
            "and cancellations."
        ),
        source="Bloomberg",
        published_at=datetime(2025, 2, 1, 10, 0, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "negative",
        "sentiment_reasoning": (
            "Company lowered annual revenue forecast by 20% due to slumping demand."
        ),
        "events": [
            {
                "event_type": "earnings",
                "evidence": (
                    "Company lowers annual forecast by 20% citing cancellations."
                ),
            }
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert processed.sentiment == "negative"
    assert "forecast" in processed.sentiment_reasoning.lower()


def test_valid_neutral_article():
    """Verify routine corporate announcement classifies as neutral."""
    article = NewsArticle(
        article_id="art-1003",
        ticker="ABC",
        title="ABC Corp Announces Date for Annual Shareholders Meeting",
        summary=(
            "The annual meeting of shareholders will take place virtually on May 12."
        ),
        source="PR Newswire",
        published_at=datetime(2025, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "neutral",
        "sentiment_reasoning": (
            "Routine scheduling announcement with no financial or operational impact."
        ),
        "events": [],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert processed.sentiment == "neutral"
    assert processed.important_events == []


def test_ambiguous_article_returns_neutral():
    """Verify mixed commentary or insufficient evidence classifies as neutral."""
    article = NewsArticle(
        article_id="art-1004",
        ticker="MSFT",
        title="Analysts Debate Cloud Market Dynamics Heading Into New Quarter",
        summary=(
            "Market participants express mixed views on enterprise IT budget pacing."
        ),
        source="Dow Jones",
        published_at=datetime(2025, 3, 5, 8, 30, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "neutral",
        "sentiment_reasoning": (
            "Mixed commentary from third-party observers with no reported metrics."
        ),
        "events": [],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert processed.sentiment == "neutral"
    assert processed.important_events == []


# ===========================================================================
# 2. Important Event Identification Tests (8.2.3)
# ===========================================================================


def test_earnings_event_detection(base_article: NewsArticle):
    """Verify earnings event is identified and structured with factual evidence."""
    mock_payload = {
        "sentiment": "positive",
        "sentiment_reasoning": "Record quarterly revenue reported.",
        "events": [
            {
                "event_type": "earnings",
                "evidence": "Reported quarterly revenue of $22.1 billion.",
            }
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(base_article)

    assert len(processed.important_events) == 1
    ev = processed.important_events[0]
    assert ev.event_type == "earnings"
    assert "$22.1 billion" in ev.evidence


def test_management_change_event_detection():
    """Verify leadership changes classify as management_change."""
    article = NewsArticle(
        article_id="art-1005",
        ticker="TECH",
        title=(
            "TechCorp Appoints Jane Doe as Chief Executive Officer Following "
            "Founder Retirement"
        ),
        summary=(
            "Jane Doe, former COO, has been named Chief Executive Officer "
            "effective immediately."
        ),
        source="Wall Street Journal",
        published_at=datetime(2025, 2, 10, 11, 0, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "neutral",
        "sentiment_reasoning": (
            "Leadership transition announced following planned retirement."
        ),
        "events": [
            {
                "event_type": "management_change",
                "evidence": "Jane Doe named Chief Executive Officer.",
            }
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert len(processed.important_events) == 1
    assert processed.important_events[0].event_type == "management_change"
    assert "Jane Doe" in processed.important_events[0].evidence


def test_regulatory_action_event_detection():
    """Verify regulatory probes or fines classify as regulatory_action."""
    article = NewsArticle(
        article_id="art-1006",
        ticker="BANK",
        title="Consumer Agency Fines MegaBank $150 Million Over Overdraft Practices",
        summary="Regulator orders bank to pay civil penalty and refund fees.",
        source="Associated Press",
        published_at=datetime(2025, 2, 12, 15, 0, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "negative",
        "sentiment_reasoning": (
            "Regulator imposed a $150M civil penalty for improper practices."
        ),
        "events": [
            {
                "event_type": "regulatory_action",
                "evidence": (
                    "Consumer agency fines MegaBank $150 million and orders "
                    "fee refunds."
                ),
            }
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert len(processed.important_events) == 1
    assert processed.important_events[0].event_type == "regulatory_action"
    assert "$150 million" in processed.important_events[0].evidence


def test_merger_acquisition_event_detection():
    """Verify buyout or acquisition news classifies as merger_acquisition."""
    article = NewsArticle(
        article_id="art-1007",
        ticker="ACQ",
        title="Global Systems to Acquire CloudNet in $5.2 Billion All-Cash Deal",
        summary="Boards of both firms have unanimously approved the merger agreement.",
        source="Reuters",
        published_at=datetime(2025, 2, 20, 13, 0, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "positive",
        "sentiment_reasoning": (
            "Definitive all-cash acquisition agreement reached at $5.2B valuation."
        ),
        "events": [
            {
                "event_type": "merger_acquisition",
                "evidence": (
                    "Global Systems to acquire CloudNet in $5.2 billion all-cash deal."
                ),
            }
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert len(processed.important_events) == 1
    assert processed.important_events[0].event_type == "merger_acquisition"
    assert "$5.2 billion" in processed.important_events[0].evidence


def test_article_with_no_important_event():
    """Verify standard product news contains zero important financial events."""
    article = NewsArticle(
        article_id="art-1008",
        ticker="AUTO",
        title="AutoMaker Unveils New Electric Sedan at Annual Auto Show",
        summary="The new model features an estimated 350-mile battery range.",
        source="Car & Driver",
        published_at=datetime(2025, 2, 25, 14, 0, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "neutral",
        "sentiment_reasoning": (
            "Product announcement without financial details or restructuring."
        ),
        "events": [],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert processed.important_events == []


def test_multiple_supported_events_on_single_article():
    """Verify article with earnings and executive change captures both events."""
    article = NewsArticle(
        article_id="art-1009",
        ticker="CORP",
        title="CORP Reports Q4 Loss and Announces CFO Resignation",
        summary=(
            "Company posted a fourth-quarter net loss of $50M and disclosed that CFO "
            "John Smith is stepping down immediately."
        ),
        source="Dow Jones",
        published_at=datetime(2025, 3, 1, 9, 0, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "negative",
        "sentiment_reasoning": (
            "Quarterly net loss of $50M accompanied by unexpected CFO resignation."
        ),
        "events": [
            {
                "event_type": "earnings",
                "evidence": "Posted fourth-quarter net loss of $50M.",
            },
            {
                "event_type": "management_change",
                "evidence": "CFO John Smith is stepping down immediately.",
            },
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert len(processed.important_events) == 2
    types = {ev.event_type for ev in processed.important_events}
    assert types == {"earnings", "management_change"}


# ===========================================================================
# 3. Missing Fields & Robustness Tests (8.2.1)
# ===========================================================================


def test_missing_summary_handling():
    """Verify article without summary preserves summary=None and prompts title."""
    article = NewsArticle(
        article_id="art-1010",
        ticker="NVDA",
        title="Nvidia Expands Cloud Computing Partnership with Major Provider",
        summary=None,
        source="Reuters",
        published_at=datetime(2025, 3, 2, 10, 0, 0, tzinfo=timezone.utc),
    )
    mock_payload = {
        "sentiment": "positive",
        "sentiment_reasoning": (
            "Expanded strategic cloud partnership reported in headline."
        ),
        "events": [],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(article)

    assert processed.summary is None
    assert processed.headline == article.title
    assert "No summary provided" in provider.prompts_received[0]


def test_missing_or_empty_article_title_raises():
    """Verify article with empty/whitespace title is rejected with ValueError."""
    with pytest.raises(ValueError, match="title cannot be empty"):
        NewsArticle(
            article_id="art-invalid",
            title="   ",
            source="Source",
            published_at=datetime.now(timezone.utc),
        )


def test_malformed_structured_llm_output_raises():
    """Verify unparseable LLM output triggers LLMStructuredOutputError."""
    article = NewsArticle(
        article_id="art-1011",
        title="Some News Title",
        source="Source",
        published_at=datetime.now(timezone.utc),
    )
    # Return non-JSON text that cannot be parsed
    provider = MockNewsLLMProvider(responses=["NOT_JSON_AT_ALL{"])
    processor = NewsProcessor(llm_provider=provider)

    with pytest.raises(LLMStructuredOutputError):
        processor.process_article(article)


def test_unsupported_sentiment_value_rejected():
    """Verify model output with invalid sentiment raises structured validation error."""
    article = NewsArticle(
        article_id="art-1012",
        title="Headline",
        source="Source",
        published_at=datetime.now(timezone.utc),
    )
    mock_payload = {
        "sentiment": "bullish",  # Invalid; only positive, negative, neutral
        "sentiment_reasoning": "Strong indicators.",
        "events": [],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    with pytest.raises(LLMStructuredOutputError):
        processor.process_article(article)


def test_unsupported_event_type_rejected():
    """Verify model output with unsupported event type raises validation error."""
    article = NewsArticle(
        article_id="art-1013",
        title="Headline",
        source="Source",
        published_at=datetime.now(timezone.utc),
    )
    mock_payload = {
        "sentiment": "neutral",
        "sentiment_reasoning": "Routine announcement.",
        "events": [
            {
                "event_type": "product_launch",  # Not in the 4 allowed categories
                "evidence": "Unveiled new product line.",
            }
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    with pytest.raises(LLMStructuredOutputError):
        processor.process_article(article)


# ===========================================================================
# 4. Grounding, Preservation & Safety Tests
# ===========================================================================


def test_evidence_remains_grounded(base_article: NewsArticle):
    """Verify evidence string synthesizes sentiment reasoning and event citations."""
    mock_payload = {
        "sentiment": "positive",
        "sentiment_reasoning": "Record data center sales reported.",
        "events": [
            {
                "event_type": "earnings",
                "evidence": "Data center revenue up 409% to $18.4 billion.",
            }
        ],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(base_article)

    assert processed.evidence is not None
    assert "Record data center sales reported." in processed.evidence
    assert (
        "[earnings] Data center revenue up 409% to $18.4 billion." in processed.evidence
    )


def test_article_id_preserved(base_article: NewsArticle):
    """Verify input article_id is identically preserved in ProcessedNewsArticle."""
    provider = MockNewsLLMProvider()
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(base_article)

    assert processed.article_id == base_article.article_id


def test_source_url_and_published_at_preserved(base_article: NewsArticle):
    """Verify source, url, and published_at are preserved without drift."""
    provider = MockNewsLLMProvider()
    processor = NewsProcessor(llm_provider=provider)

    processed = processor.process_article(base_article)

    assert processed.source == base_article.source
    assert processed.url == str(base_article.url)
    assert processed.published_at == base_article.published_at


def test_prohibited_recommendation_language_rejected(base_article: NewsArticle):
    """Verify model output attempting to offer buy/sell/hold ratings is rejected."""
    mock_payload = {
        "sentiment": "positive",
        "sentiment_reasoning": (
            "Great results, strong buy rating recommended for investors."
        ),
        "events": [],
    }
    provider = MockNewsLLMProvider(responses=[mock_payload])
    processor = NewsProcessor(llm_provider=provider)

    with pytest.raises(ValueError, match="Prohibited advice or recommendation phrase"):
        processor.process_article(base_article)


# ===========================================================================
# 5. Batch Processing Tests
# ===========================================================================


def test_batch_processing_one_failure_does_not_corrupt_batch(
    base_article: NewsArticle,
):
    """Verify failure on one article falls back without aborting remaining batch."""
    art1 = base_article
    art2 = NewsArticle(
        article_id="art-2002",
        ticker="NVDA",
        title="Nvidia Hardware Update",
        summary="General summary text.",
        source="Wire",
        published_at=datetime.now(timezone.utc),
    )
    art3 = NewsArticle(
        article_id="art-2003",
        ticker="NVDA",
        title="Nvidia Announces Enterprise Cloud Deployment",
        summary="Enterprise customers deploy new clusters.",
        source="Bloomberg",
        published_at=datetime.now(timezone.utc),
    )

    call_num = 0

    def responder(prompt: str) -> str:
        nonlocal call_num
        call_num += 1
        # Fail the second article
        if "art-2002" in prompt:
            return "NOT_VALID_JSON{{"
        return json.dumps(
            {
                "sentiment": "positive",
                "sentiment_reasoning": "Operational milestone reached.",
                "events": [],
            }
        )

    provider = MockNewsLLMProvider(responder=responder)
    processor = NewsProcessor(llm_provider=provider)

    batch_result = processor.process_batch([art1, art2, art3])

    assert batch_result.total_processed == 3
    assert batch_result.successful_count == 2
    assert batch_result.failed_count == 1
    assert len(batch_result.articles) == 3

    # Art 1 is successfully classified
    assert batch_result.articles[0].article_id == "art-1001"
    assert batch_result.articles[0].sentiment == "positive"
    assert batch_result.articles[0].is_classified is True
    assert batch_result.articles[0].processing_error is None

    # Art 2 safely fell back to neutral and is flagged as unclassified failure
    assert batch_result.articles[1].article_id == "art-2002"
    assert batch_result.articles[1].sentiment == "neutral"
    assert batch_result.articles[1].is_classified is False
    assert batch_result.articles[1].processing_error is not None
    assert "defaulted to neutral" in batch_result.articles[1].sentiment_reasoning

    # Art 3 is successfully classified
    assert batch_result.articles[2].article_id == "art-2003"
    assert batch_result.articles[2].sentiment == "positive"
    assert batch_result.articles[2].is_classified is True
    assert batch_result.articles[2].processing_error is None


def test_batch_processing_deterministic_ordering(base_article: NewsArticle):
    """Verify batch output retains strict input ordering."""
    articles = [
        base_article,
        NewsArticle(
            article_id="art-batch-2",
            title="Second Headline",
            source="Source",
            published_at=datetime.now(timezone.utc),
        ),
        NewsArticle(
            article_id="art-batch-3",
            title="Third Headline",
            source="Source",
            published_at=datetime.now(timezone.utc),
        ),
    ]
    provider = MockNewsLLMProvider()
    processor = NewsProcessor(llm_provider=provider)

    result = processor.process_batch(articles)

    assert [a.article_id for a in result.articles] == [
        "art-1001",
        "art-batch-2",
        "art-batch-3",
    ]


def test_failed_classification_distinguishable_from_genuine_neutral():
    """Verify processing failure is not mistaken for a genuine neutral signal."""
    art_fail = NewsArticle(
        article_id="art-fail",
        title="Breaking News",
        source="Wire",
        published_at=datetime.now(timezone.utc),
    )
    provider = MockNewsLLMProvider(fail_with=RuntimeError("API error"))
    processor = NewsProcessor(llm_provider=provider)

    batch_result = processor.process_batch([art_fail])

    article = batch_result.articles[0]
    # State is unambiguously marked as unclassified
    assert article.is_classified is False
    assert article.processing_error is not None
    assert "API error" in article.processing_error
    # Downstream consumers can filter out unclassified articles
    genuine_neutral = [
        a for a in batch_result.articles if a.is_classified and a.sentiment == "neutral"
    ]
    assert len(genuine_neutral) == 0
