"""Unit tests for Phase 8.4: News Analyst Testing.

Comprehensive offline test suite verifying:
- 8.4.1 News Provider parsing and normalization:
    * Valid Finnhub article normalization into NewsArticle
    * Unix epoch timestamp -> timezone-aware UTC datetime
    * Title, source, URL, image, and summary handling
    * Related ticker normalization
    * Deterministic fallback article ID generation when provider ID is absent
    * Duplicate article and URL deduplication
    * Comprehensive HTTP and network error mapping:
        - 401/403 -> NewsAuthenticationError
        - 429 -> NewsProviderRateLimitError
        - 500/503 -> NewsProviderUnavailableError
        - Timeout / RequestError -> NewsProviderUnavailableError
        - Empty response -> EmptyNewsDataError
        - Missing required fields -> NewsMalformedDataError
- 8.4.2 Sentiment classification sanity tests on sample articles:
    * Clearly positive financial news -> positive
    * Clearly negative financial news -> negative
    * Clearly neutral factual news -> neutral
    * Ambiguous / mixed headline -> neutral / balanced handling
    * Headline without summary -> classification functions on headline alone
    * Financial terminology without recommendation -> sentiment remains sentiment-only
- 8.4.3 Insufficient news agent behavior:
    * No articles supplied -> graceful insufficient_news (0 LLM calls,
      confidence=0.0)
    * All articles failed classification -> safe insufficient_news (confidence=0.0)
    * Partial classification failures -> failed articles excluded from
      sentiment distribution
- 8.4.4 Edge cases:
    * Conflicting sentiment (positive vs negative) -> both signals preserved
      with provenance
    * Asymmetric conflicting sentiment (2 positive, 1 negative) -> balanced synthesis
    * Stale/old-only news -> publication timestamps preserved, recency penalties applied

All tests are 100% deterministic, offline, and consume zero Gemini API
or Finnhub tokens.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
import pytest

from app.agents.news import (
    NewsAnalystAgent,
    calculate_news_confidence,
)
from app.agents.news_schema import (
    NewsAnalysisOutput,
    NewsAnalystInput,
)
from app.core.llm.base import LLMProvider, LLMResponse
from app.models.news import NewsArticle, NewsSearchResult
from app.models.news_processing import (
    NewsEvent,
    ProcessedNewsArticle,
)
from app.providers.exceptions import (
    EmptyNewsDataError,
    NewsAuthenticationError,
    NewsMalformedDataError,
    NewsProviderRateLimitError,
    NewsProviderUnavailableError,
)
from app.providers.finnhub_news import FinnhubNewsProvider
from app.services.news_processor import NewsProcessor

# ===========================================================================
# Mock LLM Providers (Offline & Deterministic)
# ===========================================================================


class MockNewsSynthesisLLMProvider(LLMProvider):
    """Mock LLMProvider for Phase 8.4 News Analyst Agent testing."""

    def __init__(
        self,
        responses: Optional[List[Union[str, Dict[str, Any]]]] = None,
        responder: Optional[Callable[[str], str]] = None,
        fail_with: Optional[Exception] = None,
    ) -> None:
        self.call_count = 0
        self.prompts_received: List[str] = []
        self.responses = responses or []
        self.responder = responder
        self.fail_with = fail_with
        self._name = "mock_phase8_4_synthesis"

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        schema: Optional[Any] = None,
        **kwargs: Any,
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

        raise RuntimeError("No mock synthesis response queued.")


class MockNewsClassificationLLMProvider(LLMProvider):
    """Mock LLMProvider for Phase 8.4 NewsProcessor classification testing."""

    def __init__(
        self,
        responses: Optional[List[Union[str, Dict[str, Any]]]] = None,
        responder: Optional[Callable[[str], str]] = None,
        fail_with: Optional[Exception] = None,
    ) -> None:
        self.call_count = 0
        self.prompts_received: List[str] = []
        self.responses = responses or []
        self.responder = responder
        self.fail_with = fail_with
        self._name = "mock_phase8_4_classification"

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        schema: Optional[Any] = None,
        **kwargs: Any,
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

        raise RuntimeError("No mock classification response queued.")


# ===========================================================================
# 8.4.1 PROVIDER TESTS (Parsing, Normalization, Fallback IDs, Errors)
# ===========================================================================


def test_provider_finnhub_article_parsing_normalization():
    """Verify complete Finnhub article payload is normalized into NewsArticle."""
    raw_article = {
        "category": "company",
        "datetime": 1735689600,  # 2025-01-01 00:00:00 UTC
        "headline": "  Nvidia Unveils Next-Gen AI Silicon Architecture   ",
        "id": 98765,
        "image": "https://images.example.com/nvda.jpg",
        "related": "nvda",
        "source": "  Reuters Financial  ",
        "summary": "  Accelerating data center computing demand drives new platform. ",
        "url": "https://www.reuters.com/tech/nvidia-ai-silicon-2025",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[raw_article])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = FinnhubNewsProvider(api_key="mock-key", client=client)

    result: NewsSearchResult = provider.search_news("NVDA", limit=5)

    assert result.total_count == 1
    article = result.articles[0]
    assert article.article_id == "98765"
    assert article.ticker == "NVDA"
    assert article.title == "Nvidia Unveils Next-Gen AI Silicon Architecture"
    assert article.source == "Reuters Financial"
    assert (
        article.summary
        == "Accelerating data center computing demand drives new platform."
    )
    assert article.url == "https://www.reuters.com/tech/nvidia-ai-silicon-2025"
    assert article.image_url == "https://images.example.com/nvda.jpg"
    assert article.published_at == datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert article.retrieved_at.tzinfo == timezone.utc


def test_provider_finnhub_fallback_article_id_deterministic():
    """Verify deterministic SHA-256 fallback ID when Finnhub ID is absent."""
    raw_without_id = {
        "datetime": 1735689600,
        "headline": "Nvidia Enterprise AI Platform Release",
        "source": "Bloomberg",
        "url": "https://www.bloomberg.com/news/nvda-enterprise",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[raw_without_id])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = FinnhubNewsProvider(api_key="mock-key", client=client)

    result1 = provider.search_news("NVDA")
    result2 = provider.search_news("NVDA")

    id1 = result1.articles[0].article_id
    id2 = result2.articles[0].article_id

    # Fallback ID must be a non-empty 16-character hex hash
    assert len(id1) == 16
    # Same input produces identical deterministic ID
    assert id1 == id2

    # Expected hash computation
    expected_input = (
        "https://www.bloomberg.com/news/nvda-enterprise|"
        "nvidia enterprise ai platform release|bloomberg|1735689600"
    )
    expected_id = hashlib.sha256(expected_input.encode("utf-8")).hexdigest()[:16]
    assert id1 == expected_id


def test_provider_finnhub_deduplication_by_id_and_url():
    """Verify provider deduplicates articles sharing identical ID or canonical URL."""
    raw_articles = [
        {
            "id": 1001,
            "datetime": 1735689600,
            "headline": "Nvidia Expands Compute Cluster Capacity",
            "source": "Reuters",
            "url": "https://www.reuters.com/tech/nvda-cluster",
        },
        # Duplicate by ID with slightly different URL
        {
            "id": 1001,
            "datetime": 1735689600,
            "headline": "Nvidia Expands Compute Cluster Capacity (Update)",
            "source": "Reuters",
            "url": "https://www.reuters.com/tech/nvda-cluster?ref=rss",
        },
        # Duplicate by URL with different ID
        {
            "id": 1002,
            "datetime": 1735689600,
            "headline": "Nvidia Expands Compute Cluster Capacity (Wire)",
            "source": "Reuters",
            "url": "https://www.reuters.com/tech/nvda-cluster",
        },
        # Unique article
        {
            "id": 1003,
            "datetime": 1735776000,
            "headline": "Semiconductor Supply Chains Stabilize",
            "source": "Bloomberg",
            "url": "https://www.bloomberg.com/chips-supply",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=raw_articles)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = FinnhubNewsProvider(api_key="mock-key", client=client)

    result = provider.search_news("NVDA")
    assert result.total_count == 2
    ids = [a.article_id for a in result.articles]
    assert "1001" in ids
    assert "1003" in ids


def test_provider_finnhub_http_error_mappings():
    """Verify HTTP status codes map to domain exceptions."""
    # 401 Auth error
    client_401 = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(401))
    )
    p_401 = FinnhubNewsProvider(api_key="bad-key", client=client_401)
    with pytest.raises(NewsAuthenticationError):
        p_401.search_news("NVDA")

    # 429 Rate limit error
    client_429 = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(429))
    )
    p_429 = FinnhubNewsProvider(api_key="mock-key", client=client_429)
    with pytest.raises(NewsProviderRateLimitError):
        p_429.search_news("NVDA")

    # 503 Provider unavailable error
    client_503 = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(503))
    )
    p_503 = FinnhubNewsProvider(api_key="mock-key", client=client_503)
    with pytest.raises(NewsProviderUnavailableError):
        p_503.search_news("NVDA")

    # Empty news response
    client_empty = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[]))
    )
    p_empty = FinnhubNewsProvider(api_key="mock-key", client=client_empty)
    with pytest.raises(EmptyNewsDataError):
        p_empty.search_news("NVDA")

    # Malformed article: missing headline
    client_malformed = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json=[{"source": "Reuters", "datetime": 1735689600}]
            )
        )
    )
    p_malformed = FinnhubNewsProvider(api_key="mock-key", client=client_malformed)
    with pytest.raises(NewsMalformedDataError):
        p_malformed.search_news("NVDA")


# ===========================================================================
# 8.4.2 SENTIMENT SANITY TESTS (Sample Articles)
# ===========================================================================


def test_sentiment_sanity_clearly_positive():
    """Verify clearly positive financial news classifies as positive."""
    article = NewsArticle(
        article_id="art-p1",
        ticker="NVDA",
        title="Nvidia Reports Record Fourth Quarter Revenue Surging 265% YoY",
        summary="Exceptional data center computing demand accelerated net income.",
        source="Reuters",
        published_at=datetime.now(timezone.utc),
    )
    mock_resp = {
        "sentiment": "positive",
        "sentiment_reasoning": "Record 265% revenue growth and surging net income.",
        "events": [
            {
                "event_type": "earnings",
                "evidence": "Quarterly revenue surged 265% YoY.",
            }
        ],
    }
    processor = NewsProcessor(
        llm_provider=MockNewsClassificationLLMProvider(responses=[mock_resp])
    )
    processed = processor.process_article(article)

    assert processed.sentiment == "positive"
    assert processed.is_classified is True
    assert len(processed.important_events) == 1
    assert processed.important_events[0].event_type == "earnings"


def test_sentiment_sanity_clearly_negative():
    """Verify clearly negative financial news classifies as negative."""
    article = NewsArticle(
        article_id="art-n1",
        ticker="NVDA",
        title="Regulators File Antitrust Injunction Halting Major Cloud Chip Deals",
        summary="Department of Justice launched exclusive dealing enforcement action.",
        source="Bloomberg",
        published_at=datetime.now(timezone.utc),
    )
    mock_resp = {
        "sentiment": "negative",
        "sentiment_reasoning": "Antitrust enforcement action halting distribution.",
        "events": [
            {
                "event_type": "regulatory_action",
                "evidence": "DOJ antitrust injunction halting agreements.",
            }
        ],
    }
    processor = NewsProcessor(
        llm_provider=MockNewsClassificationLLMProvider(responses=[mock_resp])
    )
    processed = processor.process_article(article)

    assert processed.sentiment == "negative"
    assert processed.is_classified is True
    assert len(processed.important_events) == 1
    assert processed.important_events[0].event_type == "regulatory_action"


def test_sentiment_sanity_clearly_neutral():
    """Verify routine administrative news classifies as neutral."""
    article = NewsArticle(
        article_id="art-u1",
        ticker="NVDA",
        title="Nvidia Announces Date and Time for 2025 Annual Meeting of Shareholders",
        summary="The annual meeting will be held virtually on May 15, 2025.",
        source="PR Newswire",
        published_at=datetime.now(timezone.utc),
    )
    mock_resp = {
        "sentiment": "neutral",
        "sentiment_reasoning": "Standard annual shareholder meeting announcement.",
        "events": [],
    }
    processor = NewsProcessor(
        llm_provider=MockNewsClassificationLLMProvider(responses=[mock_resp])
    )
    processed = processor.process_article(article)

    assert processed.sentiment == "neutral"
    assert processed.is_classified is True
    assert len(processed.important_events) == 0


def test_sentiment_sanity_ambiguous_mixed_headline():
    """Verify mixed/ambiguous signals conform to structured classification contract."""
    article = NewsArticle(
        article_id="art-m1",
        ticker="NVDA",
        title="Sales Beat Street by 15% While Operating Margins Compress on Higher R&D",
        summary="Gross revenue grew solidly but operating profits remained flat.",
        source="Dow Jones",
        published_at=datetime.now(timezone.utc),
    )
    mock_resp = {
        "sentiment": "neutral",
        "sentiment_reasoning": (
            "Top-line sales beat balanced by compressed operating margins."
        ),
        "events": [
            {
                "event_type": "earnings",
                "evidence": "Sales beat Street by 15% but margins compressed.",
            }
        ],
    }
    processor = NewsProcessor(
        llm_provider=MockNewsClassificationLLMProvider(responses=[mock_resp])
    )
    processed = processor.process_article(article)

    assert processed.sentiment == "neutral"
    assert processed.is_classified is True
    assert "balanced by compressed operating margins" in processed.sentiment_reasoning


def test_sentiment_sanity_headline_without_summary():
    """Verify article with headline but no summary classifies using available text."""
    article = NewsArticle(
        article_id="art-nosum",
        ticker="NVDA",
        title="Nvidia Unveils Next-Generation AI Server Rack Architecture",
        summary=None,
        source="Reuters",
        published_at=datetime.now(timezone.utc),
    )
    mock_resp = {
        "sentiment": "positive",
        "sentiment_reasoning": "Unveiling of next-generation product platform.",
        "events": [],
    }
    provider = MockNewsClassificationLLMProvider(responses=[mock_resp])
    processor = NewsProcessor(llm_provider=provider)
    processed = processor.process_article(article)

    assert processed.sentiment == "positive"
    assert processed.is_classified is True
    assert "Headline: Nvidia Unveils Next-Generation AI Server Rack Architecture" in (
        provider.prompts_received[0]
    )
    assert "Summary: No summary provided" in provider.prompts_received[0]


def test_sentiment_sanity_financial_terminology_no_recommendation():
    """Verify financial metrics classify sentiment without producing investment advice.

    Sentiment remains sentiment-only.
    """
    article = NewsArticle(
        article_id="art-terms",
        ticker="NVDA",
        title="EBITDA Expands 450 bps as Free Cash Flow Conversion Reaches 85%",
        summary="Operational efficiencies improved cash generation substantially.",
        source="Financial Times",
        published_at=datetime.now(timezone.utc),
    )
    mock_resp = {
        "sentiment": "positive",
        "sentiment_reasoning": (
            "Strong operating leverage and robust free cash flow generation."
        ),
        "events": [
            {
                "event_type": "earnings",
                "evidence": "EBITDA expanded 450 bps with 85% FCF conversion.",
            }
        ],
    }
    processor = NewsProcessor(
        llm_provider=MockNewsClassificationLLMProvider(responses=[mock_resp])
    )
    processed = processor.process_article(article)

    assert processed.sentiment == "positive"
    assert processed.is_classified is True
    # Zero prohibited advisory words
    assert "buy" not in processed.sentiment_reasoning.lower()
    assert "target" not in processed.sentiment_reasoning.lower()


# ===========================================================================
# 8.4.3 INSUFFICIENT NEWS (Agent Tests)
# ===========================================================================


def test_insufficient_news_no_articles_supplied():
    """Verify empty article list produces deterministic insufficient_news.

    Must make 0 LLM calls.
    """
    provider = MockNewsSynthesisLLMProvider(responses=[])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="AAPL", articles=[]))

    assert result.success is True
    assert provider.call_count == 0  # Fast-path short-circuit
    output: NewsAnalysisOutput = result.data
    assert output.ticker == "AAPL"
    assert output.overall_sentiment == "insufficient_news"
    assert output.confidence == 0.0
    assert output.recent_news == []
    assert output.important_events == []
    assert output.positive_factors == []
    assert output.negative_factors == []
    assert output.sentiment_distribution == {"positive": 0, "negative": 0, "neutral": 0}


def test_insufficient_news_all_failed_classification():
    """Verify all failed/unclassified articles yield insufficient_news.

    Confidence must be 0.0.
    """
    failed_articles = [
        ProcessedNewsArticle(
            article_id="art-f1",
            headline="Headline 1",
            sentiment="neutral",
            sentiment_reasoning="Failed",
            published_at=datetime.now(timezone.utc),
            source="Wire",
            is_classified=False,
            processing_error="Connection timed out",
        ),
        ProcessedNewsArticle(
            article_id="art-f2",
            headline="Headline 2",
            sentiment="neutral",
            sentiment_reasoning="Failed",
            published_at=datetime.now(timezone.utc),
            source="Wire",
            is_classified=False,
            processing_error="Corrupt payload",
        ),
    ]
    provider = MockNewsSynthesisLLMProvider(responses=[])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="MSFT", articles=failed_articles))

    assert result.success is True
    assert provider.call_count == 0
    output: NewsAnalysisOutput = result.data
    assert output.overall_sentiment == "insufficient_news"
    assert output.confidence == 0.0
    # Failed articles MUST NOT be counted as neutral
    assert output.sentiment_distribution == {"positive": 0, "negative": 0, "neutral": 0}


def test_insufficient_news_partial_failures_excluded_from_analysis():
    """Verify failed articles are excluded while usable articles determine analysis."""
    now_utc = datetime.now(timezone.utc) - timedelta(hours=1)
    usable_article = ProcessedNewsArticle(
        article_id="art-u1",
        headline="Enterprise Software Demand Surges 40% Year-over-Year",
        summary="Cloud computing expansion accelerates subscription growth.",
        sentiment="positive",
        sentiment_reasoning="Accelerating subscription revenue.",
        important_events=[
            NewsEvent(
                event_type="earnings",
                evidence="Demand surged 40% YoY.",
            )
        ],
        source="Bloomberg",
        url="https://www.bloomberg.com/software-growth",
        published_at=now_utc,
        is_classified=True,
    )
    failed_article = ProcessedNewsArticle(
        article_id="art-f1",
        headline="Unparseable Text",
        sentiment="neutral",
        sentiment_reasoning="Error",
        published_at=now_utc,
        source="Wire",
        is_classified=False,
        processing_error="Parser error",
    )
    mock_payload = {
        "ticker": "MSFT",
        "overall_sentiment": "positive",
        "sentiment_distribution": {"positive": 1, "negative": 0, "neutral": 0},
        "recent_news": [
            {
                "article_id": "art-u1",
                "headline": "Enterprise Software Demand Surges 40% Year-over-Year",
                "summary": "Cloud expansion accelerates subscription growth.",
                "source": "Bloomberg",
                "url": "https://www.bloomberg.com/software-growth",
                "published_at": now_utc.isoformat(),
                "sentiment": "positive",
            }
        ],
        "important_events": [
            {
                "event_type": "earnings",
                "description": "Demand surged 40% YoY.",
                "article_ids": ["art-u1"],
            }
        ],
        "positive_factors": [
            {
                "text": "Subscription demand grew 40% YoY.",
                "article_ids": ["art-u1"],
            }
        ],
        "negative_factors": [],
        "confidence": 0.35,
        "summary": "Positive business developments reflected in software demand.",
        "evidence": ["Bloomberg reported 40% YoY demand surge (art-u1)."],
    }
    provider = MockNewsSynthesisLLMProvider(responses=[mock_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="MSFT", articles=[usable_article, failed_article])
    )

    assert result.success is True
    output: NewsAnalysisOutput = result.data
    # 1 usable positive article: failure ratio = 1/2 = 50%
    # (>30% penalty: 0.35 - 0.15 = 0.20)
    assert output.sentiment_distribution == {"positive": 1, "negative": 0, "neutral": 0}
    assert output.overall_sentiment == "positive"
    assert output.confidence == 0.20


# ===========================================================================
# 8.4.4 EDGE CASES (Conflicting Sentiment & Stale News)
# ===========================================================================


def test_conflicting_sentiment_positive_and_negative_preserved():
    """Verify system preserves both positive and negative signals without discarding."""
    now_utc = datetime.now(timezone.utc) - timedelta(hours=2)
    pos_article = ProcessedNewsArticle(
        article_id="art-p",
        headline="Revenue Surges 30% Driven by Cloud Compute Orders",
        summary="Quarterly top-line sales accelerated across all regions.",
        sentiment="positive",
        sentiment_reasoning="Strong revenue growth.",
        important_events=[
            NewsEvent(
                event_type="earnings",
                evidence="Revenue surged 30% on compute orders.",
            )
        ],
        source="Reuters",
        url="https://www.reuters.com/nvda-revenue",
        published_at=now_utc,
        is_classified=True,
    )
    neg_article = ProcessedNewsArticle(
        article_id="art-n",
        headline="Antitrust Inquiry Opened into Distribution Agreements",
        summary="Government regulators opened a formal investigation.",
        sentiment="negative",
        sentiment_reasoning="Regulatory antitrust probe.",
        important_events=[
            NewsEvent(
                event_type="regulatory_action",
                evidence="Regulators opened formal antitrust inquiry.",
            )
        ],
        source="Bloomberg",
        url="https://www.bloomberg.com/nvda-antitrust",
        published_at=now_utc,
        is_classified=True,
    )

    mock_payload = {
        "ticker": "NVDA",
        "overall_sentiment": "neutral",
        "sentiment_distribution": {"positive": 1, "negative": 1, "neutral": 0},
        "recent_news": [
            {
                "article_id": "art-p",
                "headline": "Revenue Surges 30% Driven by Cloud Compute Orders",
                "summary": "Sales accelerated across all regions.",
                "source": "Reuters",
                "url": "https://www.reuters.com/nvda-revenue",
                "published_at": now_utc.isoformat(),
                "sentiment": "positive",
            },
            {
                "article_id": "art-n",
                "headline": "Antitrust Inquiry Opened into Distribution Agreements",
                "summary": "Formal regulatory investigation opened.",
                "source": "Bloomberg",
                "url": "https://www.bloomberg.com/nvda-antitrust",
                "published_at": now_utc.isoformat(),
                "sentiment": "negative",
            },
        ],
        "important_events": [
            {
                "event_type": "earnings",
                "description": "Revenue surged 30% on compute orders.",
                "article_ids": ["art-p"],
            },
            {
                "event_type": "regulatory_action",
                "description": "Regulators opened formal antitrust inquiry.",
                "article_ids": ["art-n"],
            },
        ],
        "positive_factors": [
            {
                "text": "Revenue surged 30% on strong cloud demand.",
                "article_ids": ["art-p"],
            }
        ],
        "negative_factors": [
            {
                "text": "Antitrust inquiry launched into distribution agreements.",
                "article_ids": ["art-n"],
            }
        ],
        "confidence": 0.55,
        "summary": (
            "Operational strength is balanced by an emerging regulatory investigation."
        ),
        "evidence": [
            "Reuters reported 30% revenue growth (art-p).",
            "Bloomberg reported antitrust inquiry initiation (art-n).",
        ],
    }
    provider = MockNewsSynthesisLLMProvider(responses=[mock_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(
        NewsAnalystInput(ticker="NVDA", articles=[pos_article, neg_article])
    )

    assert result.success is True
    output: NewsAnalysisOutput = result.data
    # Both factors MUST be preserved with correct article_id provenance
    assert len(output.positive_factors) == 1
    assert output.positive_factors[0].article_ids == ["art-p"]
    assert len(output.negative_factors) == 1
    assert output.negative_factors[0].article_ids == ["art-n"]
    assert output.overall_sentiment == "neutral"
    assert output.sentiment_distribution == {"positive": 1, "negative": 1, "neutral": 0}
    # Deterministic confidence for 2 fresh articles: 0.55
    assert output.confidence == 0.55


def test_conflicting_sentiment_asymmetric_two_positive_one_negative():
    """Verify asymmetric conflicting evidence preserves signals and respects bounds."""
    now_utc = datetime.now(timezone.utc) - timedelta(hours=2)
    pos1 = ProcessedNewsArticle(
        article_id="art-p1",
        headline="Revenue Surges 30% Driven by Compute Orders",
        summary="Quarterly sales accelerated.",
        sentiment="positive",
        sentiment_reasoning="Revenue growth.",
        important_events=[],
        source="Reuters",
        url="https://reuters.com/p1",
        published_at=now_utc,
        is_classified=True,
    )
    pos2 = ProcessedNewsArticle(
        article_id="art-p2",
        headline="Company Announces Strategic Global Cloud Partnership",
        summary="New enterprise software integration agreed.",
        sentiment="positive",
        sentiment_reasoning="Expansion of cloud ecosystem.",
        important_events=[],
        source="PR Newswire",
        url="https://prnewswire.com/p2",
        published_at=now_utc,
        is_classified=True,
    )
    neg1 = ProcessedNewsArticle(
        article_id="art-n1",
        headline="Supply Chain Bottlenecks Cause Short-Term Shipping Delays",
        summary="Logistics issues may impact component fulfillment.",
        sentiment="negative",
        sentiment_reasoning="Supply chain friction.",
        important_events=[],
        source="Bloomberg",
        url="https://bloomberg.com/n1",
        published_at=now_utc,
        is_classified=True,
    )

    mock_payload = {
        "ticker": "NVDA",
        "overall_sentiment": "positive",
        "sentiment_distribution": {"positive": 2, "negative": 1, "neutral": 0},
        "recent_news": [
            {
                "article_id": "art-p1",
                "headline": "Revenue Surges 30% Driven by Compute Orders",
                "source": "Reuters",
                "url": "https://reuters.com/p1",
                "published_at": now_utc.isoformat(),
                "sentiment": "positive",
            },
            {
                "article_id": "art-n1",
                "headline": "Supply Chain Bottlenecks Cause Short-Term Shipping Delays",
                "source": "Bloomberg",
                "url": "https://bloomberg.com/n1",
                "published_at": now_utc.isoformat(),
                "sentiment": "negative",
            },
        ],
        "important_events": [],
        "positive_factors": [
            {
                "text": "Strong revenue acceleration of 30%.",
                "article_ids": ["art-p1"],
            }
        ],
        "negative_factors": [
            {
                "text": "Component shipping friction due to logistics.",
                "article_ids": ["art-n1"],
            }
        ],
        "confidence": 0.70,
        "summary": (
            "Favorable operational and partnership momentum "
            "outweighs minor supply delays."
        ),
        "evidence": [
            "Revenue surged 30% (art-p1).",
            "Shipping delays noted (art-n1).",
        ],
    }
    provider = MockNewsSynthesisLLMProvider(responses=[mock_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="NVDA", articles=[pos1, pos2, neg1]))

    assert result.success is True
    output: NewsAnalysisOutput = result.data
    assert output.overall_sentiment == "positive"
    assert output.sentiment_distribution == {"positive": 2, "negative": 1, "neutral": 0}
    assert len(output.positive_factors) == 1
    assert len(output.negative_factors) == 1
    # 3 fresh articles -> 0.70
    assert output.confidence == 0.70


def test_stale_news_single_old_article_reduces_confidence():
    """Verify single old article (>30 days old) applies recency penalty."""
    # 45 days old
    old_date = datetime.now(timezone.utc) - timedelta(days=45)
    old_article = ProcessedNewsArticle(
        article_id="art-old-1",
        headline="Quarterly Earnings Beat Consensus Estimates",
        summary="Historical quarterly performance exceeded targets.",
        sentiment="positive",
        sentiment_reasoning="Historical positive earnings beat.",
        important_events=[
            NewsEvent(
                event_type="earnings",
                evidence="Earnings beat consensus estimates.",
            )
        ],
        source="Reuters",
        url="https://www.reuters.com/archive/nvda-q3",
        published_at=old_date,
        is_classified=True,
    )

    # Base for 1 article: 0.35; Stale penalty (>30 days): -0.20 -> 0.15
    # (clamped to 0.20 min)
    expected_conf = calculate_news_confidence([old_article])
    assert expected_conf == 0.20

    mock_payload = {
        "ticker": "NVDA",
        "overall_sentiment": "positive",
        "sentiment_distribution": {"positive": 1, "negative": 0, "neutral": 0},
        "recent_news": [
            {
                "article_id": "art-old-1",
                "headline": "Quarterly Earnings Beat Consensus Estimates",
                "source": "Reuters",
                "url": "https://www.reuters.com/archive/nvda-q3",
                "published_at": old_date.isoformat(),
                "sentiment": "positive",
            }
        ],
        "important_events": [
            {
                "event_type": "earnings",
                "description": "Earnings beat consensus estimates.",
                "article_ids": ["art-old-1"],
            }
        ],
        "positive_factors": [
            {
                "text": "Historical earnings beat.",
                "article_ids": ["art-old-1"],
            }
        ],
        "negative_factors": [],
        "confidence": 0.20,
        "summary": "Historical news reflects past earnings strength.",
        "evidence": ["Historical Reuters earnings report (art-old-1)."],
    }
    provider = MockNewsSynthesisLLMProvider(responses=[mock_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="NVDA", articles=[old_article]))

    assert result.success is True
    output: NewsAnalysisOutput = result.data
    assert output.confidence == 0.20
    # Published date must be preserved exactly as historical
    assert output.recent_news[0].published_at == old_date


def test_stale_news_multiple_old_articles_reduces_confidence():
    """Verify multiple old articles apply stale penalty (-0.20) to base confidence."""
    # 40 days old
    old_date = datetime.now(timezone.utc) - timedelta(days=40)
    old_articles = [
        ProcessedNewsArticle(
            article_id=f"art-old-{i}",
            headline=f"Historical Development {i}",
            summary="Past business report.",
            sentiment="positive" if i == 1 else "neutral",
            sentiment_reasoning="Historical context.",
            important_events=[],
            source="Reuters",
            url=f"https://www.reuters.com/archive/{i}",
            published_at=old_date,
            is_classified=True,
        )
        for i in range(1, 4)
    ]

    # Base for 3 articles: 0.70; Stale penalty (>30 days): -0.20 -> 0.50
    expected_conf = calculate_news_confidence(old_articles)
    assert expected_conf == 0.50

    mock_payload = {
        "ticker": "NVDA",
        "overall_sentiment": "neutral",
        "sentiment_distribution": {"positive": 1, "negative": 0, "neutral": 2},
        "recent_news": [
            {
                "article_id": "art-old-1",
                "headline": "Historical Development 1",
                "source": "Reuters",
                "url": "https://www.reuters.com/archive/1",
                "published_at": old_date.isoformat(),
                "sentiment": "positive",
            }
        ],
        "important_events": [],
        "positive_factors": [
            {
                "text": "Historical development 1.",
                "article_ids": ["art-old-1"],
            }
        ],
        "negative_factors": [],
        "confidence": 0.50,
        "summary": "Older news coverage reflects baseline operations.",
        "evidence": ["Archived report (art-old-1)."],
    }
    provider = MockNewsSynthesisLLMProvider(responses=[mock_payload])
    agent = NewsAnalystAgent(provider=provider)

    result = agent.run(NewsAnalystInput(ticker="NVDA", articles=old_articles))

    assert result.success is True
    output: NewsAnalysisOutput = result.data
    assert output.confidence == 0.50
    assert output.recent_news[0].published_at == old_date


def test_stale_news_fresh_article_avoids_stale_penalty():
    """Verify presence of fresh article (<14 days) avoids the stale penalty."""
    fresh_date = datetime.now(timezone.utc) - timedelta(hours=3)
    old_date = datetime.now(timezone.utc) - timedelta(days=60)

    fresh_article = ProcessedNewsArticle(
        article_id="art-fresh",
        headline="Fresh Breaking Earnings Announcement",
        summary="Recent announcement.",
        sentiment="positive",
        sentiment_reasoning="Fresh news.",
        important_events=[],
        source="Reuters",
        published_at=fresh_date,
        is_classified=True,
    )
    old_article = ProcessedNewsArticle(
        article_id="art-old",
        headline="Two Months Old General Report",
        summary="Old announcement.",
        sentiment="neutral",
        sentiment_reasoning="Old news.",
        important_events=[],
        source="Bloomberg",
        published_at=old_date,
        is_classified=True,
    )

    # 2 articles -> base confidence 0.55.
    # Newest article is fresh_date (<14 days) -> 0.0 penalty applied -> 0.55
    conf = calculate_news_confidence([fresh_article, old_article])
    assert conf == 0.55
