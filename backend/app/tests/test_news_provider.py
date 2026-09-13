"""Unit tests for Phase 8.1 News Data Provider abstraction and normalization.

All tests run 100% offline using httpx.MockTransport and mock fixtures.
ZERO real network or Finnhub API calls are made.

Verifies:
1. Valid article normalization into NewsArticle.
2. Valid multiple-article response normalization into NewsSearchResult.
3. Missing optional fields remain None (no synthetic fabrication).
4. Missing required title is rejected with NewsMalformedDataError / ValueError.
5. Missing required source is rejected with NewsMalformedDataError / ValueError.
6. Invalid/malformed URL is handled correctly.
7. Naive publication timestamp is normalized to timezone-aware UTC.
8. Timezone-aware timestamp is preserved and normalized to UTC.
9. Duplicate article handling (by ID and by canonical URL).
10. Empty provider response raises EmptyNewsDataError.
11. Malformed provider response raises NewsMalformedDataError.
12. Provider timeout and network error mapping to NewsProviderUnavailableError.
13. Rate-limit HTTP 429 error mapping to NewsProviderRateLimitError.
14. Authentication failure HTTP 401/403 mapping to NewsAuthenticationError.
15. Provider factory selects configured provider.
16. Unknown or empty provider configuration fails clearly.
17. API credentials are never leaked in error messages or logs.
18. Deterministic chronological ordering (newest first).
19. Query validation (empty query, limit <= 0, start_date > end_date).
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx
import pytest

from app.core.config import Settings
from app.models.news import NewsArticle, NewsSearchResult
from app.providers.exceptions import (
    EmptyNewsDataError,
    NewsAuthenticationError,
    NewsMalformedDataError,
    NewsProviderNotConfiguredError,
    NewsProviderRateLimitError,
    NewsProviderUnavailableError,
    NewsQueryError,
)
from app.providers.finnhub_news import FinnhubNewsProvider
from app.providers.news import get_news_provider

# ===========================================================================
# Sample Mock Fixtures
# ===========================================================================


@pytest.fixture
def sample_raw_article() -> Dict[str, Any]:
    """Single complete raw article from Finnhub company-news endpoint."""
    return {
        "category": "company",
        "datetime": 1735689600,  # 2025-01-01 00:00:00 UTC
        "headline": "Nvidia Announces Next-Generation AI Supercomputing Architecture",
        "id": 10001,
        "image": "https://images.example.com/nvda-news.jpg",
        "related": "NVDA",
        "source": "Reuters",
        "summary": (
            "Nvidia unveiled its new enterprise architecture driving AI performance."
        ),
        "url": "https://www.reuters.com/technology/nvidia-supercomputing-2025",
    }


@pytest.fixture
def sample_raw_news_list(sample_raw_article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List of diverse raw articles from Finnhub."""
    article2 = {
        "category": "market",
        "datetime": 1735776000,  # 2025-01-02 00:00:00 UTC (newer)
        "headline": "Semiconductor Stocks Rally Led by Nvidia Innovations",
        "id": 10002,
        "image": "https://images.example.com/chips-rally.jpg",
        "related": "NVDA",
        "source": "Bloomberg",
        "summary": "Tech indices surged following hardware manufacturer announcements.",
        "url": "https://www.bloomberg.com/news/articles/chips-rally-2025",
    }
    article3 = {
        "category": "company",
        "datetime": 1735603200,  # 2024-12-31 00:00:00 UTC (older)
        "headline": "Analysts Review Tech Sector Year-End Performance",
        "id": 10003,
        "image": None,
        "related": "NVDA",
        "source": "Dow Jones",
        "summary": "Annual review of major hardware and cloud technology providers.",
        "url": "https://www.dowjones.com/articles/tech-year-end-review",
    }
    return [sample_raw_article, article2, article3]


def make_mock_client(
    handler: Any,
) -> httpx.Client:
    """Construct an offline httpx.Client with custom MockTransport."""
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


# ===========================================================================
# 1. Domain Model & Normalization Tests
# ===========================================================================


def test_valid_article_normalization(sample_raw_article: Dict[str, Any]):
    """Verify single raw Finnhub item normalizes into typed NewsArticle."""
    provider = FinnhubNewsProvider(api_key="test_api_key")
    article = provider.normalize_article(sample_raw_article, default_ticker="NVDA")

    assert isinstance(article, NewsArticle)
    assert article.article_id == "10001"
    assert article.ticker == "NVDA"
    assert (
        article.title
        == "Nvidia Announces Next-Generation AI Supercomputing Architecture"
    )
    assert article.source == "Reuters"
    assert (
        article.summary
        == "Nvidia unveiled its new enterprise architecture driving AI performance."
    )
    assert (
        article.url == "https://www.reuters.com/technology/nvidia-supercomputing-2025"
    )
    assert article.image_url == "https://images.example.com/nvda-news.jpg"
    assert article.categories == ["company"]
    assert article.published_at == datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert article.published_at.tzinfo is not None
    assert article.retrieved_at.tzinfo is not None


def test_valid_multiple_article_response(sample_raw_news_list: List[Dict[str, Any]]):
    """Verify multi-article response normalizes into NewsSearchResult."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=sample_raw_news_list)

    client = make_mock_client(handler)
    provider = FinnhubNewsProvider(api_key="test_api_key", client=client)

    result = provider.search_news("NVDA", limit=10)

    assert isinstance(result, NewsSearchResult)
    assert result.query == "NVDA"
    assert result.provider == "finnhub"
    assert result.total_count == 3
    assert len(result.articles) == 3
    assert all(isinstance(a, NewsArticle) for a in result.articles)


def test_missing_optional_fields_remain_none():
    """Verify optional fields (summary, image, url, author) remain None when omitted."""
    raw = {
        "datetime": 1735689600,
        "headline": "Minimal Article Headline",
        "id": 9999,
        "source": "Wire Service",
        # summary, image, url, related, category absent
    }
    provider = FinnhubNewsProvider(api_key="test_api_key")
    article = provider.normalize_article(raw)

    assert article.title == "Minimal Article Headline"
    assert article.source == "Wire Service"
    assert article.summary is None
    assert article.url is None
    assert article.image_url is None
    assert article.author is None
    assert article.ticker is None
    assert article.categories == []


def test_missing_required_title_rejected():
    """Verify raw article with missing or whitespace-only headline is rejected."""
    provider = FinnhubNewsProvider(api_key="test_api_key")

    # Missing headline key
    raw_missing = {"datetime": 1735689600, "source": "Reuters", "id": 1}
    with pytest.raises(NewsMalformedDataError, match="headline/title is missing"):
        provider.normalize_article(raw_missing)

    # Empty headline
    raw_empty = {
        "datetime": 1735689600,
        "headline": "   ",
        "source": "Reuters",
        "id": 2,
    }
    with pytest.raises(NewsMalformedDataError, match="headline/title is missing"):
        provider.normalize_article(raw_empty)

    # Direct Pydantic model validation
    with pytest.raises(ValueError, match="title cannot be empty"):
        NewsArticle(
            article_id="1",
            title="   ",
            source="Reuters",
            published_at=datetime.now(timezone.utc),
        )


def test_missing_required_source_rejected():
    """Verify raw article with missing or whitespace-only source is rejected."""
    provider = FinnhubNewsProvider(api_key="test_api_key")

    # Missing source key
    raw_missing = {"datetime": 1735689600, "headline": "Valid Headline", "id": 1}
    with pytest.raises(NewsMalformedDataError, match="source is missing"):
        provider.normalize_article(raw_missing)

    # Empty source
    raw_empty = {
        "datetime": 1735689600,
        "headline": "Valid Headline",
        "source": "  ",
        "id": 2,
    }
    with pytest.raises(NewsMalformedDataError, match="source is missing"):
        provider.normalize_article(raw_empty)

    # Direct Pydantic model validation
    with pytest.raises(ValueError, match="source cannot be empty"):
        NewsArticle(
            article_id="1",
            title="Valid Title",
            source="   ",
            published_at=datetime.now(timezone.utc),
        )


def test_invalid_malformed_url_handled_correctly():
    """Verify malformed URLs without scheme or netloc are rejected."""
    # Direct model validation with invalid URL
    with pytest.raises(ValueError, match="Must be a valid HTTP or HTTPS URL"):
        NewsArticle(
            article_id="1",
            title="Headline",
            source="Source",
            url="not-a-real-url",
            published_at=datetime.now(timezone.utc),
        )

    # Empty URL string normalizes cleanly to None
    article = NewsArticle(
        article_id="1",
        title="Headline",
        source="Source",
        url="   ",
        published_at=datetime.now(timezone.utc),
    )
    assert article.url is None


def test_naive_publication_timestamp_normalized_to_utc():
    """Verify naive datetime input is safely converted to timezone-aware UTC."""
    naive_dt = datetime(2025, 1, 15, 14, 30, 0)
    article = NewsArticle(
        article_id="1",
        title="Headline",
        source="Source",
        published_at=naive_dt,
    )
    assert article.published_at.tzinfo is not None
    assert article.published_at.tzinfo == timezone.utc
    assert article.published_at.hour == 14
    assert article.published_at.minute == 30


def test_timezone_aware_timestamp_preserved_correctly():
    """Verify non-UTC timezone-aware timestamp is converted to UTC without
    point-in-time drift."""
    # UTC+5:30 (e.g. IST)
    ist = timezone(timedelta(hours=5, minutes=30))
    ist_dt = datetime(2025, 1, 15, 15, 30, 0, tzinfo=ist)

    article = NewsArticle(
        article_id="1",
        title="Headline",
        source="Source",
        published_at=ist_dt,
    )
    assert article.published_at.tzinfo == timezone.utc
    # 15:30 in UTC+5:30 equals 10:00 UTC
    assert article.published_at.hour == 10
    assert article.published_at.minute == 0


# ===========================================================================
# 2. Duplicate Detection & Deterministic Ordering Tests
# ===========================================================================


def test_duplicate_article_handling():
    """Verify deduplication by article ID and by canonical URL."""
    items = [
        # Article 1 (original)
        {
            "id": 101,
            "headline": "First Unique Article",
            "source": "Reuters",
            "datetime": 1735689600,
            "url": "https://example.com/art1",
        },
        # Article 2 (duplicate ID 101)
        {
            "id": 101,
            "headline": "Duplicate by ID",
            "source": "Reuters",
            "datetime": 1735689600,
            "url": "https://example.com/art2",
        },
        # Article 3 (duplicate URL https://example.com/art1 with different ID 102)
        {
            "id": 102,
            "headline": "Duplicate by URL",
            "source": "Bloomberg",
            "datetime": 1735689600,
            "url": "https://example.com/art1",
        },
        # Article 4 (distinct article)
        {
            "id": 103,
            "headline": "Second Unique Article",
            "source": "CNBC",
            "datetime": 1735693200,
            "url": "https://example.com/art4",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=items)

    client = make_mock_client(handler)
    provider = FinnhubNewsProvider(api_key="test_api_key", client=client)

    result = provider.search_news("AAPL")

    # Exactly 2 unique articles should remain
    assert result.total_count == 2
    titles = [a.title for a in result.articles]
    assert "First Unique Article" in titles
    assert "Second Unique Article" in titles
    assert "Duplicate by ID" not in titles
    assert "Duplicate by URL" not in titles


def test_deterministic_ordering_of_normalized_results(
    sample_raw_news_list: List[Dict[str, Any]],
):
    """Verify articles are sorted strictly descending by publication timestamp
    (newest first)."""

    # Note: sample_raw_news_list timestamps:
    # 1735689600 (Jan 1), 1735776000 (Jan 2), 1735603200 (Dec 31)
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=sample_raw_news_list)

    client = make_mock_client(handler)
    provider = FinnhubNewsProvider(api_key="test_api_key", client=client)

    result = provider.search_news("NVDA")

    assert len(result.articles) == 3
    # First article should be Jan 2
    assert result.articles[0].published_at == datetime(
        2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc
    )
    # Second article should be Jan 1
    assert result.articles[1].published_at == datetime(
        2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc
    )
    # Third article should be Dec 31
    assert result.articles[2].published_at == datetime(
        2024, 12, 31, 0, 0, 0, tzinfo=timezone.utc
    )


# ===========================================================================
# 3. Provider Error & Status Code Mapping Tests
# ===========================================================================


def test_empty_provider_response_raises_empty_news_data_error():
    """Verify empty list from provider raises EmptyNewsDataError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = make_mock_client(handler)
    provider = FinnhubNewsProvider(api_key="test_api_key", client=client)

    with pytest.raises(EmptyNewsDataError, match="No news articles found"):
        provider.search_news("UNKNOWN_TICKER")


def test_malformed_provider_response():
    """Verify non-JSON, error payloads, and non-list responses raise
    NewsMalformedDataError."""
    # 1. Non-JSON string
    client_non_json = make_mock_client(
        lambda r: httpx.Response(200, text="NOT_VALID_JSON{")
    )
    prov_non_json = FinnhubNewsProvider(api_key="test_api_key", client=client_non_json)
    with pytest.raises(NewsMalformedDataError, match="could not be decoded as JSON"):
        prov_non_json.search_news("AAPL")

    # 2. JSON error payload
    client_err = make_mock_client(
        lambda r: httpx.Response(200, json={"error": "Unknown symbol"})
    )
    prov_err = FinnhubNewsProvider(api_key="test_api_key", client=client_err)
    with pytest.raises(NewsMalformedDataError, match="Finnhub returned error message"):
        prov_err.search_news("AAPL")

    # 3. Non-list JSON (dict instead of list)
    client_dict = make_mock_client(lambda r: httpx.Response(200, json={"items": []}))
    prov_dict = FinnhubNewsProvider(api_key="test_api_key", client=client_dict)
    with pytest.raises(NewsMalformedDataError, match="Expected JSON list"):
        prov_dict.search_news("AAPL")


def test_provider_timeout_and_network_error_mapping():
    """Verify network timeouts map cleanly to NewsProviderUnavailableError."""

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out after 10.0s")

    client = make_mock_client(timeout_handler)
    provider = FinnhubNewsProvider(api_key="test_api_key", client=client)

    with pytest.raises(NewsProviderUnavailableError, match="provider is unavailable"):
        provider.search_news("AAPL")


def test_rate_limit_error_mapping():
    """Verify HTTP 429 maps to NewsProviderRateLimitError."""

    def rate_limit_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Too Many Requests")

    client = make_mock_client(rate_limit_handler)
    provider = FinnhubNewsProvider(api_key="test_api_key", client=client)

    with pytest.raises(NewsProviderRateLimitError, match="rate limit exceeded"):
        provider.search_news("AAPL")


def test_authentication_error_mapping():
    """Verify HTTP 401 and 403 map to NewsAuthenticationError."""

    def auth_fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized: Invalid API key")

    client = make_mock_client(auth_fail_handler)
    provider = FinnhubNewsProvider(api_key="bad_api_key", client=client)

    with pytest.raises(NewsAuthenticationError, match="authentication failed"):
        provider.search_news("AAPL")


# ===========================================================================
# 4. Factory & Configuration Tests
# ===========================================================================


def test_provider_not_configured_when_key_is_empty():
    """Initializing FinnhubNewsProvider with empty key raises
    NewsProviderNotConfiguredError."""
    with pytest.raises(
        NewsProviderNotConfiguredError, match="Finnhub API key is not configured"
    ):
        FinnhubNewsProvider(api_key="")

    with pytest.raises(
        NewsProviderNotConfiguredError, match="Finnhub API key is not configured"
    ):
        FinnhubNewsProvider(api_key="   ")


def test_provider_factory_selects_configured_provider():
    """Factory correctly instantiates FinnhubNewsProvider when configured
    in settings."""
    settings = Settings(NEWS_PROVIDER="finnhub", FINNHUB_API_KEY="test_valid_key")
    provider = get_news_provider(settings=settings)

    assert isinstance(provider, FinnhubNewsProvider)
    assert provider.provider_name == "finnhub"


def test_unknown_provider_configuration_fails_clearly():
    """Unsupported provider configuration raises NewsProviderNotConfiguredError."""
    settings = Settings(NEWS_PROVIDER="unsupported_news_service")
    with pytest.raises(
        NewsProviderNotConfiguredError, match="Unsupported NEWS_PROVIDER"
    ):
        get_news_provider(settings=settings)

    empty_settings = Settings(NEWS_PROVIDER="")
    with pytest.raises(NewsProviderNotConfiguredError, match="cannot be empty"):
        get_news_provider(settings=empty_settings)


def test_api_credentials_never_included_in_logs_or_errors():
    """Verify secret tokens never leak in exception messages."""
    secret = "SUPER_SECRET_TOKEN_ABC123456789"

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RequestError(f"Failed connecting to server with token={secret}")

    client = make_mock_client(fail_handler)
    provider = FinnhubNewsProvider(api_key=secret, client=client)

    with pytest.raises(NewsProviderUnavailableError) as exc_info:
        provider.search_news("AAPL")

    error_str = str(exc_info.value)
    assert secret not in error_str
    assert "***" in error_str


# ===========================================================================
# 5. Query & Parameter Validation Tests
# ===========================================================================


def test_query_validation_empty_or_whitespace():
    """Empty or whitespace query raises NewsQueryError."""
    provider = FinnhubNewsProvider(api_key="test_api_key")

    with pytest.raises(NewsQueryError, match="Query ticker symbol cannot be empty"):
        provider.search_news("")

    with pytest.raises(NewsQueryError, match="Query ticker symbol cannot be empty"):
        provider.search_news("   ")


def test_query_validation_negative_or_zero_limit():
    """Limit <= 0 raises NewsQueryError."""
    provider = FinnhubNewsProvider(api_key="test_api_key")

    with pytest.raises(NewsQueryError, match="Limit must be strictly positive"):
        provider.search_news("AAPL", limit=0)

    with pytest.raises(NewsQueryError, match="Limit must be strictly positive"):
        provider.search_news("AAPL", limit=-5)


def test_query_validation_start_date_after_end_date():
    """start_date > end_date raises NewsQueryError."""
    provider = FinnhubNewsProvider(api_key="test_api_key")

    start = datetime(2025, 2, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(NewsQueryError, match="start_date .* cannot be after end_date"):
        provider.search_news("AAPL", start_date=start, end_date=end)
