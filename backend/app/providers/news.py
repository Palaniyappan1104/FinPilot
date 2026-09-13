"""News Provider interface and factory for FinPilot.

Phase 8.1 establishes the provider-agnostic interface for retrieving financial
and company news and the factory for instantiating the configured provider.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Union

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.news import NewsSearchResult
from app.providers.exceptions import (
    NewsProviderNotConfiguredError,
)

logger = get_logger("app.providers.news")


class NewsProvider(ABC):
    """Abstract interface defining the contract for financial news providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier name (e.g. 'finnhub')."""
        pass

    @abstractmethod
    def search_news(
        self,
        query: str,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        limit: int = 20,
    ) -> NewsSearchResult:
        """Retrieve and normalize financial news articles for the given query/ticker.

        Args:
            query: Stock ticker symbol or company keyword (e.g. 'AAPL', 'NVDA').
            start_date: Optional start of publication date range.
            end_date: Optional end of publication date range.
            limit: Maximum number of articles to return (default: 20).

        Returns:
            NewsSearchResult: Strongly typed normalized domain model carrying
            deduplicated articles, query metadata, and chronological ordering.

        Raises:
            NewsAuthenticationError: If provider credentials are invalid or missing.
            NewsProviderNotConfiguredError: If provider configuration is invalid.
            NewsProviderUnavailableError: If provider is unreachable or timed out.
            NewsProviderRateLimitError: If provider explicit rate limit (HTTP 429).
            NewsMalformedDataError: If provider response is corrupt or unexpected.
            EmptyNewsDataError: If zero articles are returned for the query.
            NewsQueryError: If query parameters or date range are invalid.
            NewsDataError: For any other unhandled news provider errors.
        """
        pass


def get_news_provider(
    settings: Optional[Settings] = None,
) -> NewsProvider:
    """Instantiate and return the configured NewsProvider.

    Reads provider type from settings.NEWS_PROVIDER (default: 'finnhub').

    Args:
        settings: Optional application settings instance.

    Returns:
        NewsProvider: Configured concrete provider implementation.

    Raises:
        NewsProviderNotConfiguredError: If configured provider is unsupported.
    """
    app_settings = settings or get_settings()
    provider_name = (
        (getattr(app_settings, "NEWS_PROVIDER", "finnhub") or "").strip().lower()
    )

    if not provider_name:
        raise NewsProviderNotConfiguredError(
            message="NEWS_PROVIDER configuration cannot be empty.",
            provider="unknown",
        )

    if provider_name == "finnhub":
        logger.info("Initializing news provider: Finnhub")
        from app.providers.finnhub_news import FinnhubNewsProvider

        return FinnhubNewsProvider(api_key=getattr(app_settings, "FINNHUB_API_KEY", ""))

    logger.error("Unsupported NEWS_PROVIDER configured: '%s'", provider_name)
    raise NewsProviderNotConfiguredError(
        message=(
            f"Unsupported NEWS_PROVIDER '{provider_name}'. "
            "Supported providers are: ['finnhub']."
        ),
        provider=provider_name,
    )
