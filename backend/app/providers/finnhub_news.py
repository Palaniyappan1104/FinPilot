"""Finnhub news provider implementation for FinPilot.

Phase 8.1 implements the concrete NewsProvider backed by Finnhub's company-news
REST API endpoint.

Responsibilities:
- Retrieve company and market news for a given ticker.
- Normalize Finnhub response payloads into strongly typed NewsArticle models.
- Deduplicate articles based on article ID and canonical URL.
- Ensure all timestamps are timezone-aware UTC.
- Enforce strict security: NEVER log or expose API keys or credentials.
- Map HTTP/provider failures to typed domain exceptions.
"""

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Union

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.news import NewsArticle, NewsSearchResult
from app.providers.exceptions import (
    EmptyNewsDataError,
    NewsAuthenticationError,
    NewsDataError,
    NewsMalformedDataError,
    NewsProviderNotConfiguredError,
    NewsProviderRateLimitError,
    NewsProviderUnavailableError,
    NewsQueryError,
)
from app.providers.news import NewsProvider

logger = get_logger("app.providers.finnhub_news")


class FinnhubNewsProvider(NewsProvider):
    """Concrete NewsProvider backed by Finnhub's REST API."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        timeout: float = 10.0,
    ) -> None:
        """Initialize FinnhubNewsProvider with credentials and optional HTTP client.

        Args:
            api_key: Finnhub API token (defaults to FINNHUB_API_KEY from settings).
            client: Optional injected httpx.Client for test isolation.
            timeout: HTTP request timeout in seconds.

        Raises:
            NewsProviderNotConfiguredError: If API key is missing or empty.
        """
        raw_key = api_key if api_key is not None else get_settings().FINNHUB_API_KEY
        self._api_key = (raw_key or "").strip()

        if not self._api_key:
            raise NewsProviderNotConfiguredError(
                message=(
                    "Finnhub API key is not configured. "
                    "Set FINNHUB_API_KEY in environment or settings."
                ),
                provider=self.provider_name,
            )

        self._client = client
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        """Return provider identifier name."""
        return "finnhub"

    def _get_client(self) -> httpx.Client:
        """Return the injected client or instantiate a temporary client."""
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self._timeout)

    def _sanitize_text(self, text: str) -> str:
        """Ensure API key never leaks in logs or exception messages."""
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, "***")
        return text

    def normalize_article(
        self,
        raw: Dict[str, Any],
        default_ticker: Optional[str] = None,
    ) -> NewsArticle:
        """Normalize a raw Finnhub article dictionary into a typed NewsArticle.

        Args:
            raw: Raw dictionary item from Finnhub response list.
            default_ticker: Fallback ticker if missing in item.

        Returns:
            NewsArticle: Validated domain model.

        Raises:
            NewsMalformedDataError: If required fields (title, source) are missing
            or data types are invalid.
        """
        if not isinstance(raw, dict):
            raise NewsMalformedDataError(
                f"Expected dictionary for article payload, got {type(raw).__name__}.",
                provider=self.provider_name,
            )

        headline = str(raw.get("headline") or "").strip()
        if not headline:
            raise NewsMalformedDataError(
                "Article headline/title is missing or empty.",
                provider=self.provider_name,
            )

        source = str(raw.get("source") or "").strip()
        if not source:
            raise NewsMalformedDataError(
                "Article source is missing or empty.",
                provider=self.provider_name,
            )

        # Finnhub provides 'datetime' as a Unix epoch timestamp (seconds)
        raw_dt = raw.get("datetime")
        if raw_dt is None:
            raise NewsMalformedDataError(
                "Article publication datetime is missing.",
                provider=self.provider_name,
            )

        # Build deterministic ID if provider ID is absent
        raw_id = raw.get("id")
        raw_url = str(raw.get("url") or "").strip() or None
        if raw_id is not None and str(raw_id).strip():
            article_id = str(raw_id).strip()
        else:
            hash_input = f"{raw_url or ''}|{headline.lower()}|{source.lower()}|{raw_dt}"
            article_id = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

        ticker = str(raw.get("related") or "").strip().upper() or default_ticker
        summary = str(raw.get("summary") or "").strip() or None
        image_url = str(raw.get("image") or "").strip() or None

        category = str(raw.get("category") or "").strip()
        categories = [category] if category else []

        try:
            return NewsArticle(
                article_id=article_id,
                ticker=ticker,
                title=headline,
                summary=summary,
                source=source,
                url=raw_url,
                published_at=raw_dt,
                retrieved_at=datetime.now(timezone.utc),
                image_url=image_url,
                categories=categories,
            )
        except Exception as e:
            raise NewsMalformedDataError(
                f"Failed to construct NewsArticle: {self._sanitize_text(str(e))}",
                provider=self.provider_name,
            ) from e

    def search_news(
        self,
        query: str,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        limit: int = 20,
    ) -> NewsSearchResult:
        """Retrieve, validate, and normalize financial news from Finnhub.

        Args:
            query: Stock ticker symbol (e.g. 'AAPL', 'NVDA').
            start_date: Optional start publication date.
            end_date: Optional end publication date.
            limit: Maximum number of articles to return (default: 20).

        Returns:
            NewsSearchResult: Normalized collection of news articles.

        Raises:
            NewsQueryError: If query or date ranges are invalid.
            NewsAuthenticationError: If API credentials fail.
            NewsProviderRateLimitError: If HTTP 429 is encountered.
            NewsProviderUnavailableError: On timeout or network failure.
            NewsMalformedDataError: If response is corrupt or unexpected structure.
            EmptyNewsDataError: If zero articles are returned.
        """
        ticker = (query or "").strip().upper()
        if not ticker:
            raise NewsQueryError(
                "Query ticker symbol cannot be empty or whitespace.",
                provider=self.provider_name,
            )

        if limit <= 0:
            raise NewsQueryError(
                f"Limit must be strictly positive (> 0), got {limit}.",
                provider=self.provider_name,
            )

        # Date normalization
        now_utc = datetime.now(timezone.utc)
        if end_date is None:
            end_dt = now_utc
        elif isinstance(end_date, str):
            try:
                end_dt = datetime.fromisoformat(end_date.strip())
            except ValueError:
                d = date.fromisoformat(end_date.strip())
                end_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        else:
            end_dt = end_date
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        if start_date is None:
            start_dt = end_dt - timedelta(days=7)
        elif isinstance(start_date, str):
            try:
                start_dt = datetime.fromisoformat(start_date.strip())
            except ValueError:
                d = date.fromisoformat(start_date.strip())
                start_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        else:
            start_dt = start_date
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

        if start_dt > end_dt:
            raise NewsQueryError(
                f"start_date ({start_dt.isoformat()}) cannot be after "
                f"end_date ({end_dt.isoformat()}).",
                provider=self.provider_name,
            )

        from_str = start_dt.strftime("%Y-%m-%d")
        to_str = end_dt.strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/company-news"
        params = {
            "symbol": ticker,
            "from": from_str,
            "to": to_str,
        }
        # Pass token in headers to prevent logging token in query parameters
        headers = {
            "X-Finnhub-Token": self._api_key,
        }

        logger.info(
            "Querying Finnhub news for ticker='%s', from='%s', to='%s'",
            ticker,
            from_str,
            to_str,
        )

        client = self._get_client()
        should_close = self._client is None

        try:
            resp = client.get(url, params=params, headers=headers)
        except (httpx.TimeoutException, httpx.RequestError) as e:
            msg = self._sanitize_text(str(e))
            logger.error("Finnhub network/timeout error: %s", msg)
            raise NewsProviderUnavailableError(
                f"Finnhub provider is unavailable: {msg}",
                provider=self.provider_name,
            ) from e
        except Exception as e:
            msg = self._sanitize_text(str(e))
            logger.error("Finnhub unexpected client error: %s", msg)
            raise NewsDataError(
                f"Unexpected error communicating with Finnhub: {msg}",
                provider=self.provider_name,
            ) from e
        finally:
            if should_close:
                client.close()

        # Handle HTTP status codes
        if resp.status_code in (401, 403):
            logger.error("Finnhub authentication error (HTTP %d)", resp.status_code)
            raise NewsAuthenticationError(
                "Finnhub authentication failed: invalid or unauthorized API key.",
                provider=self.provider_name,
            )
        if resp.status_code == 429:
            logger.warning("Finnhub rate limit exceeded (HTTP 429)")
            raise NewsProviderRateLimitError(
                "Finnhub API rate limit exceeded.",
                provider=self.provider_name,
            )
        if resp.status_code >= 500:
            logger.error("Finnhub server error (HTTP %d)", resp.status_code)
            raise NewsProviderUnavailableError(
                f"Finnhub server returned HTTP {resp.status_code}.",
                provider=self.provider_name,
            )
        if resp.status_code != 200:
            logger.error("Finnhub unexpected status (HTTP %d)", resp.status_code)
            raise NewsDataError(
                f"Finnhub returned unexpected HTTP status {resp.status_code}.",
                provider=self.provider_name,
            )

        # Parse JSON
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            logger.error("Finnhub returned non-JSON response")
            raise NewsMalformedDataError(
                "Finnhub response could not be decoded as JSON.",
                provider=self.provider_name,
            ) from e

        if isinstance(data, dict) and "error" in data:
            err_msg = self._sanitize_text(str(data["error"]))
            logger.error("Finnhub returned error payload: %s", err_msg)
            raise NewsMalformedDataError(
                f"Finnhub returned error message: {err_msg}",
                provider=self.provider_name,
            )

        if not isinstance(data, list):
            raise NewsMalformedDataError(
                f"Expected JSON list from Finnhub, got {type(data).__name__}.",
                provider=self.provider_name,
            )

        if not data:
            logger.info("Finnhub returned empty news list for ticker='%s'", ticker)
            raise EmptyNewsDataError(
                f"No news articles found for '{ticker}' between "
                f"{from_str} and {to_str}.",
                provider=self.provider_name,
            )

        # Deduplicate and normalize
        seen_ids: Set[str] = set()
        seen_urls: Set[str] = set()
        articles: List[NewsArticle] = []

        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                article = self.normalize_article(item, default_ticker=ticker)
            except NewsMalformedDataError:
                # Re-raise if item payload is malformed
                raise

            # Duplicate check by article_id and canonical URL
            if article.article_id in seen_ids:
                logger.debug("Skipping duplicate article_id: %s", article.article_id)
                continue
            if article.url and article.url in seen_urls:
                logger.debug("Skipping duplicate article URL: %s", article.url)
                continue

            seen_ids.add(article.article_id)
            if article.url:
                seen_urls.add(article.url)
            articles.append(article)

        if not articles:
            raise EmptyNewsDataError(
                f"Zero valid news articles remained for '{ticker}'.",
                provider=self.provider_name,
            )

        # Deterministic sort: latest published first
        articles.sort(key=lambda a: a.published_at, reverse=True)
        articles = articles[:limit]

        return NewsSearchResult(
            query=ticker,
            articles=articles,
            provider=self.provider_name,
            retrieved_at=now_utc,
            total_count=len(articles),
        )
