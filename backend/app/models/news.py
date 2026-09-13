"""Normalized domain models for financial news articles and search results.

Phase 8.1 defines strongly typed models for news articles and search results:
- NewsArticle: Canonical representation of a single normalized news article.
- NewsSearchResult: Collection container carrying query metadata and articles.

Boundary & Validation Rules:
- Captures raw/underlying article data from news providers.
- Strictly isolated from third-party provider objects.
- Preserves actual publication timestamps as timezone-aware UTC.
- Preserves missing optional metadata as None; NEVER fabricates synthetic article data.
- Enforces non-empty title and source, and validates URL syntax when provided.
"""

from datetime import date, datetime, timezone
from typing import Any, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_datetime(v: Any, field_name: str) -> datetime:
    """Normalize timestamps, ISO strings, and epoch numbers to timezone-aware UTC."""
    if v is None:
        raise ValueError(f"{field_name} cannot be None.")

    # Epoch numbers (int or float)
    if isinstance(v, (int, float)):
        # Finnhub timestamps are in seconds; handle millisecond timestamps if > 1e11
        ts = float(v)
        if ts > 1e11:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    # ISO strings or date strings
    if isinstance(v, str):
        cleaned = v.strip()
        if not cleaned:
            raise ValueError(f"{field_name} string cannot be empty or whitespace.")
        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            try:
                d = date.fromisoformat(cleaned)
                dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            except ValueError:
                raise ValueError(f"Unsupported {field_name} date format: '{cleaned}'")
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    # Python datetime objects
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    raise ValueError(f"Unsupported {field_name} type: {type(v)} ({v})")


def _validate_url(v: Optional[str], field_name: str) -> Optional[str]:
    """Validate URL format if provided; normalize empty strings to None."""
    if v is None:
        return None
    cleaned = v.strip()
    if not cleaned:
        return None
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid {field_name}: '{cleaned}'. Must be a valid HTTP or HTTPS URL."
        )
    return cleaned


class NewsArticle(BaseModel):
    """Normalized domain model representing a single financial news article.

    Attributes:
        article_id: Unique article identifier (provider-assigned or deterministic hash).
        ticker: Associated stock ticker symbol if applicable (e.g. 'AAPL').
        title: Headline or title of the news article (non-empty).
        summary: Brief summary or description text if provided.
        source: Publisher or news source name (non-empty, e.g. 'Reuters', 'Bloomberg').
        url: Canonical URL link to the original article if provided.
        published_at: Timezone-aware UTC publication timestamp.
        retrieved_at: Timezone-aware UTC timestamp when fetched by FinPilot.
        author: Byline or author name if provided.
        image_url: URL link to article header/thumbnail image if provided.
        categories: Topical or sector tags associated with the article.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    article_id: str = Field(
        ..., description="Unique article identifier or deterministic hash."
    )
    ticker: Optional[str] = Field(
        default=None, description="Associated ticker symbol if applicable."
    )
    title: str = Field(..., description="Article headline or title.")
    summary: Optional[str] = Field(
        default=None, description="Article summary or description."
    )
    source: str = Field(..., description="Publisher or news source name.")
    url: Optional[str] = Field(
        default=None, description="Canonical URL to the full article."
    )
    published_at: datetime = Field(
        ..., description="Timezone-aware UTC publication timestamp."
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timezone-aware UTC retrieval timestamp.",
    )
    author: Optional[str] = Field(
        default=None, description="Author or byline if provided."
    )
    image_url: Optional[str] = Field(
        default=None, description="Thumbnail or header image URL."
    )
    categories: List[str] = Field(
        default_factory=list, description="Topical or sector classification tags."
    )

    @field_validator("article_id")
    @classmethod
    def validate_article_id(cls, v: str) -> str:
        """Validate non-empty article ID."""
        if not v or not v.strip():
            raise ValueError("article_id cannot be empty or whitespace.")
        return v.strip()

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: Optional[str]) -> Optional[str]:
        """Normalize ticker to uppercase."""
        if v is None:
            return None
        cleaned = v.strip().upper()
        return cleaned if cleaned else None

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate non-empty title string."""
        if not v or not v.strip():
            raise ValueError("title cannot be empty or whitespace.")
        return v.strip()

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, v: Optional[str]) -> Optional[str]:
        """Normalize summary text."""
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate non-empty source name."""
        if not v or not v.strip():
            raise ValueError("source cannot be empty or whitespace.")
        return v.strip()

    @field_validator("url", mode="before")
    @classmethod
    def validate_url_field(cls, v: Optional[str]) -> Optional[str]:
        """Validate canonical article URL."""
        return _validate_url(v, "url")

    @field_validator("image_url", mode="before")
    @classmethod
    def validate_image_url_field(cls, v: Optional[str]) -> Optional[str]:
        """Validate article image URL."""
        return _validate_url(v, "image_url")

    @field_validator("published_at", mode="before")
    @classmethod
    def normalize_published_at(cls, v: Any) -> datetime:
        """Ensure published_at is timezone-aware UTC."""
        return _normalize_datetime(v, "published_at")

    @field_validator("retrieved_at", mode="before")
    @classmethod
    def normalize_retrieved_at(cls, v: Any) -> datetime:
        """Ensure retrieved_at is timezone-aware UTC."""
        return _normalize_datetime(v, "retrieved_at")

    @field_validator("author")
    @classmethod
    def validate_author(cls, v: Optional[str]) -> Optional[str]:
        """Normalize author string."""
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None


class NewsSearchResult(BaseModel):
    """Container carrying query metadata and normalized news articles.

    Attributes:
        query: Search query or ticker symbol requested.
        articles: Normalized NewsArticle instances.
        provider: Identifier of data provider (e.g. 'finnhub').
        retrieved_at: Timezone-aware UTC retrieval timestamp.
        total_count: Total count of articles returned.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    query: str = Field(..., description="Searched symbol or keyword query.")
    articles: List[NewsArticle] = Field(
        default_factory=list, description="Retrieved normalized articles."
    )
    provider: str = Field(..., description="Provider identifier name.")
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timezone-aware UTC retrieval timestamp.",
    )
    total_count: int = Field(
        default=0, description="Number of returned articles in this result."
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate non-empty search query."""
        if not v or not v.strip():
            raise ValueError("query cannot be empty or whitespace.")
        return v.strip().upper()

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate non-empty provider name."""
        if not v or not v.strip():
            raise ValueError("provider cannot be empty or whitespace.")
        return v.strip().lower()

    @field_validator("retrieved_at", mode="before")
    @classmethod
    def normalize_retrieved_at(cls, v: Any) -> datetime:
        """Ensure retrieved_at is timezone-aware UTC."""
        return _normalize_datetime(v, "retrieved_at")
