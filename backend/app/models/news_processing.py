"""Domain models and schemas for Phase 8.2 news processing.

Phase 8.2 implements article-level processing:
- Text extraction representation (headline, summary, source, url, published_at, etc.)
- Sentiment classification (positive, negative, neutral)
- Important event identification (earnings, management_change,
  regulatory_action, merger_acquisition)
- Structured output container (ProcessedNewsArticle) and batch processing results.

Constraints:
- Allowed sentiments are strictly: "positive", "negative", "neutral".
- Allowed event types are strictly: "earnings", "management_change",
  "regulatory_action", "merger_acquisition".
- Does NOT invent missing metadata or summaries.
- Prohibits investment recommendations or buy/sell language.
"""

from datetime import datetime, timezone
from typing import Any, List, Literal, Optional, Set
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

SentimentType = Literal["positive", "negative", "neutral"]
NewsEventType = Literal[
    "earnings",
    "management_change",
    "regulatory_action",
    "merger_acquisition",
]

ALLOWED_SENTIMENTS: Set[str] = {"positive", "negative", "neutral"}
ALLOWED_EVENT_TYPES: Set[str] = {
    "earnings",
    "management_change",
    "regulatory_action",
    "merger_acquisition",
}


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


class NewsEvent(BaseModel):
    """Structured representation of an identified important financial event.

    Attributes:
        event_type: Category of the event.
        evidence: Textual snippet or factual evidence from the article.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    event_type: NewsEventType = Field(
        ...,
        description=(
            "Identified event category. Allowed: 'earnings', "
            "'management_change', 'regulatory_action', 'merger_acquisition'."
        ),
    )
    evidence: str = Field(
        ...,
        description="Factual evidence or quote from the article supporting this event.",
    )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Enforce strict allowed event types."""
        cleaned = v.strip().lower()
        if cleaned not in ALLOWED_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{cleaned}'. Must be one of: "
                f"{sorted(ALLOWED_EVENT_TYPES)}"
            )
        return cleaned

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, v: str) -> str:
        """Ensure evidence is non-empty."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Event evidence cannot be empty or whitespace.")
        return cleaned


class ArticleClassification(BaseModel):
    """Structured schema requested from the LLM for a single news article.

    Attributes:
        sentiment: Sentiment classification of the article.
        sentiment_reasoning: Explanation grounded strictly in supplied article text.
        events: List of identified financial events (empty if none supported).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    sentiment: SentimentType = Field(
        ...,
        description="Classified sentiment: 'positive', 'negative', or 'neutral'.",
    )
    sentiment_reasoning: str = Field(
        ...,
        description="Grounded rationale explaining why this sentiment was assigned.",
    )
    events: List[NewsEvent] = Field(
        default_factory=list,
        description="List of detected important financial events from article text.",
    )

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        """Enforce strict allowed sentiment categories."""
        cleaned = v.strip().lower()
        if cleaned not in ALLOWED_SENTIMENTS:
            raise ValueError(
                f"Invalid sentiment '{cleaned}'. Must be one of: "
                f"{sorted(ALLOWED_SENTIMENTS)}"
            )
        return cleaned

    @field_validator("sentiment_reasoning")
    @classmethod
    def validate_sentiment_reasoning(cls, v: str) -> str:
        """Ensure sentiment reasoning is non-empty."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("sentiment_reasoning cannot be empty or whitespace.")
        return cleaned


class ProcessedNewsArticle(BaseModel):
    """Canonical domain model for a processed news article with extracted metadata,

    sentiment, and detected events.

    Attributes:
        article_id: Unique article identifier from raw news ingestion.
        headline: Grounded title/headline of the news article.
        summary: Optional original summary text if provided by the source.
        sentiment: Classified sentiment ('positive', 'negative', or 'neutral').
        sentiment_reasoning: Explanation grounded strictly in the article content.
        important_events: Detected events supported by article text.
        source: Publisher or news organization name.
        url: Canonical URL to the article if available.
        published_at: Timezone-aware UTC publication timestamp.
        ticker: Primary ticker symbol associated with the article.
        related_tickers: List of related ticker symbols.
        evidence: Optional summary evidence string.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    article_id: str = Field(..., description="Unique article identifier.")
    headline: str = Field(..., description="Original article headline/title.")
    summary: Optional[str] = Field(
        default=None, description="Original article summary if provided."
    )
    sentiment: SentimentType = Field(
        ..., description="Sentiment: 'positive', 'negative', or 'neutral'."
    )
    sentiment_reasoning: str = Field(
        ..., description="Grounded explanation for assigned sentiment."
    )
    important_events: List[NewsEvent] = Field(
        default_factory=list,
        description="Identified important financial events.",
    )
    source: str = Field(..., description="Publisher/source name.")
    url: Optional[str] = Field(default=None, description="Original article URL.")
    published_at: datetime = Field(
        ..., description="Timezone-aware UTC publication timestamp."
    )
    ticker: Optional[str] = Field(
        default=None, description="Primary stock ticker symbol."
    )
    related_tickers: List[str] = Field(
        default_factory=list, description="Related stock ticker symbols."
    )
    evidence: Optional[str] = Field(
        default=None, description="Synthesized factual evidence summary."
    )
    is_classified: bool = Field(
        default=True,
        description="True if classified by model; False if fallback on error.",
    )
    processing_error: Optional[str] = Field(
        default=None,
        description="Error details if classification encountered an error.",
    )

    @field_validator("article_id")
    @classmethod
    def validate_article_id(cls, v: str) -> str:
        """Ensure article_id is non-empty."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("article_id cannot be empty or whitespace.")
        return cleaned

    @field_validator("headline")
    @classmethod
    def validate_headline(cls, v: str) -> str:
        """Ensure headline is non-empty."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("headline cannot be empty or whitespace.")
        return cleaned

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Ensure source is non-empty."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("source cannot be empty or whitespace.")
        return cleaned

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        """Enforce strict allowed sentiment categories."""
        cleaned = v.strip().lower()
        if cleaned not in ALLOWED_SENTIMENTS:
            raise ValueError(
                f"Invalid sentiment '{cleaned}'. Must be one of: "
                f"{sorted(ALLOWED_SENTIMENTS)}"
            )
        return cleaned

    @field_validator("url", mode="before")
    @classmethod
    def validate_url_field(cls, v: Optional[str]) -> Optional[str]:
        """Validate URL format."""
        return _validate_url(v, "url")

    @field_validator("published_at", mode="before")
    @classmethod
    def normalize_published_at(cls, v: Any) -> datetime:
        """Ensure published_at is timezone-aware UTC."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if isinstance(v, str):
            cleaned = v.strip()
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        raise ValueError(f"Unsupported published_at datetime format: {v}")


class NewsProcessingBatchResult(BaseModel):
    """Container for batch article processing results.

    Attributes:
        articles: Ordered list of ProcessedNewsArticle instances.
        total_processed: Total count of articles submitted.
        successful_count: Count of articles successfully processed.
        failed_count: Count of articles where classification fell back to neutral.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    articles: List[ProcessedNewsArticle] = Field(
        default_factory=list, description="List of processed news articles."
    )
    total_processed: int = Field(default=0, description="Total articles processed.")
    successful_count: int = Field(
        default=0, description="Count of successfully classified articles."
    )
    failed_count: int = Field(
        default=0, description="Count of articles with processing fallbacks."
    )
