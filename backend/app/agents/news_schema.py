"""Normalized domain and structured output schemas for News Analyst Agent.

Phase 8.3 defines the input and structured output models for synthesizing
article-level signals into an overall news analysis assessment:
- NewsAnalystInput: Carries ticker, ProcessedNewsArticle list, and optional context.
- RecentNewsItem: Traceable summary item linking back to original article.
- NewsAnalysisOutput: Comprehensive structured news assessment.
- NewsAnalysisValidationError: Typed exception for grounding or safety violations.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.news_processing import (
    NewsEventType,
    ProcessedNewsArticle,
    SentimentType,
)

OverallSentimentType = Literal[
    "positive",
    "negative",
    "neutral",
    "insufficient_news",
]

ALLOWED_OVERALL_SENTIMENTS: Set[str] = {
    "positive",
    "negative",
    "neutral",
    "insufficient_news",
}


class NewsAnalysisValidationError(ValueError):
    """Raised when News Analyst output violates grounding, safety, or schema rules."""

    pass


class FactorItem(BaseModel):
    """Structured narrative factor linked to source article provenance."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    text: str = Field(..., description="Narrative factor description.")
    article_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Source article IDs directly supporting this factor.",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Factor text cannot be empty or whitespace.")
        return cleaned

    @field_validator("article_ids")
    @classmethod
    def validate_article_ids(cls, v: List[str]) -> List[str]:
        cleaned = [aid.strip() for aid in v if aid.strip()]
        if not cleaned:
            raise ValueError("Factor must contain at least one non-empty article_id.")
        return cleaned


class NewsAnalysisEvent(BaseModel):
    """Structured financial event with article-level source provenance."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    event_type: NewsEventType = Field(
        ...,
        description=(
            "Identified event category. Allowed: 'earnings', 'management_change', "
            "'regulatory_action', 'merger_acquisition'."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Factual summary or evidence of the event grounded in source articles."
        ),
    )
    article_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Source article IDs where this event was detected.",
    )

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Event description cannot be empty or whitespace.")
        return cleaned

    @field_validator("article_ids")
    @classmethod
    def validate_article_ids(cls, v: List[str]) -> List[str]:
        cleaned = [aid.strip() for aid in v if aid.strip()]
        if not cleaned:
            raise ValueError("Event must contain at least one non-empty article_id.")
        return cleaned


class RecentNewsItem(BaseModel):
    """Traceable news item representing a key recent article with full provenance."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    article_id: str = Field(..., description="Unique source article identifier.")
    headline: str = Field(..., description="Original article headline.")
    summary: Optional[str] = Field(
        default=None, description="Article summary if available."
    )
    source: str = Field(..., description="Publisher or wire source.")
    url: Optional[str] = Field(
        default=None, description="Canonical article link if available."
    )
    published_at: datetime = Field(
        ..., description="Timezone-aware UTC publication timestamp."
    )
    sentiment: SentimentType = Field(
        ..., description="Article-level classified sentiment."
    )

    @field_validator("article_id")
    @classmethod
    def validate_article_id(cls, v: str) -> str:
        """Validate non-empty article ID."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("article_id cannot be empty or whitespace.")
        return cleaned

    @field_validator("headline")
    @classmethod
    def validate_headline(cls, v: str) -> str:
        """Validate non-empty headline."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("headline cannot be empty or whitespace.")
        return cleaned

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate non-empty source."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("source cannot be empty or whitespace.")
        return cleaned


class NewsAnalystInput(BaseModel):
    """Input payload for the News Analyst Agent."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Normalized target ticker symbol.")
    articles: List[ProcessedNewsArticle] = Field(
        default_factory=list,
        description="Collection of processed news articles from Phase 8.2.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Investor investment time horizon (e.g. 'short term', '1 year').",
    )
    risk_tolerance: Optional[str] = Field(
        default=None,
        description="Investor risk tolerance (e.g. 'conservative', 'aggressive').",
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Specific analytical guidance passed from CIO routing.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate non-empty uppercase ticker."""
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty or whitespace.")
        return cleaned


class NewsAnalysisOutput(BaseModel):
    """Structured analytical assessment produced by the News Analyst Agent."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Normalized uppercase ticker symbol.")
    overall_sentiment: OverallSentimentType = Field(
        ...,
        description=(
            "Synthesized news sentiment: 'positive', 'negative', 'neutral', "
            "or 'insufficient_news'."
        ),
    )
    sentiment_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of article-level sentiments (positive, negative, neutral).",
    )
    recent_news: List[RecentNewsItem] = Field(
        default_factory=list,
        description="Curated collection of recent relevant articles with citations.",
    )
    important_events: List[NewsAnalysisEvent] = Field(
        default_factory=list,
        description="Grounded financial events identified across the articles.",
    )
    positive_factors: List[FactorItem] = Field(
        default_factory=list,
        description="Key positive narrative factors grounded in cited articles.",
    )
    negative_factors: List[FactorItem] = Field(
        default_factory=list,
        description="Key negative narrative factors grounded in cited articles.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic confidence score representing news evidence quality."
        ),
    )
    summary: str = Field(
        ...,
        description="Synthesized narrative interpretation of the company news posture.",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Traceable article references supporting the synthesis.",
    )
    insufficient_news_reason: Optional[str] = Field(
        default=None,
        description="Explanation if overall_sentiment is 'insufficient_news'.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize uppercase ticker symbol."""
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty or whitespace.")
        return cleaned

    @field_validator("overall_sentiment")
    @classmethod
    def validate_overall_sentiment(cls, v: str) -> str:
        """Enforce strict allowed overall sentiments."""
        cleaned = v.strip().lower()
        if cleaned not in ALLOWED_OVERALL_SENTIMENTS:
            raise ValueError(
                f"Invalid overall_sentiment '{cleaned}'. Must be one of: "
                f"{sorted(ALLOWED_OVERALL_SENTIMENTS)}"
            )
        return cleaned
