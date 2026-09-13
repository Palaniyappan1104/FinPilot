"""News processing service for Phase 8.2.

Consumes normalized NewsArticle models and performs article-level processing:
1. Deterministic article text extraction representation.
2. Grounded sentiment classification (positive, negative, neutral) via LLM.
3. Identification of important financial events:
   - earnings
   - management_change
   - regulatory_action
   - merger_acquisition
4. Structured output validation and robust batch processing.

Grounding & Safety Rules:
- The LLM is an interpreter of supplied text only; it never invents facts.
- When article content or evidence is ambiguous or missing, sentiment
  defaults to neutral and events default to empty.
- Strictly prohibits buy/sell/hold investment recommendations and price targets.
- A failure on one article falls back to neutral safely without crashing the batch.
"""

from typing import List, Optional

from app.core.llm.base import LLMProvider
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger
from app.models.news import NewsArticle
from app.models.news_processing import (
    ALLOWED_EVENT_TYPES,
    ALLOWED_SENTIMENTS,
    ArticleClassification,
    NewsEvent,
    NewsProcessingBatchResult,
    ProcessedNewsArticle,
)

logger = get_logger("app.services.news_processor")

PROHIBITED_PHRASES = [
    "buy recommendation",
    "sell recommendation",
    "hold recommendation",
    "recommend buying",
    "recommend selling",
    "recommend to buy",
    "recommend to sell",
    "should buy",
    "should sell",
    "must buy",
    "must sell",
    "strong buy",
    "strong sell",
    "buy rating",
    "sell rating",
    "hold rating",
    "rating: buy",
    "rating: sell",
    "rating: hold",
    "price target",
    "target price",
    "target of $",
    "price goal",
    "projected price",
    "guaranteed return",
    "guaranteed profit",
    "risk-free",
    "investment recommendation",
]

NEWS_PROCESSOR_SYSTEM_PROMPT = (
    "You are FinPilot's Financial News Processing and Sentiment Specialist.\n"
    "Your task is to analyze the supplied news article headline and summary to\n"
    "determine:\n"
    "1. Article Sentiment\n"
    "2. Important Financial Events\n\n"
    "CRITICAL GROUNDING AND ACCURACY RULES:\n"
    "- Classify ONLY from the supplied headline and summary text.\n"
    "- Do NOT invent facts, metrics, events, or company developments.\n"
    "- If summary is missing or brief, base analysis strictly on the headline.\n"
    "- If evidence is ambiguous, mixed, or insufficient, classify sentiment as\n"
    "  'neutral' and events as empty [].\n"
    "- Distinguish factual business reporting from speculation or hype.\n\n"
    "SENTIMENT CLASSIFICATION RULES:\n"
    "Allowed values: 'positive', 'negative', 'neutral'\n"
    "- 'positive': Reports positive business or financial developments (e.g. profit\n"
    "  beat, revenue growth, margin expansion, contract wins, regulatory approval).\n"
    "- 'negative': Reports adverse business or financial developments (e.g. "
    "earnings miss,\n"
    "  guidance cuts, regulatory probes, fines, litigation, executive pressure).\n"
    "- 'neutral': Routine corporate announcements, general market recaps, ambiguous\n"
    "  news, or insufficient evidence.\n\n"
    "IMPORTANT EVENT IDENTIFICATION RULES:\n"
    "Supported event types ONLY:\n"
    "- 'earnings': Quarterly/annual results, revenue/EPS beats/misses, guidance.\n"
    "- 'management_change': CEO, CFO, executive leadership, or board transitions.\n"
    "- 'regulatory_action': Government inquiries, SEC probes, fines, sanctions.\n"
    "- 'merger_acquisition': Mergers, acquisitions, buyouts, divestitures.\n\n"
    "Event Constraints:\n"
    "- If the article does NOT clearly contain any of these 4 specific events,\n"
    "  return an empty events list [].\n"
    "- Do NOT invent events or classify generic news into these categories.\n"
    "- An article may have zero, one, or multiple events if explicitly supported.\n"
    "- Supply exact textual evidence or quotes from the article for each event.\n\n"
    "SAFETY AND COMPLIANCE RULES:\n"
    "- NEVER output investment advice, buy/sell/hold ratings, or trade ideas.\n"
    "- NEVER provide price targets or return predictions.\n"
    "- NEVER use phrases like 'strong buy', 'should sell', or 'target price'."
)


def format_news_prompt(article: NewsArticle) -> str:
    """Construct an isolated, grounded prompt for a single NewsArticle."""
    summary_text = (
        article.summary.strip()
        if article.summary and article.summary.strip()
        else "No summary provided. Classify strictly based on the headline."
    )

    ticker_info = f"Ticker: {article.ticker}" if article.ticker else "Ticker: Unknown"

    prompt = (
        f"{NEWS_PROCESSOR_SYSTEM_PROMPT}\n\n"
        "ARTICLE TO ANALYZE:\n"
        f"- Article ID: {article.article_id}\n"
        f"- Headline: {article.title}\n"
        f"- Summary: {summary_text}\n"
        f"- Source: {article.source}\n"
        f"- Published At: {article.published_at.isoformat()}\n"
        f"- {ticker_info}\n\n"
        "Analyze the article and output structured JSON conforming to "
        "ArticleClassification:\n"
        "- sentiment: 'positive' | 'negative' | 'neutral'\n"
        "- sentiment_reasoning: grounded explanation based solely on the text\n"
        "- events: list of detected events (each with event_type and evidence)\n"
    )
    return prompt


class NewsProcessor:
    """Service that processes news articles into structured sentiment and events."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        """Initialize NewsProcessor with an optional LLMProvider."""
        self._llm_provider = llm_provider

    @property
    def llm_provider(self) -> LLMProvider:
        """Return the active LLM provider, instantiating default if None."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider()
        return self._llm_provider

    def _validate_safety(self, text: str, field_name: str) -> None:
        """Ensure no prohibited investment recommendation or price target phrases."""
        lower_text = text.lower()
        for phrase in PROHIBITED_PHRASES:
            if phrase in lower_text:
                raise ValueError(
                    f"Prohibited advice or recommendation phrase detected "
                    f"in {field_name}: '{phrase}'."
                )

    def process_article(self, article: NewsArticle) -> ProcessedNewsArticle:
        """Process a single NewsArticle into a ProcessedNewsArticle."""
        if not article.title or not article.title.strip():
            raise ValueError(
                f"Article {article.article_id} has empty or missing headline/title."
            )

        prompt = format_news_prompt(article)

        classification: ArticleClassification = generate_structured(
            provider=self.llm_provider,
            prompt=prompt,
            schema=ArticleClassification,
        )

        # Enforce sentiment domain bounds
        if classification.sentiment not in ALLOWED_SENTIMENTS:
            raise ValueError(
                f"Invalid sentiment '{classification.sentiment}' returned by model. "
                f"Allowed: {sorted(ALLOWED_SENTIMENTS)}"
            )

        # Enforce event domain bounds
        validated_events: List[NewsEvent] = []
        for ev in classification.events:
            if ev.event_type not in ALLOWED_EVENT_TYPES:
                raise ValueError(
                    f"Invalid event_type '{ev.event_type}' returned by model. "
                    f"Allowed: {sorted(ALLOWED_EVENT_TYPES)}"
                )
            self._validate_safety(ev.evidence, "event evidence")
            validated_events.append(ev)

        # Enforce safety rules on reasoning
        self._validate_safety(classification.sentiment_reasoning, "sentiment_reasoning")

        # Build evidence synthesis
        evidence_parts: List[str] = [classification.sentiment_reasoning]
        for ev in validated_events:
            evidence_parts.append(f"[{ev.event_type}] {ev.evidence}")
        evidence_summary = " | ".join(evidence_parts)

        return ProcessedNewsArticle(
            article_id=article.article_id,
            headline=article.title.strip(),
            summary=article.summary.strip() if article.summary else None,
            sentiment=classification.sentiment,
            sentiment_reasoning=classification.sentiment_reasoning.strip(),
            important_events=validated_events,
            source=article.source.strip(),
            url=str(article.url) if article.url else None,
            published_at=article.published_at,
            ticker=article.ticker,
            related_tickers=[article.ticker] if article.ticker else [],
            evidence=evidence_summary,
        )

    def process_batch(self, articles: List[NewsArticle]) -> NewsProcessingBatchResult:
        """Process a collection of NewsArticle objects in deterministic order."""
        results: List[ProcessedNewsArticle] = []
        successful_count = 0
        failed_count = 0

        for article in articles:
            try:
                processed = self.process_article(article)
                results.append(processed)
                successful_count += 1
            except Exception as e:
                logger.warning(
                    "Article %s processing failed (%s: %s). Falling back to neutral.",
                    article.article_id,
                    type(e).__name__,
                    str(e),
                )
                err_msg = (
                    f"Classification unavailable due to processing error "
                    f"({type(e).__name__}); defaulted to neutral."
                )
                fallback = ProcessedNewsArticle(
                    article_id=article.article_id,
                    headline=article.title.strip() if article.title else "Untitled",
                    summary=article.summary.strip() if article.summary else None,
                    sentiment="neutral",
                    sentiment_reasoning=err_msg,
                    important_events=[],
                    source=article.source.strip() if article.source else "Unknown",
                    url=str(article.url) if article.url else None,
                    published_at=article.published_at,
                    ticker=article.ticker,
                    related_tickers=[article.ticker] if article.ticker else [],
                    evidence=None,
                    is_classified=False,
                    processing_error=str(e),
                )
                results.append(fallback)
                failed_count += 1

        return NewsProcessingBatchResult(
            articles=results,
            total_processed=len(articles),
            successful_count=successful_count,
            failed_count=failed_count,
        )
