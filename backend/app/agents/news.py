"""News Analyst Agent implementation for FinPilot.

Phase 8.3 implements the specialist agent responsible for synthesizing
article-level signals (sentiment, important events, recency) into a
coherent, structured qualitative assessment of company news posture.

Core Architectural Principles:
- Consumes pre-processed ProcessedNewsArticle models from Phase 8.2.
- The LLM is a synthesis and interpretation layer: it never invents articles,
  dates, sources, URLs, financial facts, or events.
- If news data is empty or all articles failed classification, the agent
  gracefully outputs overall_sentiment='insufficient_news' with confidence=0.0.
- Preserves full article traceability in recent_news and evidence items.
- Strictly prohibits buy/sell/hold recommendations and price targets.
- Confidence strictly reflects data volume, recency, and signal consistency.
- Exposes a LangGraph-compatible node adapter for workflow integration.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from app.agents.base import AgentResult, BaseAgent
from app.agents.news_schema import (
    ALLOWED_OVERALL_SENTIMENTS,
    NewsAnalysisOutput,
    NewsAnalysisValidationError,
    NewsAnalystInput,
)
from app.agents.state import GraphState
from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import LLMError, LLMStructuredOutputError
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger
from app.models.news_processing import (
    NewsProcessingBatchResult,
    ProcessedNewsArticle,
)

logger = get_logger("app.agents.news")

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

NEWS_ANALYST_SYSTEM_PROMPT = (
    "You are FinPilot's Senior Financial News and Sentiment Analyst.\n"
    "Your task is to synthesize the supplied article-level news findings into a\n"
    "grounded, objective analytical assessment of the company's current news\n"
    "posture.\n\n"
    "CRITICAL GROUNDING AND SYNTHESIS RULES:\n"
    "1. Analyze ONLY the supplied pre-processed articles.\n"
    "2. NEVER invent or hallucinate news stories, events, dates, or sources.\n"
    "3. NEVER create synthetic citations or reference articles not in the input.\n"
    "4. Overall sentiment must respect article-level sentiment evidence:\n"
    "   - If all usable articles are positive, overall sentiment cannot be negative.\n"
    "   - If all usable articles are negative, overall sentiment cannot be positive.\n"
    "   - In mixed evidence, choose neutral or a balanced assessment.\n"
    "5. Every positive factor MUST include article_ids from supplied articles.\n"
    "6. Every negative factor MUST include article_ids from supplied articles.\n"
    "7. Every important event MUST include article_ids and use supplied evidence.\n"
    "   Do not invent event details or create unsupported events.\n"
    "8. Confidence is determined by evidence quality, not by LLM opinion.\n\n"
    "SAFETY AND REGULATORY BOUNDARIES:\n"
    "- NEVER issue buy/sell/hold ratings, recommendations, or trade ideas.\n"
    "- NEVER predict or state target prices or projected price goals.\n"
    "- NEVER promise or guarantee financial returns or risk-free outcomes.\n"
    "- High confidence indicates only that the news evidence is comprehensive and\n"
    "  consistent; it is NEVER an investment probability or directional prediction."
)


def calculate_news_confidence(
    articles: List[ProcessedNewsArticle],
) -> float:
    """Deterministically calculate a confidence ceiling for news analysis.

    Reflects:
    - Number of successfully classified articles.
    - Ratio of unclassified/failed articles.
    - Recency of the newest article.
    """
    classified = [a for a in articles if a.is_classified]
    n_classified = len(classified)

    if n_classified == 0:
        return 0.0

    # Base confidence based on usable article volume
    if n_classified == 1:
        base_conf = 0.35
    elif n_classified == 2:
        base_conf = 0.55
    elif n_classified == 3:
        base_conf = 0.70
    elif n_classified == 4:
        base_conf = 0.80
    else:
        base_conf = 0.90

    # Recency check
    now = datetime.now(timezone.utc)
    newest_dt = max(a.published_at for a in classified)
    days_old = (now - newest_dt).total_seconds() / 86400.0

    if days_old > 30.0:
        base_conf = max(0.20, base_conf - 0.20)
    elif days_old > 14.0:
        base_conf = max(0.30, base_conf - 0.10)

    # Failure penalty if part of the batch failed
    if len(articles) > 0:
        failure_ratio = (len(articles) - n_classified) / len(articles)
        if failure_ratio > 0.3:
            base_conf = max(0.20, base_conf - 0.15)

    return round(base_conf, 2)


def format_news_synthesis_prompt(
    input_data: NewsAnalystInput,
    confidence_ceiling: float,
) -> str:
    """Construct an isolated synthesis prompt for the News Analyst."""
    usable = [a for a in input_data.articles if a.is_classified]

    # Article details
    article_entries = []
    for a in usable:
        events_str = (
            ", ".join(f"{ev.event_type} ('{ev.evidence}')" for ev in a.important_events)
            if a.important_events
            else "None"
        )
        entry = (
            f"Article ID: {a.article_id}\n"
            f"Headline: {a.headline}\n"
            f"Summary: {a.summary or 'None'}\n"
            f"Source: {a.source}\n"
            f"Published: {a.published_at.isoformat()}\n"
            f"URL: {a.url or 'None'}\n"
            f"Classified Sentiment: {a.sentiment}\n"
            f"Sentiment Reasoning: {a.sentiment_reasoning}\n"
            f"Detected Events: {events_str}"
        )
        article_entries.append(entry)

    articles_text = (
        "\n\n---\n\n".join(article_entries)
        if article_entries
        else "No usable classified articles."
    )

    prompt = (
        f"{NEWS_ANALYST_SYSTEM_PROMPT}\n\n"
        f"TARGET TICKER: {input_data.ticker}\n"
        f"USABLE ARTICLES COUNT: {len(usable)}\n"
        f"DATA SUFFICIENCY CONFIDENCE CEILING: {confidence_ceiling}\n\n"
        "SUPPLIED PROCESSED ARTICLES:\n"
        f"{articles_text}\n\n"
        "Generate a structured NewsAnalysisOutput conforming to the schema:\n"
        "- ticker: must match target ticker\n"
        "- overall_sentiment: positive, negative, neutral, or insufficient_news\n"
        "- sentiment_distribution: count of positive, negative, neutral articles\n"
        "- recent_news: list of RecentNewsItem (must match supplied article_id, "
        "headline, source, date, url)\n"
        "- important_events: list of NewsAnalysisEvent with event_type, description, "
        "and supporting article_ids\n"
        "- positive_factors: list of FactorItem with text and supporting article_ids\n"
        "- negative_factors: list of FactorItem with text and supporting article_ids\n"
        "- confidence: numeric value between 0.0 and 1.0 (system-determined ceiling: "
        f"{confidence_ceiling})\n"
        "- summary: objective narrative synthesis of news posture\n"
        "- evidence: list of key factual evidence citations\n"
    )
    return prompt


def validate_news_analysis_grounding(
    output: NewsAnalysisOutput,
    input_data: NewsAnalystInput,
    confidence_ceiling: float,
) -> None:
    """Enforce strict grounding, provenance, and safety rules on model output.

    Raises:
        NewsAnalysisValidationError: If any rule is violated.
    """
    # 1. Ticker validation
    if output.ticker.strip().upper() != input_data.ticker.strip().upper():
        raise NewsAnalysisValidationError(
            f"Ticker mismatch: expected '{input_data.ticker}', got '{output.ticker}'."
        )

    # 2. Overall sentiment validation
    if output.overall_sentiment not in ALLOWED_OVERALL_SENTIMENTS:
        raise NewsAnalysisValidationError(
            f"Invalid overall_sentiment '{output.overall_sentiment}'."
        )

    usable_articles = [a for a in input_data.articles if a.is_classified]
    if len(usable_articles) == 0 and output.overall_sentiment != "insufficient_news":
        raise NewsAnalysisValidationError(
            "Overall sentiment must be 'insufficient_news' when zero usable "
            "articles are provided."
        )

    # Build sentiment distribution deterministically from usable articles
    det_distribution = {"positive": 0, "negative": 0, "neutral": 0}
    for a in usable_articles:
        if a.sentiment in det_distribution:
            det_distribution[a.sentiment] += 1
    output.sentiment_distribution = det_distribution

    # Sentiment contradiction guardrails
    if det_distribution["positive"] > 0 and det_distribution["negative"] == 0:
        if output.overall_sentiment == "negative":
            raise NewsAnalysisValidationError(
                "Overall sentiment cannot be 'negative' when there are positive "
                "articles and zero negative articles in the evidence."
            )
    if det_distribution["negative"] > 0 and det_distribution["positive"] == 0:
        if output.overall_sentiment == "positive":
            raise NewsAnalysisValidationError(
                "Overall sentiment cannot be 'positive' when there are negative "
                "articles and zero positive articles in the evidence."
            )

    # Map input articles by ID
    article_map: Dict[str, ProcessedNewsArticle] = {
        a.article_id: a for a in input_data.articles
    }

    # 3. Recent news provenance validation
    for item in output.recent_news:
        if item.article_id not in article_map:
            raise NewsAnalysisValidationError(
                f"Recent news references unknown article_id '{item.article_id}'."
            )
        orig = article_map[item.article_id]
        if item.source != orig.source:
            raise NewsAnalysisValidationError(
                f"Source mismatch for article '{item.article_id}': "
                f"expected '{orig.source}', got '{item.source}'."
            )

    # 4. Factor provenance validation (positive & negative factors)
    for factor in output.positive_factors:
        if not factor.article_ids:
            raise NewsAnalysisValidationError(
                f"Positive factor '{factor.text}' has no article_ids."
            )
        for aid in factor.article_ids:
            if aid not in article_map:
                raise NewsAnalysisValidationError(
                    f"Positive factor references unknown article_id '{aid}'."
                )

    for factor in output.negative_factors:
        if not factor.article_ids:
            raise NewsAnalysisValidationError(
                f"Negative factor '{factor.text}' has no article_ids."
            )
        for aid in factor.article_ids:
            if aid not in article_map:
                raise NewsAnalysisValidationError(
                    f"Negative factor references unknown article_id '{aid}'."
                )

    # 5. Important events provenance validation
    for ev in output.important_events:
        if not ev.article_ids:
            raise NewsAnalysisValidationError(
                f"Important event '{ev.event_type}' has no article_ids."
            )
        for aid in ev.article_ids:
            if aid not in article_map:
                raise NewsAnalysisValidationError(
                    f"Important event references unknown article_id '{aid}'."
                )
            art = article_map[aid]
            matching_events = [
                ae for ae in art.important_events if ae.event_type == ev.event_type
            ]
            if not matching_events:
                raise NewsAnalysisValidationError(
                    f"Event type '{ev.event_type}' was not detected in "
                    f"referenced article '{aid}'."
                )
            # Factual grounding: event description must have token overlap with evidence
            known_tokens = set(
                re.findall(
                    r"\b\w{3,}\b",
                    (
                        " ".join(ae.evidence for ae in matching_events)
                        + " "
                        + art.headline
                        + " "
                        + (art.summary or "")
                    ).lower(),
                )
            )
            desc_tokens = set(re.findall(r"\b\w{3,}\b", ev.description.lower()))
            if desc_tokens and not desc_tokens.intersection(known_tokens):
                raise NewsAnalysisValidationError(
                    f"Event description for '{ev.event_type}' in article '{aid}' "
                    f"lacks factual grounding: '{ev.description}'."
                )

    # 6. Prohibited phrases check
    texts_to_check = (
        [output.summary]
        + [f.text for f in output.positive_factors]
        + [f.text for f in output.negative_factors]
        + [ev.description for ev in output.important_events]
        + output.evidence
    )
    for text in texts_to_check:
        lower = text.lower()
        for phrase in PROHIBITED_PHRASES:
            if phrase in lower:
                raise NewsAnalysisValidationError(
                    f"Prohibited advice or recommendation phrase detected: '{phrase}'."
                )

    # 7. Final confidence assignment (deterministic)
    output.confidence = confidence_ceiling


class NewsAnalystAgent(BaseAgent):
    """Specialist agent synthesizing processed news articles into news analysis."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        name: str = "news_analyst",
    ) -> None:
        """Initialize NewsAnalystAgent with an optional LLMProvider."""
        super().__init__()
        self._provider = provider
        self._name = name

    @property
    def provider(self) -> LLMProvider:
        """Return the active LLM provider, instantiating default if None."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    @property
    def name(self) -> str:
        """Unique identifier name of the agent."""
        return self._name

    @property
    def input_schema(self) -> type:
        """Expected input schema."""
        return NewsAnalystInput

    @property
    def output_schema(self) -> type:
        """Structured output schema."""
        return NewsAnalysisOutput

    def run(
        self,
        input_data: Union[
            NewsAnalystInput,
            NewsProcessingBatchResult,
            List[ProcessedNewsArticle],
            Dict[str, Any],
        ],
    ) -> AgentResult:
        """Execute grounded news synthesis.

        Args:
            input_data: NewsAnalystInput, batch result, article list, or dictionary.

        Returns:
            AgentResult containing NewsAnalysisOutput or descriptive failure.
        """
        # Parse input payload
        if isinstance(input_data, NewsAnalystInput):
            parsed_input = input_data
        elif isinstance(input_data, NewsProcessingBatchResult):
            ticker = (
                input_data.articles[0].ticker
                if input_data.articles and input_data.articles[0].ticker
                else "UNKNOWN"
            )
            parsed_input = NewsAnalystInput(ticker=ticker, articles=input_data.articles)
        elif isinstance(input_data, list):
            ticker = (
                input_data[0].ticker
                if input_data and input_data[0].ticker
                else "UNKNOWN"
            )
            parsed_input = NewsAnalystInput(ticker=ticker, articles=input_data)
        elif isinstance(input_data, dict):
            try:
                parsed_input = NewsAnalystInput.model_validate(input_data)
            except Exception as err:
                return AgentResult.create_failure(
                    error=f"Invalid news analyst input payload: {err}"
                )
        else:
            return AgentResult.create_failure(
                error=(
                    "Input must be an instance of NewsAnalystInput, "
                    "NewsProcessingBatchResult, list, or dict."
                )
            )

        usable_articles = [a for a in parsed_input.articles if a.is_classified]
        confidence_ceiling = calculate_news_confidence(parsed_input.articles)

        # Fast-path for zero usable news: deterministic insufficient_news response
        if len(usable_articles) == 0:
            logger.info(
                "Zero usable news articles for ticker='%s'; generating "
                "insufficient_news output.",
                parsed_input.ticker,
            )
            output = NewsAnalysisOutput(
                ticker=parsed_input.ticker,
                overall_sentiment="insufficient_news",
                sentiment_distribution={"positive": 0, "negative": 0, "neutral": 0},
                recent_news=[],
                important_events=[],
                positive_factors=[],
                negative_factors=[],
                confidence=0.0,
                summary=(
                    f"Insufficient usable news data available for ticker "
                    f"'{parsed_input.ticker}'. No recent classified news articles "
                    "could be analyzed."
                ),
                evidence=[],
                insufficient_news_reason=(
                    "Zero successfully classified news articles available for analysis."
                ),
            )
            return AgentResult.create_success(data=output, confidence=0.0)

        prompt = format_news_synthesis_prompt(parsed_input, confidence_ceiling)

        try:
            output: NewsAnalysisOutput = generate_structured(
                provider=self.provider,
                prompt=prompt,
                schema=NewsAnalysisOutput,
            )

            # Enforce deterministic confidence before validation
            output.confidence = confidence_ceiling

            # Grounding and safety validation
            validate_news_analysis_grounding(
                output=output,
                input_data=parsed_input,
                confidence_ceiling=confidence_ceiling,
            )

            return AgentResult.create_success(
                data=output,
                confidence=output.confidence,
            )
        except NewsAnalysisValidationError as err:
            logger.error("News grounding validation failed: %s", err)
            return AgentResult.create_failure(
                error=f"News analysis grounding validation failed: {err}"
            )
        except LLMStructuredOutputError as err:
            logger.error("Structured output generation failed: %s", err)
            return AgentResult.create_failure(
                error=f"News analysis structured output validation failed: {err}"
            )
        except LLMError as err:
            logger.error("LLM Provider error: %s", err)
            return AgentResult.create_failure(
                error=f"LLM provider error during news analysis: {err}"
            )
        except Exception as err:
            logger.error(
                "Unexpected error during news analysis: %s", err, exc_info=True
            )
            return AgentResult.create_failure(
                error=f"Unexpected news analysis error: {err}"
            )


def news_analyst_node(
    state: GraphState,
    agent: Optional[NewsAnalystAgent] = None,
) -> Dict[str, Any]:
    """LangGraph node adapter for the News Analyst Agent.

    Extracts news data from state, invokes NewsAnalystAgent, and updates
    'news_result' in GraphState.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured NewsAnalystAgent.

    Returns:
        Dict[str, Any]: State update mapping for 'news_result'.
    """
    active_agent = agent or NewsAnalystAgent()
    ticker = state.get("target_company") or state.get("ticker") or "UNKNOWN"

    raw_news = state.get("news_data") or state.get("processed_news") or []
    articles: List[ProcessedNewsArticle] = []

    if isinstance(raw_news, list):
        for item in raw_news:
            if isinstance(item, ProcessedNewsArticle):
                articles.append(item)
            elif isinstance(item, dict):
                try:
                    articles.append(ProcessedNewsArticle.model_validate(item))
                except Exception:
                    pass

    profile = state.get("investor_profile") or {}
    input_data = NewsAnalystInput(
        ticker=ticker,
        articles=articles,
        time_horizon=profile.get("time_horizon"),
        risk_tolerance=profile.get("risk_tolerance"),
    )

    result: AgentResult = active_agent.run(input_data)

    if result.success and result.data:
        return {
            "news_result": (
                result.data.model_dump()
                if hasattr(result.data, "model_dump")
                else result.data
            )
        }
    return {"news_result": {"error": result.error, "success": False}}
