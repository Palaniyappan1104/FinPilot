"""Report Aggregator Agent implementation for FinPilot (Phase 11.2).

Aggregates independent specialist outputs (Technical, Fundamental, News, Research,
Risk) and investor profile into a coherent, evidence-preserving unified analysis.

Fulfills Phase 11.2 requirements:
- Merging and true cross-referencing (consensus, conflicts, cross-domain observations).
- Explicit handling of partial inputs (missing, failed, or empty specialists).
- Deterministic conflict and agreement detection + grounded LLM synthesis.
- Preservation of specialist attribution and source provenance.
- Strict FinPilot safety rules (no buy/sell advice, price targets, guaranteed returns).
"""

from typing import Any, Dict, List, Optional, Union

from app.agents.aggregator_consistency import validate_aggregation_consistency
from app.agents.aggregator_prompt import format_aggregator_prompt
from app.agents.aggregator_schema import (
    PROHIBITED_ADVICE_PATTERNS,
    AggregatedEvidenceItem,
    AggregatorSynthesisOutput,
    CrossSpecialistObservation,
    ReportAggregatorInput,
    ReportAggregatorValidationError,
    SignalConflict,
    SpecialistType,
    SynthesisFinding,
    UnifiedSpecialistAnalysis,
)
from app.agents.base import AgentResult, BaseAgent
from app.agents.state import GraphState
from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import LLMError, LLMStructuredOutputError
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger

logger = get_logger("app.agents.aggregator")


def _val(x: Any) -> str:
    """Safely extract string representation from enum or string."""
    if x is None:
        return ""
    if hasattr(x, "value"):
        return str(x.value)
    return str(x)


# ===========================================================================
# DETERMINISTIC CROSS-REFERENCING DETECTORS (Phase 11.2)
# ===========================================================================


def detect_signal_conflicts(input_data: ReportAggregatorInput) -> List[SignalConflict]:
    """Deterministically identify contradictions and tensions between specialists.

    Evaluates:
    - Technical momentum vs Fundamental outlook (e.g. uptrend vs unfavorable)
    - Technical price trend vs News sentiment (e.g. uptrend vs negative sentiment)
    - Fundamental health vs News sentiment (e.g. favorable vs negative news)
    - Fundamental growth vs Valuation multiple tension (e.g. growth vs expensive)
    - Risk severity vs Technical momentum (e.g. critical risk vs uptrend)
    """
    conflicts: List[SignalConflict] = []
    ev_items = input_data.extract_attributed_evidence()

    def _filter_ev(specialists: List[SpecialistType]) -> List[AggregatedEvidenceItem]:
        return [e for e in ev_items if e.specialist in specialists][:4]

    tech = input_data.technical if input_data.has_technical else None
    fund = input_data.fundamental if input_data.has_fundamental else None
    news = input_data.news if input_data.has_news else None
    risk = input_data.risk if input_data.has_risk else None

    # 1. Technical vs Fundamental Conflict
    if tech and fund:
        tech_trend_str = _val(tech.trend).lower()
        fund_assess_str = _val(fund.overall_assessment).lower()
        tech_bullish = tech_trend_str in ("uptrend", "bullish")
        tech_bearish = tech_trend_str in ("downtrend", "bearish")
        fund_unfavorable = fund_assess_str in ("unfavorable", "bearish")
        fund_favorable = fund_assess_str in ("favorable", "bullish")

        tech_summary = (
            tech.interpretation.overall_summary
            if hasattr(tech, "interpretation") and tech.interpretation
            else ""
        )
        fund_summary = fund.overall_summary if hasattr(fund, "overall_summary") else ""

        if tech_bullish and fund_unfavorable:
            conflicts.append(
                SignalConflict(
                    topic="Technical Momentum vs Fundamental Outlook",
                    description=(
                        f"Technical analysis identifies an {tech_trend_str} with "
                        "upward momentum, whereas fundamental assessment concludes "
                        "an unfavorable operational and valuation outlook."
                    ),
                    specialist_positions={
                        "technical": f"Trend: {tech_trend_str} ({tech_summary})",
                        "fundamental": (
                            f"Assessment: {fund_assess_str} ({fund_summary})"
                        ),
                    },
                    involved_specialists=["technical", "fundamental"],
                    evidence=_filter_ev(["technical", "fundamental"]),
                )
            )
        elif tech_bearish and fund_favorable:
            conflicts.append(
                SignalConflict(
                    topic="Technical Downtrend vs Favorable Fundamentals",
                    description=(
                        f"Technical indicators reflect a {tech_trend_str}, "
                        "contrasting with a fundamentally favorable assessment "
                        "supported by operational performance."
                    ),
                    specialist_positions={
                        "technical": f"Trend: {tech_trend_str} ({tech_summary})",
                        "fundamental": (
                            f"Assessment: {fund_assess_str} ({fund_summary})"
                        ),
                    },
                    involved_specialists=["technical", "fundamental"],
                    evidence=_filter_ev(["technical", "fundamental"]),
                )
            )

    # 2. Technical vs News Conflict
    if tech and news:
        tech_trend_str = _val(tech.trend).lower()
        news_sent_str = _val(news.overall_sentiment).lower()
        tech_bullish = tech_trend_str in ("uptrend", "bullish")
        tech_bearish = tech_trend_str in ("downtrend", "bearish")
        news_negative = news_sent_str in ("negative", "bearish")
        news_positive = news_sent_str in ("positive", "bullish")

        if tech_bullish and news_negative:
            conflicts.append(
                SignalConflict(
                    topic="Technical Strength vs Negative News Sentiment",
                    description=(
                        "Price action and moving averages indicate positive "
                        "momentum, but recent news flow reflects predominantly "
                        "negative market sentiment."
                    ),
                    specialist_positions={
                        "technical": f"Trend: {tech_trend_str}",
                        "news": f"Sentiment: {news_sent_str}",
                    },
                    involved_specialists=["technical", "news"],
                    evidence=_filter_ev(["technical", "news"]),
                )
            )
        elif tech_bearish and news_positive:
            conflicts.append(
                SignalConflict(
                    topic="Technical Weakness vs Positive News Sentiment",
                    description=(
                        "Technical indicators show downward price pressure "
                        "despite encouraging and positive news headlines."
                    ),
                    specialist_positions={
                        "technical": f"Trend: {tech_trend_str}",
                        "news": f"Sentiment: {news_sent_str}",
                    },
                    involved_specialists=["technical", "news"],
                    evidence=_filter_ev(["technical", "news"]),
                )
            )

    # 3. Fundamental vs News Conflict
    if fund and news:
        fund_assess_str = _val(fund.overall_assessment).lower()
        news_sent_str = _val(news.overall_sentiment).lower()
        fund_favorable = fund_assess_str in ("favorable", "bullish")
        fund_unfavorable = fund_assess_str in ("unfavorable", "bearish")
        news_negative = news_sent_str in ("negative", "bearish")
        news_positive = news_sent_str in ("positive", "bullish")

        if fund_favorable and news_negative:
            conflicts.append(
                SignalConflict(
                    topic="Fundamental Strength vs Adverse Headline News",
                    description=(
                        "Underlying financial metrics demonstrate sound fundamental "
                        "execution, contrasting with near-term negative press "
                        "coverage and market sentiment."
                    ),
                    specialist_positions={
                        "fundamental": f"Assessment: {fund_assess_str}",
                        "news": f"Sentiment: {news_sent_str}",
                    },
                    involved_specialists=["fundamental", "news"],
                    evidence=_filter_ev(["fundamental", "news"]),
                )
            )
        elif fund_unfavorable and news_positive:
            conflicts.append(
                SignalConflict(
                    topic="Fundamental Weakness vs Optimistic News Sentiment",
                    description=(
                        "Positive headline sentiment contrasts with underlying "
                        "fundamental weaknesses and balance sheet or profitability "
                        "challenges."
                    ),
                    specialist_positions={
                        "fundamental": f"Assessment: {fund_assess_str}",
                        "news": f"Sentiment: {news_sent_str}",
                    },
                    involved_specialists=["fundamental", "news"],
                    evidence=_filter_ev(["fundamental", "news"]),
                )
            )

    # 4. Risk vs Technical Momentum
    if risk and tech:
        risk_lvl_str = _val(risk.overall_risk_level).lower()
        risk_elevated = risk_lvl_str in ("high", "critical")
        tech_trend_str = _val(tech.trend).lower()
        tech_bullish = tech_trend_str in ("uptrend", "bullish")

        if risk_elevated and tech_bullish:
            conflicts.append(
                SignalConflict(
                    topic="Elevated Risk Profile vs Technical Uptrend",
                    description=(
                        f"Risk analysis identifies elevated {risk_lvl_str} risk, "
                        f"contrasting with an prevailing {tech_trend_str} in "
                        "technical indicators."
                    ),
                    specialist_positions={
                        "risk": f"Overall Risk: {risk_lvl_str}",
                        "technical": f"Trend: {tech_trend_str}",
                    },
                    involved_specialists=["risk", "technical"],
                    evidence=_filter_ev(["risk", "technical"]),
                )
            )

    return conflicts


def detect_synthesis_agreements(
    input_data: ReportAggregatorInput,
) -> List[SynthesisFinding]:
    """Deterministically identify corroborated agreement points across specialists.

    Evaluates:
    - Multi-domain positive consensus (tech uptrend + fund favorable + news positive)
    - Multi-domain negative/cautious consensus (tech downtrend + fund unfavorable)
    - Solvency and cash flow agreement (fund cash flow strong + low financial risk)
    - Growth and research corroboration (fund growth + research findings)
    """
    findings: List[SynthesisFinding] = []
    ev_items = input_data.extract_attributed_evidence()

    def _filter_ev(specialists: List[SpecialistType]) -> List[AggregatedEvidenceItem]:
        return [e for e in ev_items if e.specialist in specialists][:4]

    tech = input_data.technical if input_data.has_technical else None
    fund = input_data.fundamental if input_data.has_fundamental else None
    news = input_data.news if input_data.has_news else None
    risk = input_data.risk if input_data.has_risk else None
    res = input_data.research if input_data.has_research else None

    # 1. Multi-Domain Positive Alignment
    pos_specs: List[SpecialistType] = []
    if tech and _val(tech.trend).lower() in ("uptrend", "bullish"):
        pos_specs.append("technical")
    if fund and _val(fund.overall_assessment).lower() in ("favorable", "bullish"):
        pos_specs.append("fundamental")
    if news and _val(news.overall_sentiment).lower() in ("positive", "bullish"):
        pos_specs.append("news")

    if len(pos_specs) >= 2:
        findings.append(
            SynthesisFinding(
                topic="Multi-Specialist Positive Alignment",
                summary=(
                    f"Specialist signals from {', '.join(pos_specs)} corroborate "
                    "positive operating and market momentum for the asset."
                ),
                supporting_specialists=pos_specs,
                evidence=_filter_ev(pos_specs),
            )
        )

    # 2. Multi-Domain Negative / Cautious Alignment
    neg_specs: List[SpecialistType] = []
    if tech and _val(tech.trend).lower() in ("downtrend", "bearish"):
        neg_specs.append("technical")
    if fund and _val(fund.overall_assessment).lower() in ("unfavorable", "bearish"):
        neg_specs.append("fundamental")
    if news and _val(news.overall_sentiment).lower() in ("negative", "bearish"):
        neg_specs.append("news")
    if risk and _val(risk.overall_risk_level).lower() in ("high", "critical"):
        neg_specs.append("risk")

    if len(neg_specs) >= 2:
        findings.append(
            SynthesisFinding(
                topic="Multi-Specialist Downside Caution",
                summary=(
                    f"Specialists ({', '.join(neg_specs)}) mutually signal "
                    "headwinds, downside price pressure, or elevated risk factors."
                ),
                supporting_specialists=neg_specs,
                evidence=_filter_ev(neg_specs),
            )
        )

    # 3. Cash Flow and Solvency Corroboration
    if fund and risk:
        fund_cf_rating = _val(fund.cash_flow_assessment.rating).lower()
        fund_cf_strong = fund_cf_rating in ("strong", "favorable", "moderate")
        fin_risks_severe = [
            rf
            for rf in risk.financial_risks
            if _val(rf.severity).lower() in ("high", "critical")
        ]
        if fund_cf_strong and not fin_risks_severe:
            findings.append(
                SynthesisFinding(
                    topic="Cash Flow Stability & Financial Solvency",
                    summary=(
                        f"Fundamental cash flow rating is {fund_cf_rating}, and "
                        "risk assessment confirms no critical financial or "
                        "balance sheet vulnerabilities."
                    ),
                    supporting_specialists=["fundamental", "risk"],
                    evidence=_filter_ev(["fundamental", "risk"]),
                )
            )

    # 4. Research Document Corroboration with Fundamentals
    if fund and res and res.key_findings:
        fund_assess = _val(fund.overall_assessment)
        cnt = len(res.key_findings)
        findings.append(
            SynthesisFinding(
                topic="SEC Document & Fundamental Corroboration",
                summary=(
                    f"Official disclosures align with fundamental stance "
                    f"({fund_assess}), substantiating key operational "
                    f"developments ({cnt} document findings)."
                ),
                supporting_specialists=["fundamental", "research"],
                evidence=_filter_ev(["fundamental", "research"]),
            )
        )

    return findings


def detect_cross_observations(
    input_data: ReportAggregatorInput,
) -> List[CrossSpecialistObservation]:
    """Deterministically surface cross-domain observations connecting signals."""
    observations: List[CrossSpecialistObservation] = []
    ev_items = input_data.extract_attributed_evidence()

    def _filter_ev(specialists: List[SpecialistType]) -> List[AggregatedEvidenceItem]:
        return [e for e in ev_items if e.specialist in specialists][:4]

    tech = input_data.technical if input_data.has_technical else None
    fund = input_data.fundamental if input_data.has_fundamental else None
    news = input_data.news if input_data.has_news else None
    risk = input_data.risk if input_data.has_risk else None
    profile = input_data.investor_profile

    # 1. Technical Volatility vs Fundamental Leverage
    if tech and fund:
        tech_trend = _val(tech.trend)
        fund_lev = _val(fund.leverage_assessment.rating)
        obs_text = (
            f"Technical indicators reflect {tech_trend} trend while leverage "
            f"is evaluated as {fund_lev}. Interplay between debt "
            "servicing and price action represents a cross-domain monitor point."
        )
        observations.append(
            CrossSpecialistObservation(
                observation=obs_text,
                connected_specialists=["technical", "fundamental"],
                evidence=_filter_ev(["technical", "fundamental"]),
            )
        )

    # 2. News Impact on Risk Assessment
    if news and risk and news.recent_news:
        news_sent = _val(news.overall_sentiment)
        risk_lvl = (
            _val(risk.overall_risk_level) if risk.overall_risk_level else "evaluated"
        )
        obs_text = (
            f"Recent news sentiment ({news_sent}) across {len(news.recent_news)} "
            f"articles feeds into company/sector risk assessments ({risk_lvl})."
        )
        observations.append(
            CrossSpecialistObservation(
                observation=obs_text,
                connected_specialists=["news", "risk"],
                evidence=_filter_ev(["news", "risk"]),
            )
        )

    # 3. Investor Profile Constraint Interplay
    if profile and risk and profile.risk_tolerance and risk.overall_risk_level:
        risk_lvl = _val(risk.overall_risk_level)
        obs_text = (
            f"Investor risk tolerance is stated as '{profile.risk_tolerance}', "
            f"interacting with assessed overall risk level of '{risk_lvl}'."
        )
        observations.append(
            CrossSpecialistObservation(
                observation=obs_text,
                connected_specialists=["risk"],
                evidence=_filter_ev(["risk"]),
            )
        )

    return observations


def generate_deterministic_synthesis(
    input_data: ReportAggregatorInput,
    agreements: List[SynthesisFinding],
    conflicts: List[SignalConflict],
    observations: List[CrossSpecialistObservation],
) -> str:
    """Generate a coherent, evidence-preserving synthesis narrative."""
    lines: List[str] = []
    ticker = input_data.ticker

    available = input_data.available_specialists
    missing = input_data.missing_specialists
    failed = input_data.failed_specialists

    # 1. Executive Context
    lines.append(
        f"Unified multi-agent research synthesis for {ticker}. "
        f"Analysis synthesizes data from {len(available)} available specialist domains "
        f"({', '.join(available)})."
    )

    if missing:
        lines.append(
            f"Coverage note: {', '.join(missing)} specialist data was unavailable."
        )
    if failed:
        lines.append(f"Notice: {', '.join(failed)} specialist execution failed.")

    # 2. Corroborated Findings
    if agreements:
        agree_topics = [f"{a.topic} ({a.summary})" for a in agreements[:2]]
        lines.append(
            "Key areas of multi-specialist agreement: " + "; ".join(agree_topics) + "."
        )

    # 3. Signal Conflicts and Tensions
    if conflicts:
        conflict_topics = [f"{c.topic}: {c.description}" for c in conflicts[:2]]
        lines.append("Identified signal tensions: " + "; ".join(conflict_topics) + ".")
    else:
        lines.append(
            "No material direct contradictions were observed across active "
            "specialist signals."
        )

    # 4. Cross-Domain Context
    if observations:
        lines.append(f"Cross-specialist observation: {observations[0].observation}")

    # 5. Concluding Decision Support Note
    lines.append(
        f"Overall assessment indicates a composite analytical profile for {ticker} "
        "balancing identified operational factors against market conditions."
    )

    return " ".join(lines)


# ===========================================================================
# REPORT AGGREGATOR AGENT (Phase 11.2)
# ===========================================================================


class ReportAggregatorAgent(BaseAgent):
    """Report Aggregator Agent synthesizing specialist analyst outputs (Phase 11.2).

    Integrates Technical, Fundamental, News, Research, and Risk analyst outputs
    into a unified, evidence-preserving, and non-advisory assessment.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        deterministic_only: bool = False,
    ) -> None:
        super().__init__()
        self._provider = provider
        self.deterministic_only = deterministic_only

    @property
    def name(self) -> str:
        """Unique agent identifier."""
        return "ReportAggregator"

    @property
    def input_schema(self) -> Any:
        """Expected input schema model."""
        return ReportAggregatorInput

    @property
    def output_schema(self) -> Any:
        """Structured output schema model."""
        return UnifiedSpecialistAnalysis

    @property
    def provider(self) -> LLMProvider:
        """Lazy-initialize or return assigned LLMProvider."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def run(
        self,
        input_data: Union[ReportAggregatorInput, GraphState, Dict[str, Any]],
    ) -> AgentResult:
        """Execute unified aggregation and synthesis.

        Args:
            input_data: ReportAggregatorInput, GraphState, or dictionary.

        Returns:
            AgentResult: Successful result containing UnifiedSpecialistAnalysis,
            or failure result with descriptive error.
        """
        # 1. Normalize and parse input
        if isinstance(input_data, ReportAggregatorInput):
            parsed_input = input_data
        elif isinstance(input_data, dict):
            try:
                if (
                    "technical" in input_data
                    or "fundamental" in input_data
                    or "specialist_statuses" in input_data
                ):
                    parsed_input = ReportAggregatorInput.model_validate(input_data)
                elif "technical_result" in input_data or "cio_decision" in input_data:
                    parsed_input = ReportAggregatorInput.from_graph_state(input_data)
                elif "ticker" in input_data:
                    parsed_input = ReportAggregatorInput.from_partial_specialists(
                        ticker=input_data["ticker"],
                        target_company=input_data.get("target_company"),
                        investor_profile=input_data.get("investor_profile"),
                        technical=input_data.get("technical"),
                        fundamental=input_data.get("fundamental"),
                        news=input_data.get("news"),
                        research=input_data.get("research"),
                        risk=input_data.get("risk"),
                        specialist_statuses=input_data.get("specialist_statuses"),
                        specialist_errors=input_data.get("specialist_errors"),
                    )
                else:
                    return AgentResult.create_failure(
                        error=(
                            "Unable to parse ReportAggregatorInput: missing ticker "
                            "or specialists."
                        )
                    )
            except Exception as err:
                logger.error("Failed to parse ReportAggregatorInput: %s", err)
                return AgentResult.create_failure(
                    error=f"Invalid aggregator input payload: {err}"
                )
        else:
            return AgentResult.create_failure(
                error="Input must be an instance of ReportAggregatorInput or dict."
            )

        # 2. Base Unified Analysis initialization
        analysis = UnifiedSpecialistAnalysis.from_input(parsed_input)

        # 3. Deterministic Handling for Completely Empty Inputs
        if parsed_input.is_empty:
            analysis.insufficient_evidence = True
            analysis.insufficient_evidence_reason = (
                "No specialist analysis outputs are available to aggregate."
            )
            analysis.overall_synthesis = (
                f"Aggregation could not proceed for {parsed_input.ticker} due to "
                "complete absence of specialist analysis data."
            )
            analysis.confidence = 0.0
            analysis.consistency_report = validate_aggregation_consistency(
                parsed_input, analysis
            )
            return AgentResult.create_success(data=analysis, confidence=0.0)

        # 4. Deterministic Cross-Referencing Detectors
        det_conflicts = detect_signal_conflicts(parsed_input)
        det_agreements = detect_synthesis_agreements(parsed_input)
        det_observations = detect_cross_observations(parsed_input)

        analysis.areas_of_agreement = det_agreements
        analysis.signal_conflicts = det_conflicts
        analysis.cross_specialist_observations = det_observations

        # 5. Deterministic baseline synthesis
        det_synthesis = generate_deterministic_synthesis(
            parsed_input,
            det_agreements,
            det_conflicts,
            det_observations,
        )

        # Base confidence calculation
        completeness = parsed_input.core_completeness_ratio
        base_confidence = round(
            min(
                0.95,
                max(0.1, 0.4 * completeness + 0.4 - (0.1 if det_conflicts else 0.0)),
            ),
            2,
        )

        # 6. LLM Structured Synthesis (if enabled and not deterministic_only)
        if not self.deterministic_only:
            prompt = format_aggregator_prompt(
                parsed_input,
                preliminary_agreements=det_agreements,
                preliminary_conflicts=det_conflicts,
            )
            try:
                llm_output: AggregatorSynthesisOutput = generate_structured(
                    provider=self.provider,
                    prompt=prompt,
                    schema=AggregatorSynthesisOutput,
                )

                # Reconcile LLM findings with deterministic findings
                if llm_output.overall_synthesis:
                    analysis.overall_synthesis = llm_output.overall_synthesis

                # Combine agreements
                seen_agree_topics = {a.topic.lower() for a in det_agreements}
                for llm_a in llm_output.areas_of_agreement:
                    if llm_a.topic.lower() not in seen_agree_topics:
                        analysis.areas_of_agreement.append(llm_a)
                        seen_agree_topics.add(llm_a.topic.lower())

                # Combine conflicts
                seen_conflict_topics = {c.topic.lower() for c in det_conflicts}
                for llm_c in llm_output.signal_conflicts:
                    if llm_c.topic.lower() not in seen_conflict_topics:
                        analysis.signal_conflicts.append(llm_c)
                        seen_conflict_topics.add(llm_c.topic.lower())

                # Combine observations
                seen_obs = {o.observation.lower() for o in det_observations}
                for llm_o in llm_output.cross_specialist_observations:
                    if llm_o.observation.lower() not in seen_obs:
                        analysis.cross_specialist_observations.append(llm_o)
                        seen_obs.add(llm_o.observation.lower())

                if llm_output.confidence is not None:
                    analysis.confidence = llm_output.confidence
                else:
                    analysis.confidence = base_confidence

                analysis.insufficient_evidence = llm_output.insufficient_evidence
                analysis.insufficient_evidence_reason = (
                    llm_output.insufficient_evidence_reason
                )

            except (LLMError, LLMStructuredOutputError, Exception) as err:
                logger.warning(
                    "LLM structured synthesis failed, falling back to "
                    "deterministic synthesis: %s",
                    err,
                )
                analysis.overall_synthesis = det_synthesis
                analysis.confidence = base_confidence
        else:
            analysis.overall_synthesis = det_synthesis
            analysis.confidence = base_confidence

        # 7. Final Safety Validation
        try:
            for pattern in PROHIBITED_ADVICE_PATTERNS:
                if pattern.search(analysis.overall_synthesis):
                    raise ReportAggregatorValidationError(
                        f"Prohibited advisory language matching '{pattern.pattern}' "
                        "detected in synthesis."
                    )
        except ReportAggregatorValidationError as val_err:
            logger.error("Validation error in aggregator synthesis: %s", val_err)
            return AgentResult.create_failure(error=str(val_err))

        # 8. Consistency Validation (Phase 11.3)
        try:
            analysis.consistency_report = validate_aggregation_consistency(
                parsed_input, analysis
            )
        except Exception as c_err:
            logger.warning("Consistency validation encountered error: %s", c_err)

        return AgentResult.create_success(
            data=analysis,
            confidence=analysis.confidence,
        )


# ===========================================================================
# LANGGRAPH NODE ADAPTER (Phase 11.2)
# ===========================================================================


def report_aggregator_node(
    state: GraphState,
    agent: Optional[ReportAggregatorAgent] = None,
) -> Dict[str, Any]:
    """LangGraph node adapter for the Report Aggregator Agent.

    Extracts specialist results from GraphState, executes aggregation,
    and returns update for 'aggregated_result'.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured ReportAggregatorAgent.

    Returns:
        Dict[str, Any]: State update mapping for 'aggregated_result'.
    """
    active_agent = agent or ReportAggregatorAgent()
    try:
        agg_input = ReportAggregatorInput.from_graph_state(state)
    except Exception as err:
        logger.error(
            "Failed to construct ReportAggregatorInput from GraphState: %s", err
        )
        fail_result = AgentResult.create_failure(
            error=f"Failed to assemble aggregator inputs: {err}"
        )
        return fail_result.to_state_update("aggregated_result")

    result = active_agent.run(agg_input)
    if result.success and hasattr(result.data, "model_dump"):
        return {"aggregated_result": result.data.model_dump()}
    return result.to_state_update("aggregated_result")
