"""Report Generator Agent implementation for FinPilot (Phase 12.2).

Transforms aggregated specialist analysis (Phase 11) into a comprehensive,
structured, explainable final investment report conforming to Phase 12.1 schemas.

Fulfills Phase 12.2 requirements:
- Grounded recommendation generation with specific reasons tied to evidence.
- Explains findings across technical, fundamental, news, research, and risk domains.
- Distinguishes available, missing, failed, and insufficient evidence.
- Strict non-advisory safety rules (no buy/sell orders, no price targets, no
  guarantees).
- Deterministic report generation baseline + LLM structured synthesis with fallback.
- Grounding and provenance validation (rejecting fabricated numbers/citations).
- Reuses BaseAgent and LLMProvider abstractions.
"""

import re
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.aggregator_consistency import (
    EXCLUDED_NUMBERS,
    _number_in_set,
    extract_numbers_from_text,
)
from app.agents.aggregator_schema import (
    PROHIBITED_ADVICE_PATTERNS,
    UnifiedSpecialistAnalysis,
)
from app.agents.base import AgentResult, BaseAgent
from app.agents.report_prompt import format_report_generator_prompt
from app.agents.report_schema import (
    FinalReport,
    RecommendationStance,
    ReportGeneratorInput,
    ReportRecommendation,
    ReportValidationError,
)
from app.agents.state import GraphState
from app.core.llm.base import LLMProvider
from app.core.llm.exceptions import LLMError, LLMStructuredOutputError
from app.core.llm.factory import get_llm_provider
from app.core.llm.structured import generate_structured
from app.core.logging import get_logger

logger = get_logger("app.agents.report_generator")


def _val(x: Any) -> str:
    """Safely extract string representation from enum or object."""
    if x is None:
        return ""
    if hasattr(x, "value"):
        return str(x.value)
    return str(x)


# ===========================================================================
# STRUCTURED LLM SYNTHESIS OUTPUT SCHEMA (Phase 12.2)
# ===========================================================================


class GeneratedReportSynthesis(BaseModel):
    """Structured LLM synthesis payload for final report generation (Phase 12.2)."""

    model_config = ConfigDict(extra="ignore")

    recommendation_stance: RecommendationStance = Field(
        ...,
        description=(
            "Decision-support stance: favorable, cautious, neutral, "
            "unfavorable, or insufficient_evidence."
        ),
    )
    recommendation_rationale: str = Field(
        ...,
        description=(
            "Detailed analytical rationale grounded strictly in specialist "
            "findings and evidence."
        ),
    )
    profile_alignment: str = Field(
        ...,
        description=(
            "Explanation of how findings align with investor's goals, horizon, "
            "and risk tolerance."
        ),
    )
    monitoring_points: List[str] = Field(
        default_factory=list,
        description="Key upcoming indicators, earnings dates, or technical levels.",
    )
    key_reasons: List[str] = Field(
        ...,
        min_length=1,
        description="Top specific analytical reasons directly supporting the stance.",
    )
    important_risks: List[str] = Field(
        ...,
        min_length=1,
        description="Top critical risk factors the investor must monitor.",
    )
    executive_synthesis: Optional[str] = Field(
        default=None,
        description="Optional refined executive summary synthesizing the report.",
    )

    @field_validator("recommendation_rationale", "profile_alignment")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned

    @field_validator("recommendation_rationale", "profile_alignment")
    @classmethod
    def validate_safety(cls, v: str) -> str:
        for pattern in PROHIBITED_ADVICE_PATTERNS:
            if pattern.search(v):
                raise ReportValidationError(
                    f"Prohibited advisory phrase '{pattern.pattern}' in "
                    "report synthesis."
                )
        return v


# ===========================================================================
# GROUNDING AND SAFETY VALIDATION (Phase 12.2)
# ===========================================================================


def extract_analysis_numbers(analysis: UnifiedSpecialistAnalysis) -> Set[float]:
    """Collect all numeric values present in the source UnifiedSpecialistAnalysis."""
    numbers: Set[float] = set()

    def _collect(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            val = float(obj)
            if val not in EXCLUDED_NUMBERS:
                numbers.add(val)
        elif isinstance(obj, str):
            numbers.update(extract_numbers_from_text(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _collect(item)
        elif hasattr(obj, "model_dump"):
            _collect(obj.model_dump())
        elif hasattr(obj, "__dict__"):
            _collect(obj.__dict__)

    _collect(analysis.model_dump())
    return numbers


# Pattern to detect citation references like [tech_ev_0] or (ref: gross_margin)
CITATION_REF_PATTERN = re.compile(
    r"\[(?:ref:\s*)?([a-zA-Z0-9_\-:]+)\]|\((?:ref:\s*)([a-zA-Z0-9_\-:]+)\)"
)


def validate_report_grounding(
    report: FinalReport,
    analysis: UnifiedSpecialistAnalysis,
) -> None:
    """Validate that generated report fields are grounded in supplied analysis.

    Enforces Phase 12.2 safety and grounding rules:
    1. Prohibited advice patterns across all text fields.
    2. Insufficient evidence consistency (stance MUST be INSUFFICIENT_EVIDENCE
       if analysis is marked insufficient).
    3. Signal conflict awareness (stance should reflect caution or explicitly
       address tension).
    4. Evidence reference provenance (any cited reference IDs must exist in
       aggregated evidence).
    5. Numerical provenance (numerical values in generated rationale, key_reasons,
       and important_risks must exist in source analysis or represent common
       integers/dates).

    Raises:
        ReportValidationError: If any grounding or safety rule is violated.
    """
    # 1. Prohibited advice patterns
    texts_to_scan: List[str] = [
        report.overall_assessment.synthesis,
    ]
    if report.recommendation:
        texts_to_scan.append(report.recommendation.rationale)
        texts_to_scan.append(report.recommendation.profile_alignment)
    texts_to_scan.extend(report.key_reasons)
    texts_to_scan.extend(report.important_risks)

    for text in texts_to_scan:
        if not text:
            continue
        for pattern in PROHIBITED_ADVICE_PATTERNS:
            if pattern.search(text):
                raise ReportValidationError(
                    f"Report violates safety rules: contains prohibited advisory "
                    f"phrase '{pattern.pattern}'."
                )

    # 2. Insufficient evidence consistency
    if analysis.insufficient_evidence:
        if (
            report.recommendation
            and report.recommendation.stance
            != RecommendationStance.INSUFFICIENT_EVIDENCE
        ):
            raise ReportValidationError(
                "Report recommendation stance must be 'insufficient_evidence' "
                "when upstream analysis indicates insufficient evidence."
            )

    # 3. Evidence reference provenance
    known_ref_ids: Set[str] = {
        e.reference_id.lower() for e in analysis.aggregated_evidence
    }
    for e in analysis.aggregated_evidence:
        if e.chunk_id:
            known_ref_ids.add(e.chunk_id.lower())
        if e.document_id:
            known_ref_ids.add(e.document_id.lower())

    # Scan for explicit citations in narrative
    if known_ref_ids:
        narrative_to_check = " ".join(texts_to_scan)
        found_citations = CITATION_REF_PATTERN.findall(narrative_to_check)
        for match_tuple in found_citations:
            cited_id = (match_tuple[0] or match_tuple[1]).strip().lower()
            # Ignore common non-citation bracketed words like [1], [CHK_...], [table]
            if (
                cited_id
                and not cited_id.isdigit()
                and not cited_id.startswith("chk_")
                and cited_id not in known_ref_ids
            ):
                raise ReportValidationError(
                    f"Report narrative cites ungrounded evidence reference: "
                    f"'{cited_id}'."
                )

    # 4. Numerical provenance validation
    analysis_numbers = extract_analysis_numbers(analysis)
    investor_numbers: Set[float] = set()
    if analysis.investor_profile:
        if analysis.investor_profile.capital_amount is not None:
            investor_numbers.add(float(analysis.investor_profile.capital_amount))
        if analysis.investor_profile.time_horizon:
            investor_numbers.update(
                extract_numbers_from_text(analysis.investor_profile.time_horizon)
            )

    combined_allowed_numbers = analysis_numbers | investor_numbers

    # Only scan the newly generated text sections for invented financial claims
    generated_texts: List[str] = []
    if report.recommendation:
        generated_texts.append(report.recommendation.rationale)
    generated_texts.extend(report.key_reasons)
    generated_texts.extend(report.important_risks)

    for g_text in generated_texts:
        for num in extract_numbers_from_text(g_text):
            if num in EXCLUDED_NUMBERS:
                continue
            if not _number_in_set(num, combined_allowed_numbers):
                raise ReportValidationError(
                    f"Report introduces ungrounded numerical value {num} not found "
                    "in specialist analysis or investor profile."
                )


# ===========================================================================
# DETERMINISTIC REPORT GENERATION BASELINE (Phase 12.2)
# ===========================================================================


def generate_deterministic_report(
    analysis: UnifiedSpecialistAnalysis,
    report_id: Optional[str] = None,
) -> FinalReport:
    """Generate a fully grounded FinalReport using deterministic decision rules.

    Operates independently of LLM availability, providing a reliable, safe baseline:
    - If insufficient evidence: produces INSUFFICIENT_EVIDENCE recommendation.
    - If signal conflicts: produces CAUTIOUS recommendation.
    - If fundamental and technical alignment: produces FAVORABLE or UNFAVORABLE.
    - Otherwise: produces NEUTRAL recommendation.
    - Synthesizes key reasons and important risks strictly from available data.

    Args:
        analysis: Validated UnifiedSpecialistAnalysis.
        report_id: Optional persistence report identifier.

    Returns:
        FinalReport: Complete structured report conforming to Phase 12.1.
    """
    ticker = analysis.ticker
    inv_prof = analysis.investor_profile

    # 1. Stance and Rationale Derivation
    if analysis.insufficient_evidence or analysis.data_completeness_ratio < 0.4:
        stance = RecommendationStance.INSUFFICIENT_EVIDENCE
        reason = analysis.insufficient_evidence_reason or (
            "Specialist coverage is insufficient across core analytical domains."
        )
        rationale = (
            f"An analytical posture cannot be formulated for {ticker} due to "
            f"insufficient specialist data: {reason}"
        )
        profile_alignment = (
            "Analytical assessment deferred due to incomplete specialist coverage."
        )
        monitoring_points = [
            "Data availability across primary specialist domains",
            "Next quarterly financial reporting release",
        ]
        key_reasons = [
            f"Evaluation of available specialist indicators for {ticker}",
            "Preservation of safety under incomplete coverage",
        ]
        important_risks = [
            "Unverified operational profile due to missing specialist coverage",
            "Potential market volatility unaddressed by omitted specialists",
        ]

    elif analysis.signal_conflicts:
        stance = RecommendationStance.CAUTIOUS
        conflict_topics = "; ".join(c.topic for c in analysis.signal_conflicts[:2])
        rationale = (
            f"Caution is indicated for {ticker} due to analytical tension identified "
            f"across competing specialist signals: {conflict_topics}."
        )
        profile_alignment = (
            "A cautious posture prioritizes risk management and capital preservation "
            "under conflicting cross-domain signals."
        )
        monitoring_points = [
            "Resolution of specialist divergence between technicals and fundamentals",
            "Next corporate earnings announcement",
            "Key technical support and resistance levels",
        ]
        key_reasons = []
        for a in analysis.areas_of_agreement[:2]:
            key_reasons.append(f"{a.topic}: {a.summary}")
        if not key_reasons and analysis.technical_assessment:
            t = analysis.technical_assessment
            key_reasons.append(f"Technical trend: {_val(t.trend)}")
        if not key_reasons:
            key_reasons.append(
                f"Evaluated {ticker} across available specialist indicators."
            )

        important_risks = [
            f"Signal conflict: {c.topic} - {c.description}"
            for c in analysis.signal_conflicts[:2]
        ]
        if analysis.risk_assessment:
            rk = analysis.risk_assessment
            all_rf = (
                getattr(rk, "market_risks", [])
                + getattr(rk, "company_risks", [])
                + getattr(rk, "financial_risks", [])
            )
            if all_rf:
                important_risks.extend(
                    [getattr(rf, "name", str(rf)) for rf in all_rf[:2]]
                )
        if not important_risks:
            important_risks.append(
                "Potential price volatility arising from conflicting specialist signals"
            )

    elif analysis.confidence >= 0.65 and analysis.data_completeness_ratio >= 0.6:
        fund_favorable = (
            analysis.fundamental_assessment is not None
            and _val(analysis.fundamental_assessment.overall_assessment).lower()
            == "favorable"
        )
        tech_uptrend = (
            analysis.technical_assessment is not None
            and _val(analysis.technical_assessment.trend).lower() == "uptrend"
        )
        fund_unfavorable = analysis.fundamental_assessment is not None and _val(
            analysis.fundamental_assessment.overall_assessment
        ).lower() in ("unfavorable", "bearish")
        tech_downtrend = analysis.technical_assessment is not None and _val(
            analysis.technical_assessment.trend
        ).lower() in ("downtrend", "bearish")

        if fund_favorable and tech_uptrend:
            stance = RecommendationStance.FAVORABLE
            rationale = (
                f"Constructive analytical signals observed for {ticker}, supported "
                "by robust fundamental health and positive technical momentum."
            )
        elif fund_unfavorable and tech_downtrend:
            stance = RecommendationStance.UNFAVORABLE
            rationale = (
                f"Unfavorable analytical outlook for {ticker}, driven by operational "
                "fundamental headwinds and persistent technical downtrend."
            )
        else:
            stance = RecommendationStance.NEUTRAL
            rationale = (
                f"Specialist findings present balanced indicators for {ticker} "
                "across fundamental and technical factors."
            )

        horizon_str = inv_prof.time_horizon if inv_prof else "3-5 years"
        goal_str = inv_prof.investment_goal if inv_prof else "growth"
        profile_alignment = (
            f"Findings evaluated relative to stated {goal_str} goal and {horizon_str} "
            "horizon."
        )
        monitoring_points = [
            "Next quarterly earnings disclosure",
            "Price action relative to primary technical support levels",
        ]

        key_reasons = [
            f"{a.topic}: {a.summary}" for a in analysis.areas_of_agreement[:3]
        ]
        if not key_reasons:
            if fund_favorable:
                key_reasons.append(
                    f"Favorable financial health and balance sheet strength for "
                    f"{ticker}"
                )
            if tech_uptrend:
                key_reasons.append(
                    f"Technical momentum holding firmly above moving averages for "
                    f"{ticker}"
                )
            if not key_reasons:
                key_reasons.append(
                    f"Structured analytical coverage across "
                    f"{len(analysis.available_specialists)} specialist domains"
                )

        important_risks = []
        if analysis.risk_assessment:
            rk = analysis.risk_assessment
            all_rf = (
                getattr(rk, "market_risks", [])
                + getattr(rk, "company_risks", [])
                + getattr(rk, "financial_risks", [])
            )
            if all_rf:
                important_risks.extend(
                    [getattr(rf, "name", str(rf)) for rf in all_rf[:3]]
                )
        if not important_risks:
            important_risks.append(
                "Standard market, equity volatility, and operational risks"
            )

    else:
        stance = RecommendationStance.NEUTRAL
        rationale = (
            f"Specialist findings present balanced signals for {ticker} with "
            "moderate data coverage."
        )
        profile_alignment = (
            "Baseline analytical posture across available specialist coverage."
        )
        monitoring_points = [
            "Upcoming corporate financial disclosures",
            "Key support and resistance technical levels",
        ]
        key_reasons = [
            f"Synthesized perspective across {len(analysis.available_specialists)} "
            "available specialist domains"
        ]
        important_risks = [
            "General equity market risk and potential sector-level headwinds"
        ]

    rec = ReportRecommendation(
        stance=stance,
        rationale=rationale,
        profile_alignment=profile_alignment,
        monitoring_points=monitoring_points,
        time_horizon_suitability=(inv_prof.time_horizon if inv_prof else None),
        risk_tolerance_suitability=(inv_prof.risk_tolerance if inv_prof else None),
    )

    return FinalReport.from_unified_analysis(
        analysis=analysis,
        recommendation=rec,
        key_reasons=key_reasons,
        important_risks=important_risks,
        report_id=report_id,
    )


# ===========================================================================
# REPORT GENERATOR AGENT (Phase 12.2)
# ===========================================================================


class ReportGeneratorAgent(BaseAgent):
    """Report Generator Agent for FinPilot (Phase 12.2).

    Coordinates the generation of the final structured, explainable investment
    report. Combines deterministic structural mapping with grounded LLM synthesis.
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
        return "ReportGenerator"

    @property
    def input_schema(self) -> Any:
        """Expected input schema model."""
        return ReportGeneratorInput

    @property
    def output_schema(self) -> Any:
        """Structured output schema model."""
        return FinalReport

    @property
    def provider(self) -> LLMProvider:
        """Lazy-initialize or return assigned LLMProvider."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def run(
        self,
        input_data: Union[
            ReportGeneratorInput, UnifiedSpecialistAnalysis, GraphState, Dict[str, Any]
        ],
    ) -> AgentResult:
        """Execute final report generation.

        Args:
            input_data: ReportGeneratorInput, UnifiedSpecialistAnalysis, GraphState,
                        or dictionary containing 'aggregated_result'.

        Returns:
            AgentResult: Successful result containing FinalReport (InvestmentReport),
            or failure result with descriptive error.
        """
        # 1. Parse and normalize input
        if isinstance(input_data, ReportGeneratorInput):
            parsed_input = input_data
        elif isinstance(input_data, UnifiedSpecialistAnalysis):
            parsed_input = ReportGeneratorInput.from_unified_analysis(input_data)
        elif isinstance(input_data, dict):
            try:
                if "aggregated_analysis" in input_data:
                    parsed_input = ReportGeneratorInput.model_validate(input_data)
                elif "aggregated_result" in input_data:
                    parsed_input = ReportGeneratorInput.from_graph_state(input_data)
                elif "ticker" in input_data and "specialist_statuses" in input_data:
                    analysis = UnifiedSpecialistAnalysis.model_validate(input_data)
                    parsed_input = ReportGeneratorInput.from_unified_analysis(analysis)
                else:
                    return AgentResult.create_failure(
                        error=(
                            "Unable to parse input: missing 'aggregated_analysis' "
                            "or 'aggregated_result'."
                        )
                    )
            except Exception as err:
                logger.error("Failed to parse ReportGeneratorInput: %s", err)
                return AgentResult.create_failure(
                    error=f"Invalid report generator input payload: {err}"
                )
        else:
            return AgentResult.create_failure(
                error=(
                    "Input must be an instance of ReportGeneratorInput, "
                    "UnifiedSpecialistAnalysis, or dict."
                )
            )

        analysis = parsed_input.aggregated_analysis

        # 2. Build deterministic baseline report
        det_report = generate_deterministic_report(analysis)

        # 3. LLM structured synthesis (if enabled and not deterministic_only)
        final_report = det_report
        if not self.deterministic_only and not analysis.insufficient_evidence:
            prompt = format_report_generator_prompt(analysis)
            try:
                llm_output: GeneratedReportSynthesis = generate_structured(
                    provider=self.provider,
                    prompt=prompt,
                    schema=GeneratedReportSynthesis,
                )

                candidate_rec = ReportRecommendation(
                    stance=llm_output.recommendation_stance,
                    rationale=llm_output.recommendation_rationale,
                    profile_alignment=llm_output.profile_alignment,
                    monitoring_points=llm_output.monitoring_points,
                    time_horizon_suitability=(
                        analysis.investor_profile.time_horizon
                        if analysis.investor_profile
                        else None
                    ),
                    risk_tolerance_suitability=(
                        analysis.investor_profile.risk_tolerance
                        if analysis.investor_profile
                        else None
                    ),
                )

                candidate_report = FinalReport.from_unified_analysis(
                    analysis=analysis,
                    recommendation=candidate_rec,
                    key_reasons=llm_output.key_reasons,
                    important_risks=llm_output.important_risks,
                )

                if llm_output.executive_synthesis:
                    candidate_report.overall_assessment.synthesis = (
                        llm_output.executive_synthesis
                    )

                # Validate grounding and safety
                validate_report_grounding(candidate_report, analysis)
                final_report = candidate_report

            except (
                LLMError,
                LLMStructuredOutputError,
                ReportValidationError,
                Exception,
            ) as err:
                logger.warning(
                    "LLM structured report generation failed or violated grounding; "
                    "falling back to deterministic report: %s",
                    err,
                )
                final_report = det_report

        # 4. Final safety validation
        try:
            validate_report_grounding(final_report, analysis)
        except ReportValidationError as val_err:
            logger.error("Final report failed grounding validation: %s", val_err)
            return AgentResult.create_failure(error=str(val_err))

        return AgentResult.create_success(
            data=final_report,
            confidence=final_report.confidence,
        )


# ===========================================================================
# LANGGRAPH NODE ADAPTER (Phase 12.2)
# ===========================================================================


def report_generator_node(
    state: GraphState,
    agent: Optional[ReportGeneratorAgent] = None,
) -> Dict[str, Any]:
    """LangGraph node adapter for the Report Generator Agent.

    Consumes 'aggregated_result' from GraphState, executes report generation,
    and returns state update mapping for 'report'.

    Args:
        state: Current GraphState dictionary.
        agent: Optional pre-configured ReportGeneratorAgent.

    Returns:
        Dict[str, Any]: State update mapping for 'report'.
    """
    active_agent = agent or ReportGeneratorAgent()
    try:
        gen_input = ReportGeneratorInput.from_graph_state(state)
    except Exception as err:
        logger.error(
            "Failed to construct ReportGeneratorInput from GraphState: %s", err
        )
        fail_result = AgentResult.create_failure(
            error=f"Failed to assemble report generator input: {err}"
        )
        return fail_result.to_state_update("report")

    result = active_agent.run(gen_input)
    return result.to_state_update("report")
