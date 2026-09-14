"""Report Validation and Consistency Engine (Phase 12.3).

Validates that the generated FinalReport (InvestmentReport) produced by Phase 12.2
is structurally valid, internally consistent, evidence-grounded, and safe:
- Structural validation (schema conformity, identity preservation, context).
- Specialist consistency (aligns with Phase 11 inputs without contradictions).
- Recommendation consistency (grounded in evidence without inventing stances).
- Numerical provenance (specialist-aware, citation-aware quantitative validation).
- Source / evidence consistency (no phantom citations, correct attribution).
- Cross-field consistency (no internal contradictions across sections).
- Completeness validation (partial reports distinguished from complete ones).
- Safety validation (no buy/sell orders, no price targets, no guarantees).
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.aggregator_consistency import (
    EXCLUDED_NUMBERS,
    _number_in_set,
    _val,
    extract_numbers_from_text,
)
from app.agents.aggregator_schema import (
    PROHIBITED_ADVICE_PATTERNS,
    AggregatedEvidenceItem,
    SpecialistStatus,
    SpecialistType,
    UnifiedSpecialistAnalysis,
)
from app.agents.report_generator import (
    CITATION_REF_PATTERN,
)
from app.agents.report_schema import (
    FinalReport,
    RecommendationStance,
    ReportGeneratorInput,
)
from app.core.logging import get_logger

logger = get_logger("app.agents.report_consistency")


# ===========================================================================
# SPECIALIST DOMAIN KEYWORDS & HELPERS (Phase 12.3.4 & 11.3 Alignment)
# ===========================================================================

TECH_KEYWORDS: Set[str] = {
    "rsi",
    "sma",
    "macd",
    "price",
    "trend",
    "support",
    "resistance",
    "breakout",
    "moving average",
    "momentum",
    "overbought",
    "oversold",
    "consolidation",
}

FUND_KEYWORDS: Set[str] = {
    "revenue",
    "margin",
    "gross margin",
    "operating margin",
    "profit",
    "cash flow",
    "free cash flow",
    "fcf",
    "debt",
    "net cash",
    "ebitda",
    "pe ratio",
    "p/e",
    "valuation",
    "financial health",
    "interest coverage",
    "growth",
}

NEWS_KEYWORDS: Set[str] = {
    "news",
    "headline",
    "sentiment",
    "reuters",
    "article",
    "event",
    "announcement",
    "coverage",
}

RESEARCH_KEYWORDS: Set[str] = {
    "10-k",
    "10-q",
    "sec filing",
    "filing",
    "annual report",
    "services segment",
    "chunk",
    "document",
}

RISK_KEYWORDS: Set[str] = {
    "litigation",
    "antitrust",
    "volatility",
    "downside",
    "hazard",
    "regulatory scrutiny",
    "macro volatility",
    "interest rate sensitivity",
}

INVESTOR_PROFILE_KEYWORDS: Set[str] = {
    "capital",
    "portfolio",
    "allocation",
    "budget",
    "planned capital",
    "investor profile",
    "investment amount",
    "capital amount",
}


def extract_specialist_numbers_from_analysis(
    analysis: UnifiedSpecialistAnalysis,
) -> Dict[str, Set[float]]:
    """Partition numerical values from analysis by their specialist source."""
    result: Dict[str, Set[float]] = {
        "technical": set(),
        "fundamental": set(),
        "news": set(),
        "research": set(),
        "risk": set(),
    }

    def _collect_into(obj: Any, target: Set[float]) -> None:
        if obj is None:
            return
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            val = float(obj)
            if val not in EXCLUDED_NUMBERS:
                target.add(val)
        elif isinstance(obj, str):
            target.update(extract_numbers_from_text(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect_into(v, target)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _collect_into(item, target)
        elif hasattr(obj, "model_dump"):
            _collect_into(obj.model_dump(), target)
        elif hasattr(obj, "__dict__"):
            _collect_into(obj.__dict__, target)

    if analysis.technical_assessment:
        _collect_into(analysis.technical_assessment, result["technical"])
    if analysis.fundamental_assessment:
        _collect_into(analysis.fundamental_assessment, result["fundamental"])
    if analysis.news_assessment:
        _collect_into(analysis.news_assessment, result["news"])
    if analysis.research_assessment:
        _collect_into(analysis.research_assessment, result["research"])
    if analysis.risk_assessment:
        _collect_into(analysis.risk_assessment, result["risk"])

    for ev in analysis.aggregated_evidence:
        s_name = _val(ev.specialist).lower()
        if s_name in result:
            result[s_name].update(extract_numbers_from_text(ev.detail))

    return result


def extract_evidence_reference_numbers(
    analysis: UnifiedSpecialistAnalysis,
) -> Dict[str, Tuple[str, Set[float]]]:
    """Map evidence reference ID to its specialist name and numbers."""
    ref_map: Dict[str, Tuple[str, Set[float]]] = {}
    for ev in analysis.aggregated_evidence:
        s_name = _val(ev.specialist).lower()
        nums = extract_numbers_from_text(ev.detail)
        ref_id = ev.reference_id.strip().lower()
        ref_map[ref_id] = (s_name, nums)
        if ev.chunk_id:
            ref_map[ev.chunk_id.strip().lower()] = (s_name, nums)
        if ev.document_id:
            ref_map[ev.document_id.strip().lower()] = (s_name, nums)
    return ref_map


def get_specialist_stance_from_analysis(
    specialist_name: str,
    analysis: UnifiedSpecialistAnalysis,
) -> Optional[str]:
    """Return deterministic directional stance: 'positive', 'negative', 'neutral'."""
    s_clean = specialist_name.strip().lower()
    if s_clean == "technical":
        if analysis.technical_assessment:
            trend = _val(analysis.technical_assessment.trend).lower()
            if trend in ("uptrend", "bullish"):
                return "positive"
            if trend in ("downtrend", "bearish"):
                return "negative"
            if trend in ("sideways", "neutral"):
                return "neutral"
        return None
    if s_clean == "fundamental":
        if analysis.fundamental_assessment:
            assess = _val(analysis.fundamental_assessment.overall_assessment).lower()
            if assess in ("favorable", "strong", "positive"):
                return "positive"
            if assess in ("unfavorable", "weak", "negative"):
                return "negative"
            if assess in ("neutral",):
                return "neutral"
        return None
    if s_clean == "news":
        if analysis.news_assessment:
            sent = _val(analysis.news_assessment.overall_sentiment).lower()
            if sent in ("positive",):
                return "positive"
            if sent in ("negative",):
                return "negative"
            if sent in ("neutral",):
                return "neutral"
        return None
    if s_clean == "risk":
        if analysis.risk_assessment:
            lvl = _val(analysis.risk_assessment.overall_risk_level).lower()
            if lvl in ("low", "minimal"):
                return "positive"
            if lvl in ("high", "critical"):
                return "negative"
            if lvl in ("moderate",):
                return "neutral"
        return None
    return None


# ===========================================================================
# 1. VALIDATION RESULT SCHEMA (Phase 12.3.9)
# ===========================================================================


class ReportValidationStatus(str, Enum):
    """Evaluation status of an individual report validation check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ReportValidationSeverity(str, Enum):
    """Severity of a detected validation condition."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReportValidationIssue(BaseModel):
    """An individual validation check outcome or detected inconsistency."""

    model_config = ConfigDict(extra="ignore")

    check_id: str = Field(
        ...,
        description="Unique check identifier, e.g. 'CHK_STRUCT_TICKER_MATCH'.",
    )
    status: ReportValidationStatus = Field(
        ...,
        description="Outcome of check: passed, warning, or failed.",
    )
    severity: ReportValidationSeverity = Field(
        ...,
        description="Severity level: low, medium, high, or critical.",
    )
    category: str = Field(
        ...,
        description=(
            "Validation category: structural, specialist_consistency, "
            "recommendation_consistency, numerical_provenance, evidence_consistency, "
            "cross_field, completeness, safety."
        ),
    )
    affected_fields: List[str] = Field(
        default_factory=list,
        description="Report fields or specialist names involved in this issue.",
    )
    message: str = Field(
        ...,
        description="Concise description of the condition found.",
    )
    explanation: str = Field(
        ...,
        description="Detailed explanation covering expected vs actual condition.",
    )

    @field_validator("check_id", "category", "message", "explanation")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace.")
        return cleaned


class ReportValidationResult(BaseModel):
    """Structured validation report for FinalReport (Phase 12.3)."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(
        ...,
        description="Uppercase ticker symbol for the target company.",
    )
    is_valid: bool = Field(
        ...,
        description="True if no checks failed with high or critical severity.",
    )
    passed_checks_count: int = Field(
        default=0,
        ge=0,
        description="Count of checks that passed.",
    )
    warnings_count: int = Field(
        default=0,
        ge=0,
        description="Count of checks resulting in warnings.",
    )
    failures_count: int = Field(
        default=0,
        ge=0,
        description="Count of checks that failed.",
    )
    issues: List[ReportValidationIssue] = Field(
        default_factory=list,
        description="List of all evaluated check issues and passes.",
    )
    summary: str = Field(
        ...,
        description="Executive summary of the report validation results.",
    )

    # Convenience properties
    @property
    def errors(self) -> List[ReportValidationIssue]:
        """List of all failed validation issues (high/critical)."""
        return [i for i in self.issues if i.status == ReportValidationStatus.FAILED]

    @property
    def warnings(self) -> List[ReportValidationIssue]:
        """List of all warning validation issues (low/medium)."""
        return [i for i in self.issues if i.status == ReportValidationStatus.WARNING]

    @property
    def unsupported_claims(self) -> List[ReportValidationIssue]:
        """Issues related to ungrounded or unsupported claims."""
        return [
            i
            for i in self.issues
            if i.category in ("numerical_provenance", "evidence_consistency")
            and i.status != ReportValidationStatus.PASSED
        ]

    @property
    def numerical_issues(self) -> List[ReportValidationIssue]:
        """Issues related to quantitative value provenance."""
        return [
            i
            for i in self.issues
            if i.category == "numerical_provenance"
            and i.status != ReportValidationStatus.PASSED
        ]

    @property
    def evidence_issues(self) -> List[ReportValidationIssue]:
        """Issues related to citation or evidence consistency."""
        return [
            i
            for i in self.issues
            if i.category == "evidence_consistency"
            and i.status != ReportValidationStatus.PASSED
        ]

    @property
    def safety_issues(self) -> List[ReportValidationIssue]:
        """Issues related to regulatory safety or prohibited advice."""
        return [
            i
            for i in self.issues
            if i.category == "safety" and i.status != ReportValidationStatus.PASSED
        ]

    @property
    def completeness_issues(self) -> List[ReportValidationIssue]:
        """Issues related to report completeness and status tracking."""
        return [
            i
            for i in self.issues
            if i.category == "completeness"
            and i.status != ReportValidationStatus.PASSED
        ]


# ===========================================================================
# 2. VALIDATION ENGINE IMPLEMENTATION
# ===========================================================================


class ReportConsistencyValidator:
    """Deterministic validation engine for FinPilot Final Investment Reports.

    Operates completely deterministically without requiring live LLM calls.
    Validates reports against upstream analysis without acting as a
    recommendation engine.
    """

    def validate(
        self,
        report: Union[FinalReport, Any],
        source_data: Union[UnifiedSpecialistAnalysis, ReportGeneratorInput],
    ) -> ReportValidationResult:
        """Run all Phase 12.3 validation suites against the report."""
        if hasattr(report, "data") and isinstance(getattr(report, "data"), FinalReport):
            report = getattr(report, "data")
        elif not isinstance(report, FinalReport):
            raise TypeError(
                f"Unsupported report type: {type(report)}. "
                "Expected FinalReport or AgentResult[FinalReport]."
            )

        if isinstance(source_data, ReportGeneratorInput):
            analysis = source_data.aggregated_analysis
        elif isinstance(source_data, UnifiedSpecialistAnalysis):
            analysis = source_data
        else:
            raise TypeError(
                f"Unsupported source_data type: {type(source_data)}. "
                "Expected UnifiedSpecialistAnalysis or ReportGeneratorInput."
            )

        issues: List[ReportValidationIssue] = []

        # 1. Structural Validation
        self._validate_structural(report, analysis, issues)

        # 2. Specialist Consistency
        self._validate_specialist_consistency(report, analysis, issues)

        # 3. Recommendation Consistency (Grounding, Evidence Support,
        #    Honest Representation)
        self._validate_recommendation_consistency(report, analysis, issues)

        # 4. Numerical Provenance (Specialist-aware, citation-aware, profile-isolated)
        self._validate_numerical_provenance(report, analysis, issues)

        # 5. Source / Evidence Consistency
        self._validate_evidence_consistency(report, analysis, issues)

        # 6. Cross-Field Consistency
        self._validate_cross_field_consistency(report, analysis, issues)

        # 7. Completeness Validation
        self._validate_completeness(report, analysis, issues)

        # 8. Regulatory Safety Validation
        self._validate_safety(report, analysis, issues)

        # Tabulate counts
        passed_count = sum(
            1 for i in issues if i.status == ReportValidationStatus.PASSED
        )
        warnings_count = sum(
            1 for i in issues if i.status == ReportValidationStatus.WARNING
        )
        failures_count = sum(
            1 for i in issues if i.status == ReportValidationStatus.FAILED
        )

        is_valid = failures_count == 0

        # Construct summary
        if is_valid and warnings_count == 0:
            summary = (
                f"Report for {report.ticker} passed all {len(issues)} consistency "
                "and grounding checks. No errors or warnings detected."
            )
        elif is_valid:
            summary = (
                f"Report for {report.ticker} is valid with "
                f"{warnings_count} non-blocking warning(s). "
                f"All {passed_count} checks passed."
            )
        else:
            error_msgs = [
                f"[{i.check_id}] {i.message}"
                for i in issues
                if i.status == ReportValidationStatus.FAILED
            ]
            summary = (
                f"Report for {report.ticker} failed validation with {failures_count} "
                f"blocking error(s) and {warnings_count} warning(s): "
                f"{'; '.join(error_msgs[:3])}"
            )

        return ReportValidationResult(
            ticker=report.ticker,
            is_valid=is_valid,
            passed_checks_count=passed_count,
            warnings_count=warnings_count,
            failures_count=failures_count,
            issues=issues,
            summary=summary,
        )

    # -----------------------------------------------------------------------
    # 1. STRUCTURAL VALIDATION (12.3.1)
    # -----------------------------------------------------------------------

    def _validate_structural(
        self,
        report: FinalReport,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ReportValidationIssue],
    ) -> None:
        """Verify report conforms to structural contracts and preserves identity."""
        # Check 1.1: Ticker identity match
        if report.ticker != analysis.ticker:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_STRUCT_TICKER_MATCH",
                    status=ReportValidationStatus.FAILED,
                    severity=ReportValidationSeverity.CRITICAL,
                    category="structural",
                    affected_fields=["company.ticker"],
                    message=(
                        f"Ticker mismatch: report has '{report.ticker}', "
                        f"source analysis has '{analysis.ticker}'."
                    ),
                    explanation="Report ticker must exactly match source ticker.",
                )
            )
        else:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_STRUCT_TICKER_MATCH",
                    status=ReportValidationStatus.PASSED,
                    severity=ReportValidationSeverity.LOW,
                    category="structural",
                    affected_fields=["company.ticker"],
                    message=f"Ticker '{report.ticker}' matches source analysis.",
                    explanation="Target company ticker preserved correctly.",
                )
            )

        # Check 1.2: Company name consistency
        if analysis.target_company and report.company.name:
            if analysis.target_company.lower() != report.company.name.lower():
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_STRUCT_COMPANY_NAME_MATCH",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="structural",
                        affected_fields=["company.name"],
                        message=(
                            f"Company name mismatch: report has "
                            f"'{report.company.name}', "
                            f"source has '{analysis.target_company}'."
                        ),
                        explanation="Report company name must match upstream company.",
                    )
                )
            else:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_STRUCT_COMPANY_NAME_MATCH",
                        status=ReportValidationStatus.PASSED,
                        severity=ReportValidationSeverity.LOW,
                        category="structural",
                        affected_fields=["company.name"],
                        message="Company name matches source analysis.",
                        explanation="Target company name preserved correctly.",
                    )
                )

        # Check 1.3: Investor profile preservation
        if analysis.investor_profile:
            src_prof = analysis.investor_profile
            rep_prof = report.investor_profile
            if rep_prof is None:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_STRUCT_PROFILE_PRESERVED",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="structural",
                        affected_fields=["investor_profile"],
                        message="Investor profile in source omitted from report.",
                        explanation="Investor profile must be preserved in the report.",
                    )
                )
            else:
                mismatches: List[str] = []
                if (
                    src_prof.investment_goal
                    and rep_prof.investment_goal != src_prof.investment_goal
                ):
                    mismatches.append(
                        f"goal: '{rep_prof.investment_goal}' != "
                        f"'{src_prof.investment_goal}'"
                    )
                if (
                    src_prof.risk_tolerance
                    and rep_prof.risk_tolerance != src_prof.risk_tolerance
                ):
                    mismatches.append(
                        f"risk_tolerance: '{rep_prof.risk_tolerance}' != "
                        f"'{src_prof.risk_tolerance}'"
                    )

                if mismatches:
                    issues.append(
                        ReportValidationIssue(
                            check_id="CHK_STRUCT_PROFILE_PRESERVED",
                            status=ReportValidationStatus.FAILED,
                            severity=ReportValidationSeverity.HIGH,
                            category="structural",
                            affected_fields=["investor_profile"],
                            message=(
                                f"Investor profile attributes mismatch: "
                                f"{', '.join(mismatches)}."
                            ),
                            explanation=(
                                "Investor profile fields must align with source."
                            ),
                        )
                    )
                else:
                    issues.append(
                        ReportValidationIssue(
                            check_id="CHK_STRUCT_PROFILE_PRESERVED",
                            status=ReportValidationStatus.PASSED,
                            severity=ReportValidationSeverity.LOW,
                            category="structural",
                            affected_fields=["investor_profile"],
                            message="Investor profile attributes preserved accurately.",
                            explanation=(
                                "Investor profile fields must align with source."
                            ),
                        )
                    )

        # Check 1.4: Horizon preservation
        if analysis.investor_profile and analysis.investor_profile.time_horizon:
            src_horizon = analysis.investor_profile.time_horizon
            rep_horizon = getattr(
                report, "horizon", getattr(report, "investment_horizon", None)
            )
            if rep_horizon != src_horizon:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_STRUCT_HORIZON_MATCH",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="structural",
                        affected_fields=["horizon"],
                        message=(
                            f"Investment horizon mismatch: report has "
                            f"'{rep_horizon}', profile specified '{src_horizon}'."
                        ),
                        explanation="Report investment horizon must reflect profile.",
                    )
                )
            else:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_STRUCT_HORIZON_MATCH",
                        status=ReportValidationStatus.PASSED,
                        severity=ReportValidationSeverity.LOW,
                        category="structural",
                        affected_fields=["horizon"],
                        message="Investment horizon matches investor profile.",
                        explanation="Horizon value preserved accurately.",
                    )
                )

        # Check 1.5: Capital preservation
        if (
            analysis.investor_profile
            and analysis.investor_profile.capital_amount is not None
        ):
            src_cap = float(analysis.investor_profile.capital_amount)
            if report.capital is None or report.capital.amount is None:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_STRUCT_CAPITAL_MATCH",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="structural",
                        affected_fields=["capital"],
                        message=(
                            "Planned capital in investor profile was omitted in report."
                        ),
                        explanation=(
                            "Report must preserve investor capital if provided."
                        ),
                    )
                )
            elif not math.isclose(
                report.capital.amount, src_cap, rel_tol=1e-2, abs_tol=1e-2
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_STRUCT_CAPITAL_MATCH",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="structural",
                        affected_fields=["capital.amount"],
                        message=(
                            f"Capital amount mismatch: report has "
                            f"{report.capital.amount}, "
                            f"source profile specified {src_cap}."
                        ),
                        explanation="Report capital must match profile capital.",
                    )
                )
            else:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_STRUCT_CAPITAL_MATCH",
                        status=ReportValidationStatus.PASSED,
                        severity=ReportValidationSeverity.LOW,
                        category="structural",
                        affected_fields=["capital.amount"],
                        message=(
                            f"Capital amount {report.capital.amount} matches profile."
                        ),
                        explanation="Capital amount preserved accurately.",
                    )
                )

    # -----------------------------------------------------------------------
    # 2. SPECIALIST CONSISTENCY (12.3.2)
    # -----------------------------------------------------------------------

    def _validate_specialist_consistency(
        self,
        report: FinalReport,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ReportValidationIssue],
    ) -> None:
        """Verify specialist section presence and statuses align with upstream data."""
        specialist_names: List[SpecialistType] = [
            "technical",
            "fundamental",
            "news",
            "research",
            "risk",
        ]

        for spec in specialist_names:
            src_status = analysis.specialist_statuses.get(
                spec, SpecialistStatus.MISSING
            )
            rep_sec = getattr(report, spec, None)
            rep_status = report.specialist_statuses.get(spec, SpecialistStatus.MISSING)

            # Check 2.1: Missing specialist presented as available
            if src_status == SpecialistStatus.MISSING:
                if rep_sec is not None:
                    issues.append(
                        ReportValidationIssue(
                            check_id=f"CHK_SPEC_MISSING_{spec.upper()}_POPULATED",
                            status=ReportValidationStatus.FAILED,
                            severity=ReportValidationSeverity.CRITICAL,
                            category="specialist_consistency",
                            affected_fields=[spec],
                            message=(
                                f"Specialist '{spec}' is MISSING in source "
                                f"analysis but "
                                "populated in report section."
                            ),
                            explanation=(
                                "Report must not invent sections for missing "
                                "specialists."
                            ),
                        )
                    )
                elif rep_status == SpecialistStatus.AVAILABLE:
                    issues.append(
                        ReportValidationIssue(
                            check_id=f"CHK_SPEC_MISSING_{spec.upper()}_STATUS",
                            status=ReportValidationStatus.FAILED,
                            severity=ReportValidationSeverity.HIGH,
                            category="specialist_consistency",
                            affected_fields=[f"specialist_statuses.{spec}"],
                            message=(
                                f"Specialist '{spec}' is marked AVAILABLE in report "
                                f"status "
                                "map but was MISSING in upstream analysis."
                            ),
                            explanation=(
                                "Report specialist_statuses must reflect upstream."
                            ),
                        )
                    )
                else:
                    issues.append(
                        ReportValidationIssue(
                            check_id=f"CHK_SPEC_{spec.upper()}_STATUS_CONSISTENT",
                            status=ReportValidationStatus.PASSED,
                            severity=ReportValidationSeverity.LOW,
                            category="specialist_consistency",
                            affected_fields=[spec],
                            message=f"Missing specialist '{spec}' correctly omitted.",
                            explanation="Omission preserved faithfully.",
                        )
                    )

            # Check 2.2: Failed specialist presented as successful
            elif src_status == SpecialistStatus.FAILED:
                if rep_sec is not None:
                    issues.append(
                        ReportValidationIssue(
                            check_id=f"CHK_SPEC_FAILED_{spec.upper()}_POPULATED",
                            status=ReportValidationStatus.FAILED,
                            severity=ReportValidationSeverity.CRITICAL,
                            category="specialist_consistency",
                            affected_fields=[spec],
                            message=(
                                f"Specialist '{spec}' FAILED in source analysis but "
                                "is populated in report as successful."
                            ),
                            explanation=(
                                "Report must not treat failed specialist output as "
                                "data."
                            ),
                        )
                    )
                elif rep_status != SpecialistStatus.FAILED:
                    issues.append(
                        ReportValidationIssue(
                            check_id=f"CHK_SPEC_FAILED_{spec.upper()}_STATUS",
                            status=ReportValidationStatus.FAILED,
                            severity=ReportValidationSeverity.HIGH,
                            category="specialist_consistency",
                            affected_fields=[f"specialist_statuses.{spec}"],
                            message=(
                                f"Specialist '{spec}' status is '{rep_status}' in "
                                f"report "
                                "but FAILED in upstream analysis."
                            ),
                            explanation=(
                                "Report specialist_statuses must preserve FAILED "
                                "status."
                            ),
                        )
                    )
                else:
                    issues.append(
                        ReportValidationIssue(
                            check_id=f"CHK_SPEC_{spec.upper()}_STATUS_CONSISTENT",
                            status=ReportValidationStatus.PASSED,
                            severity=ReportValidationSeverity.LOW,
                            category="specialist_consistency",
                            affected_fields=[spec],
                            message=f"Failed specialist '{spec}' recorded as FAILED.",
                            explanation="Failure status preserved faithfully.",
                        )
                    )

            # Check 2.3: Available specialist consistency
            elif src_status == SpecialistStatus.AVAILABLE:
                if rep_sec is None:
                    issues.append(
                        ReportValidationIssue(
                            check_id=f"CHK_SPEC_AVAIL_{spec.upper()}_OMITTED",
                            status=ReportValidationStatus.WARNING,
                            severity=ReportValidationSeverity.MEDIUM,
                            category="specialist_consistency",
                            affected_fields=[spec],
                            message=(
                                f"Specialist '{spec}' is AVAILABLE in source "
                                f"analysis but "
                                "omitted from report section."
                            ),
                            explanation="Available specialist output was not rendered.",
                        )
                    )
                else:
                    issues.append(
                        ReportValidationIssue(
                            check_id=f"CHK_SPEC_{spec.upper()}_STATUS_CONSISTENT",
                            status=ReportValidationStatus.PASSED,
                            severity=ReportValidationSeverity.LOW,
                            category="specialist_consistency",
                            affected_fields=[spec],
                            message=f"Available specialist '{spec}' section rendered.",
                            explanation="Section matches upstream availability.",
                        )
                    )

        # Check 2.4: Directional Contradiction Detection
        if report.technical and analysis.technical_assessment:
            tech_trend = _val(analysis.technical_assessment.trend).lower()
            tech_narrative = (
                report.technical.summary + " " + (report.technical.momentum or "")
            ).lower()
            if (
                tech_trend in ("downtrend", "bearish")
                and "strong uptrend" in tech_narrative
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_SPEC_TECH_CONTRADICTION",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="specialist_consistency",
                        affected_fields=["technical.summary"],
                        message=(
                            "Technical narrative claims 'strong uptrend' while trend "
                            f"is '{tech_trend}'."
                        ),
                        explanation=(
                            "Technical section must not contradict trend metric."
                        ),
                    )
                )

        if report.fundamental and analysis.fundamental_assessment:
            fund_assess = _val(
                analysis.fundamental_assessment.overall_assessment
            ).lower()
            fund_narrative = report.fundamental.summary.lower()
            if (
                fund_assess in ("unfavorable", "weak")
                and "exceptionally strong fundamentals" in fund_narrative
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_SPEC_FUND_CONTRADICTION",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="specialist_consistency",
                        affected_fields=["fundamental.summary"],
                        message=(
                            "Fundamental narrative claims 'exceptionally strong "
                            f"fundamentals' while assessment is '{fund_assess}'."
                        ),
                        explanation="Fundamental summary must not contradict rating.",
                    )
                )

    # -----------------------------------------------------------------------
    # 3. RECOMMENDATION CONSISTENCY (12.3.3)
    # -----------------------------------------------------------------------

    def _validate_recommendation_consistency(
        self,
        report: FinalReport,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ReportValidationIssue],
    ) -> None:
        """Verify recommendation stance is supported by evidence and conflicts.

        Acts strictly as a validation layer:
        - Does NOT replace or mutate the recommendation.
        - Does NOT force a specific stance (e.g. CAUTIOUS) when disagreement exists.
        - Verifies that insufficient evidence is honestly represented.
        - Verifies that material signal conflicts are acknowledged where required.
        - Verifies directional sanity against unanimous specialist evidence.
        """
        if not report.recommendation:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_REC_POPULATED",
                    status=ReportValidationStatus.WARNING,
                    severity=ReportValidationSeverity.LOW,
                    category="recommendation_consistency",
                    affected_fields=["recommendation"],
                    message="Report recommendation is not yet populated.",
                    explanation="Recommendation is optional in partial reports.",
                )
            )
            return

        stance = report.recommendation.stance

        # Check 3.1: Insufficient evidence status consistency
        is_insufficient = analysis.insufficient_evidence or (
            analysis.data_completeness_ratio is not None
            and analysis.data_completeness_ratio < 0.4
        )
        if is_insufficient:
            if stance != RecommendationStance.INSUFFICIENT_EVIDENCE:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_REC_INSUFFICIENT_EVIDENCE_STANCE",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.CRITICAL,
                        category="recommendation_consistency",
                        affected_fields=["recommendation.stance"],
                        message=(
                            f"Recommendation stance is '{stance.value}' but upstream "
                            "analysis records insufficient_evidence=True."
                        ),
                        explanation=(
                            "When evidence is insufficient, stance must be "
                            "'insufficient_evidence'."
                        ),
                    )
                )
            else:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_REC_INSUFFICIENT_EVIDENCE_STANCE",
                        status=ReportValidationStatus.PASSED,
                        severity=ReportValidationSeverity.LOW,
                        category="recommendation_consistency",
                        affected_fields=["recommendation.stance"],
                        message=(
                            "Recommendation correctly adopts "
                            "'insufficient_evidence' stance."
                        ),
                        explanation="Aligned with data availability limitations.",
                    )
                )

        # Check 3.2: Signal Conflict and Tension Awareness
        has_conflicts = len(analysis.signal_conflicts) > 0
        if has_conflicts:
            rationale_lower = (
                (report.recommendation.rationale or "")
                + " "
                + (report.recommendation.profile_alignment or "")
            ).lower()
            explains_tension = any(
                k in rationale_lower
                for k in (
                    "conflict",
                    "tension",
                    "despite",
                    "divergence",
                    "caution",
                    "headwind",
                    "mixed",
                    "contrasting",
                    "balancing",
                )
            )

            # Strong directional stances (favorable / unfavorable) must
            # acknowledge conflicts
            if stance in (
                RecommendationStance.FAVORABLE,
                RecommendationStance.UNFAVORABLE,
            ):
                if not explains_tension:
                    issues.append(
                        ReportValidationIssue(
                            check_id="CHK_REC_CONFLICT_UNADDRESSED",
                            status=ReportValidationStatus.FAILED,
                            severity=ReportValidationSeverity.HIGH,
                            category="recommendation_consistency",
                            affected_fields=[
                                "recommendation.stance",
                                "recommendation.rationale",
                            ],
                            message=(
                                f"Report assigns definitive '{stance.value}' stance "
                                f"with {len(analysis.signal_conflicts)} unaddressed "
                                f"signal conflict(s)."
                            ),
                            explanation=(
                                "When signal conflicts exist, strong directional "
                                "stances "
                                "must acknowledge the tension or divergent signals."
                            ),
                        )
                    )
                else:
                    issues.append(
                        ReportValidationIssue(
                            check_id="CHK_REC_CONFLICT_AWARE",
                            status=ReportValidationStatus.PASSED,
                            severity=ReportValidationSeverity.LOW,
                            category="recommendation_consistency",
                            affected_fields=["recommendation.rationale"],
                            message=(
                                "Signal conflicts detected; rationale "
                                "acknowledges tension."
                            ),
                            explanation="Tension addressed in qualitative narrative.",
                        )
                    )
            else:
                # Cautious, Neutral, or other non-definitive posture with conflicts
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_REC_CONFLICT_AWARE",
                        status=ReportValidationStatus.PASSED,
                        severity=ReportValidationSeverity.LOW,
                        category="recommendation_consistency",
                        affected_fields=["recommendation.stance"],
                        message=(
                            f"Recommendation stance '{stance.value}' reflects "
                            "signal conflict awareness."
                        ),
                        explanation=(
                            "Non-definitive posture aligns with conflicting signals."
                        ),
                    )
                )

        # Check 3.3: Directional Support Against Unanimous Specialist Signals
        active_stances = [
            get_specialist_stance_from_analysis(s, analysis)
            for s in ("technical", "fundamental", "news", "risk")
        ]
        non_none_stances = [s for s in active_stances if s is not None]

        if len(non_none_stances) >= 2:
            if (
                all(s == "negative" for s in non_none_stances)
                and stance == RecommendationStance.FAVORABLE
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_REC_UNSUPPORTED_STANCE",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="recommendation_consistency",
                        affected_fields=["recommendation.stance"],
                        message=(
                            "Recommendation stance 'favorable' is unsupported "
                            "by evidence "
                            "and contradicts unanimous negative specialist inputs."
                        ),
                        explanation=(
                            "Directional recommendation must have evidence support."
                        ),
                    )
                )
            elif (
                all(s == "positive" for s in non_none_stances)
                and stance == RecommendationStance.UNFAVORABLE
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_REC_UNSUPPORTED_STANCE",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="recommendation_consistency",
                        affected_fields=["recommendation.stance"],
                        message=(
                            "Recommendation stance 'unfavorable' is unsupported "
                            "by evidence "
                            "and contradicts unanimous positive specialist inputs."
                        ),
                        explanation=(
                            "Directional recommendation must have evidence support."
                        ),
                    )
                )
            else:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_REC_EVIDENCE_SUPPORTED",
                        status=ReportValidationStatus.PASSED,
                        severity=ReportValidationSeverity.LOW,
                        category="recommendation_consistency",
                        affected_fields=["recommendation.stance"],
                        message=(
                            f"Recommendation stance '{stance.value}' is "
                            f"consistent with evidence."
                        ),
                        explanation=(
                            "Recommendation stance aligns with specialist findings."
                        ),
                    )
                )

        # Check 3.4: Empty reasons or risks for populated recommendation
        if (
            not report.key_reasons
            and stance != RecommendationStance.INSUFFICIENT_EVIDENCE
        ):
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_REC_KEY_REASONS_EMPTY",
                    status=ReportValidationStatus.WARNING,
                    severity=ReportValidationSeverity.MEDIUM,
                    category="recommendation_consistency",
                    affected_fields=["key_reasons"],
                    message=(
                        "Report has a recommendation stance but "
                        "key_reasons list is empty."
                    ),
                    explanation=(
                        "Key reasons supporting the stance should be articulated."
                    ),
                )
            )

    # -----------------------------------------------------------------------
    # 4. NUMERICAL PROVENANCE (12.3.4 - Specialist-Aware Provenance)
    # -----------------------------------------------------------------------

    def _validate_numerical_provenance(
        self,
        report: FinalReport,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ReportValidationIssue],
    ) -> None:
        """Verify numerical claims against attributed specialist and evidence sources.

        Remediated to enforce specialist-aware provenance:
        1. Validates specialist sections strictly against their corresponding domain.
        2. Validates investor capital cannot validate arbitrary analytical claims.
        3. Validates narrative statements using explicit citations and domain keywords.
        4. Distinguishes between:
           - valid numerical provenance (PASSED)
           - ambiguous/unattributed provenance (WARNING)
           - unsupported numerical claims (FAILED)
        """
        spec_numbers = extract_specialist_numbers_from_analysis(analysis)
        evidence_ref_map = extract_evidence_reference_numbers(analysis)

        investor_capital_numbers: Set[float] = set()
        if (
            analysis.investor_profile
            and analysis.investor_profile.capital_amount is not None
        ):
            investor_capital_numbers.add(
                float(analysis.investor_profile.capital_amount)
            )

        # Track results
        failures: List[ReportValidationIssue] = []
        warnings: List[ReportValidationIssue] = []
        verified_count = 0

        # -------------------------------------------------------------------
        # A. Direct Specialist Section Checks
        # -------------------------------------------------------------------
        specialist_sections: List[Tuple[str, Any, str]] = [
            ("technical", report.technical, "technical"),
            ("fundamental", report.fundamental, "fundamental"),
            ("news", report.news, "news"),
            ("research", report.research, "research"),
            ("risk", report.risk, "risk"),
        ]

        for sec_name, sec_obj, domain in specialist_sections:
            if sec_obj is None:
                continue

            # Collect numbers from this section
            sec_numbers: Set[float] = set()
            if hasattr(sec_obj, "model_dump"):
                for v in sec_obj.model_dump().values():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if float(v) not in EXCLUDED_NUMBERS:
                            sec_numbers.add(float(v))
                    elif isinstance(v, str):
                        sec_numbers.update(extract_numbers_from_text(v))
                    elif isinstance(v, (list, tuple, set)):
                        for item in v:
                            if isinstance(item, (int, float)) and not isinstance(
                                item, bool
                            ):
                                if float(item) not in EXCLUDED_NUMBERS:
                                    sec_numbers.add(float(item))
                            elif isinstance(item, str):
                                sec_numbers.update(extract_numbers_from_text(item))

            allowed_sec_numbers = spec_numbers.get(domain, set())

            for num in sec_numbers:
                if num in EXCLUDED_NUMBERS:
                    continue
                if _number_in_set(num, allowed_sec_numbers):
                    verified_count += 1
                    continue

                # Not in this specialist! Does it belong to another specialist?
                other_specs = [
                    s
                    for s, nums in spec_numbers.items()
                    if s != domain and _number_in_set(num, nums)
                ]
                if other_specs:
                    failures.append(
                        ReportValidationIssue(
                            check_id="CHK_NUM_PROVENANCE_MISMATCH",
                            status=ReportValidationStatus.FAILED,
                            severity=ReportValidationSeverity.HIGH,
                            category="numerical_provenance",
                            affected_fields=[sec_name],
                            message=(
                                f"Value {num} in {sec_name} section belongs to "
                                f"specialist(s) "
                                f"{other_specs} but was asserted in {sec_name}."
                            ),
                            explanation=(
                                f"Cross-specialist numerical mismatch: {num} "
                                f"originates in "
                                f"{other_specs}, not {sec_name}."
                            ),
                        )
                    )
                else:
                    failures.append(
                        ReportValidationIssue(
                            check_id="CHK_NUM_UNGROUNDED_VALUE",
                            status=ReportValidationStatus.FAILED,
                            severity=ReportValidationSeverity.CRITICAL,
                            category="numerical_provenance",
                            affected_fields=[sec_name],
                            message=(
                                f"Ungrounded numerical value {num} found in "
                                f"'{sec_name}'."
                            ),
                            explanation=(
                                f"Number {num} does not exist in source data for "
                                f"{sec_name}."
                            ),
                        )
                    )

        # -------------------------------------------------------------------
        # B. Narrative Statements & Profile Alignment Checks
        # -------------------------------------------------------------------
        narrative_entries: List[Tuple[str, str]] = [
            (
                "overall_assessment.synthesis",
                report.overall_assessment.synthesis,
            ),
        ]
        if report.recommendation:
            if report.recommendation.rationale:
                narrative_entries.append(
                    (
                        "recommendation.rationale",
                        report.recommendation.rationale,
                    )
                )
            if report.recommendation.profile_alignment:
                narrative_entries.append(
                    (
                        "recommendation.profile_alignment",
                        report.recommendation.profile_alignment,
                    )
                )
        for idx, reason in enumerate(report.key_reasons):
            narrative_entries.append((f"key_reasons[{idx}]", reason))
        for idx, risk in enumerate(report.important_risks):
            narrative_entries.append((f"important_risks[{idx}]", risk))

        for field_name, full_text in narrative_entries:
            # Split into sentence-like statements for precise attribution
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?;])\s+|\n+", full_text)
                if s.strip()
            ]
            for stmt_clean in sentences:
                if not stmt_clean:
                    continue

                stmt_lower = stmt_clean.lower()
                stmt_numbers = extract_numbers_from_text(stmt_clean)

                for num in stmt_numbers:
                    if num in EXCLUDED_NUMBERS:
                        continue

                    # 1. Investor Capital Isolation
                    is_investor_context = "profile_alignment" in field_name or any(
                        k in stmt_lower for k in INVESTOR_PROFILE_KEYWORDS
                    )
                    if _number_in_set(num, investor_capital_numbers):
                        if is_investor_context:
                            verified_count += 1
                            continue
                        else:
                            # Analytical statement using investor capital
                            failures.append(
                                ReportValidationIssue(
                                    check_id="CHK_NUM_INVESTOR_CAPITAL_MISUSE",
                                    status=ReportValidationStatus.FAILED,
                                    severity=ReportValidationSeverity.HIGH,
                                    category="numerical_provenance",
                                    affected_fields=[field_name],
                                    message=(
                                        f"Investor capital value {num} cannot "
                                        f"validate analytical claim: "
                                        f"'{stmt_clean[:60]}'."
                                    ),
                                    explanation=(
                                        "Investor capital amount is a portfolio "
                                        "constraint and cannot be used as evidence "
                                        "for analytical metrics."
                                    ),
                                )
                            )
                            continue

                    # 2. Check for explicit citation [ref: <id>]
                    citation_matches = CITATION_REF_PATTERN.findall(stmt_clean)
                    if citation_matches:
                        ref_ids = [
                            (m[0] or m[1]).strip().lower() for m in citation_matches
                        ]
                        valid_ref = next(
                            (r for r in ref_ids if r in evidence_ref_map), None
                        )
                        if valid_ref:
                            cited_spec, cited_nums = evidence_ref_map[valid_ref]
                            if _number_in_set(num, cited_nums) or _number_in_set(
                                num, spec_numbers.get(cited_spec, set())
                            ):
                                verified_count += 1
                                continue
                            else:
                                # Number not in cited evidence
                                other_specs = [
                                    s
                                    for s, nums in spec_numbers.items()
                                    if s != cited_spec and _number_in_set(num, nums)
                                ]
                                failures.append(
                                    ReportValidationIssue(
                                        check_id="CHK_NUM_PROVENANCE_MISMATCH",
                                        status=ReportValidationStatus.FAILED,
                                        severity=ReportValidationSeverity.HIGH,
                                        category="numerical_provenance",
                                        affected_fields=[field_name],
                                        message=(
                                            f"Value {num} cited with "
                                            f"[ref: {valid_ref}] does not "
                                            f"exist in cited specialist "
                                            f"'{cited_spec}' evidence."
                                        ),
                                        explanation=(
                                            f"Value {num} was attributed to "
                                            f"{cited_spec} via [ref: {valid_ref}], "
                                            f"but is found only in {other_specs}."
                                        ),
                                    )
                                )
                                continue

                    # 3. Check domain keywords in the sentence
                    has_fund = any(k in stmt_lower for k in FUND_KEYWORDS)
                    has_tech = any(k in stmt_lower for k in TECH_KEYWORDS)
                    has_news = any(k in stmt_lower for k in NEWS_KEYWORDS)
                    has_research = any(k in stmt_lower for k in RESEARCH_KEYWORDS)
                    has_risk = any(k in stmt_lower for k in RISK_KEYWORDS)

                    if has_fund and not has_tech:
                        if _number_in_set(num, spec_numbers["fundamental"]):
                            verified_count += 1
                            continue
                        other_specs = [
                            s
                            for s, nums in spec_numbers.items()
                            if s != "fundamental" and _number_in_set(num, nums)
                        ]
                        if other_specs:
                            failures.append(
                                ReportValidationIssue(
                                    check_id="CHK_NUM_PROVENANCE_MISMATCH",
                                    status=ReportValidationStatus.FAILED,
                                    severity=ReportValidationSeverity.HIGH,
                                    category="numerical_provenance",
                                    affected_fields=[field_name],
                                    message=(
                                        f"Value {num} in fundamental claim belongs to "
                                        f"{other_specs} "
                                        "but is asserted as a fundamental metric."
                                    ),
                                    explanation=(
                                        f"Provenance mismatch: {num} is found in "
                                        f"{other_specs}, "
                                        "not in fundamental data."
                                    ),
                                )
                            )
                        else:
                            failures.append(
                                ReportValidationIssue(
                                    check_id="CHK_NUM_UNGROUNDED_VALUE",
                                    status=ReportValidationStatus.FAILED,
                                    severity=ReportValidationSeverity.CRITICAL,
                                    category="numerical_provenance",
                                    affected_fields=[field_name],
                                    message=(
                                        f"Ungrounded numerical value {num} found in "
                                        f"'{field_name}'."
                                    ),
                                    explanation=(
                                        f"Value {num} does not appear in any "
                                        f"specialist data."
                                    ),
                                )
                            )
                        continue

                    if has_tech and not has_fund:
                        if _number_in_set(num, spec_numbers["technical"]):
                            verified_count += 1
                            continue
                        other_specs = [
                            s
                            for s, nums in spec_numbers.items()
                            if s != "technical" and _number_in_set(num, nums)
                        ]
                        if other_specs:
                            failures.append(
                                ReportValidationIssue(
                                    check_id="CHK_NUM_PROVENANCE_MISMATCH",
                                    status=ReportValidationStatus.FAILED,
                                    severity=ReportValidationSeverity.HIGH,
                                    category="numerical_provenance",
                                    affected_fields=[field_name],
                                    message=(
                                        f"Value {num} in technical claim belongs to "
                                        f"{other_specs} "
                                        "but is asserted as a technical indicator."
                                    ),
                                    explanation=(
                                        f"Provenance mismatch: {num} is found in "
                                        f"{other_specs}, "
                                        "not in technical data."
                                    ),
                                )
                            )
                        else:
                            failures.append(
                                ReportValidationIssue(
                                    check_id="CHK_NUM_UNGROUNDED_VALUE",
                                    status=ReportValidationStatus.FAILED,
                                    severity=ReportValidationSeverity.CRITICAL,
                                    category="numerical_provenance",
                                    affected_fields=[field_name],
                                    message=(
                                        f"Ungrounded numerical value {num} found in "
                                        f"'{field_name}'."
                                    ),
                                    explanation=(
                                        f"Value {num} does not appear in any "
                                        f"specialist data."
                                    ),
                                )
                            )
                        continue

                    if has_research:
                        if _number_in_set(num, spec_numbers["research"]):
                            verified_count += 1
                            continue

                    if has_news:
                        if _number_in_set(num, spec_numbers["news"]):
                            verified_count += 1
                            continue

                    if has_risk:
                        if _number_in_set(num, spec_numbers["risk"]):
                            verified_count += 1
                            continue

                    # 4. Unattributed sentence (no explicit citation and no
                    #    domain keywords)
                    candidates = [
                        s
                        for s, nums in spec_numbers.items()
                        if _number_in_set(num, nums)
                    ]
                    if len(candidates) == 0:
                        failures.append(
                            ReportValidationIssue(
                                check_id="CHK_NUM_UNGROUNDED_VALUE",
                                status=ReportValidationStatus.FAILED,
                                severity=ReportValidationSeverity.CRITICAL,
                                category="numerical_provenance",
                                affected_fields=[field_name],
                                message=(
                                    f"Ungrounded numerical value {num} found in "
                                    f"'{field_name}'."
                                ),
                                explanation=(
                                    f"Number {num} does not exist in any "
                                    f"specialist assessment or profile."
                                ),
                            )
                        )
                    elif len(candidates) > 1:
                        # Ambiguous provenance across multiple specialists
                        warnings.append(
                            ReportValidationIssue(
                                check_id="CHK_NUM_AMBIGUOUS_PROVENANCE",
                                status=ReportValidationStatus.WARNING,
                                severity=ReportValidationSeverity.MEDIUM,
                                category="numerical_provenance",
                                affected_fields=[field_name],
                                message=(
                                    f"Value {num} in '{field_name}' appears across "
                                    f"multiple specialists "
                                    f"{candidates} without explicit attribution."
                                ),
                                explanation=(
                                    "Ambiguous provenance: value exists in multiple "
                                    "specialists "
                                    "without explicit citation or domain attribution."
                                ),
                            )
                        )
                    else:
                        # Exactly one candidate specialist, but sentence is
                        # completely unattributed
                        warnings.append(
                            ReportValidationIssue(
                                check_id="CHK_NUM_UNATTRIBUTED_PROVENANCE",
                                status=ReportValidationStatus.WARNING,
                                severity=ReportValidationSeverity.LOW,
                                category="numerical_provenance",
                                affected_fields=[field_name],
                                message=(
                                    f"Value {num} in '{field_name}' originates in "
                                    f"specialist '{candidates[0]}' but lacks "
                                    f"explicit source attribution."
                                ),
                                explanation=(
                                    f"Value {num} verified in '{candidates[0]}', "
                                    f"but is unattributed in text."
                                ),
                            )
                        )

        # Record issues
        if failures:
            issues.extend(failures)
        if warnings:
            issues.extend(warnings)

        if not failures and (verified_count > 0 or not warnings):
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_NUM_PROVENANCE_VALID",
                    status=ReportValidationStatus.PASSED,
                    severity=ReportValidationSeverity.LOW,
                    category="numerical_provenance",
                    affected_fields=[
                        "recommendation",
                        "key_reasons",
                        "important_risks",
                    ],
                    message=(
                        "All numerical values in report narrative have valid "
                        "provenance."
                    ),
                    explanation=(
                        f"Verified {verified_count} quantitative claims against "
                        f"specialist data."
                    ),
                )
            )

    # -----------------------------------------------------------------------
    # 5. SOURCE / EVIDENCE CONSISTENCY (12.3.5)
    # -----------------------------------------------------------------------

    def _validate_evidence_consistency(
        self,
        report: FinalReport,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ReportValidationIssue],
    ) -> None:
        """Verify citations and evidence items match specialist attribution."""
        known_evidence_keys: Set[Tuple[str, str]] = {
            (_val(e.specialist).lower(), e.reference_id.strip().lower())
            for e in analysis.aggregated_evidence
        }
        known_ref_ids: Set[str] = {
            e.reference_id.lower() for e in analysis.aggregated_evidence
        }
        for e in analysis.aggregated_evidence:
            if e.chunk_id:
                known_ref_ids.add(e.chunk_id.lower())
            if e.document_id:
                known_ref_ids.add(e.document_id.lower())

        # Check 5.1: Citations in narrative text
        texts_to_check: List[Tuple[str, str]] = [
            (
                "overall_assessment.synthesis",
                report.overall_assessment.synthesis,
            ),
        ]
        if report.recommendation and report.recommendation.rationale:
            texts_to_check.append(
                ("recommendation.rationale", report.recommendation.rationale)
            )
        for idx, r in enumerate(report.key_reasons):
            texts_to_check.append((f"key_reasons[{idx}]", r))
        for idx, r in enumerate(report.important_risks):
            texts_to_check.append((f"important_risks[{idx}]", r))

        phantom_citations: List[Tuple[str, str]] = []
        for field_name, text in texts_to_check:
            for match_tuple in CITATION_REF_PATTERN.findall(text):
                cited_id = (match_tuple[0] or match_tuple[1]).strip().lower()
                if (
                    cited_id
                    and not cited_id.isdigit()
                    and not cited_id.startswith("chk_")
                    and cited_id not in known_ref_ids
                ):
                    phantom_citations.append((field_name, cited_id))

        if phantom_citations:
            for field_name, cited_id in phantom_citations:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_EVID_PHANTOM_CITATION",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.CRITICAL,
                        category="evidence_consistency",
                        affected_fields=[field_name],
                        message=(
                            f"Phantom citation '[ref: {cited_id}]' in "
                            f"'{field_name}'."
                        ),
                        explanation=(
                            f"Cited reference ID '{cited_id}' does not exist in the "
                            "aggregated evidence pool."
                        ),
                    )
                )
        else:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_EVID_PHANTOM_CITATION",
                    status=ReportValidationStatus.PASSED,
                    severity=ReportValidationSeverity.LOW,
                    category="evidence_consistency",
                    affected_fields=["narrative_citations"],
                    message=(
                        "All cited reference IDs exist in the aggregated "
                        "evidence pool."
                    ),
                    explanation="No phantom citations found in narrative.",
                )
            )

        # Check 5.2: Evidence sources item attribution
        unmatched_evidence: List[AggregatedEvidenceItem] = []
        for ev in report.evidence_sources:
            s_str = _val(ev.specialist).lower()
            ref_id = ev.reference_id.strip().lower()
            if (
                s_str,
                ref_id,
            ) not in known_evidence_keys and len(known_evidence_keys) > 0:
                unmatched_evidence.append(ev)

        if unmatched_evidence:
            for ev in unmatched_evidence:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_EVID_SOURCE_ATTRIBUTION_MISMATCH",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="evidence_consistency",
                        affected_fields=["evidence_sources"],
                        message=(
                            f"Evidence item ({ev.specialist}, '{ev.reference_id}') in "
                            "report not found in upstream specialist evidence."
                        ),
                        explanation=(
                            "Report evidence_sources must preserve genuine provenance."
                        ),
                    )
                )
        else:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_EVID_SOURCE_ATTRIBUTION_MISMATCH",
                    status=ReportValidationStatus.PASSED,
                    severity=ReportValidationSeverity.LOW,
                    category="evidence_consistency",
                    affected_fields=["evidence_sources"],
                    message=(
                        "All report evidence_sources correspond "
                        "to verified upstream items."
                    ),
                    explanation="Provenance correctly preserved.",
                )
            )

    # -----------------------------------------------------------------------
    # 6. CROSS-FIELD CONSISTENCY (12.3.6)
    # -----------------------------------------------------------------------

    def _validate_cross_field_consistency(
        self,
        report: FinalReport,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ReportValidationIssue],
    ) -> None:
        """Check for direct internal contradictions across report sections."""
        # Check 6.1: Insufficient evidence claim vs populated evidence
        if report.insufficient_evidence and len(report.evidence_sources) == 0:
            pass
        elif report.insufficient_evidence and len(report.evidence_sources) > 0:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_CROSS_INSUFFICIENT_WITH_EVIDENCE",
                    status=ReportValidationStatus.WARNING,
                    severity=ReportValidationSeverity.LOW,
                    category="cross_field",
                    affected_fields=[
                        "insufficient_evidence",
                        "evidence_sources",
                    ],
                    message=(
                        f"Report marks insufficient_evidence=True while "
                        f"{len(report.evidence_sources)} evidence items are retained."
                    ),
                    explanation="Partial evidence retained for transparency.",
                )
            )

        # Check 6.2: Missing specialist claim in report vs section present
        for spec, status in report.specialist_statuses.items():
            rep_sec = getattr(report, spec, None)
            if status == SpecialistStatus.MISSING and rep_sec is not None:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_CROSS_STATUS_SECTION_CONTRADICTION",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.CRITICAL,
                        category="cross_field",
                        affected_fields=[f"specialist_statuses.{spec}", spec],
                        message=(
                            f"Internal contradiction: specialist_statuses['{spec}'] is "
                            f"'missing', but report.{spec} is populated."
                        ),
                        explanation=(
                            "Section data cannot be present for a missing specialist."
                        ),
                    )
                )

        # Check 6.3: Recommendation stance vs Overall Assessment synthesis
        if report.recommendation:
            stance = report.recommendation.stance
            synthesis_lower = report.overall_assessment.synthesis.lower()

            if (
                stance == RecommendationStance.FAVORABLE
                and "severely deteriorated" in synthesis_lower
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_CROSS_STANCE_SYNTHESIS_CONTRADICTION",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="cross_field",
                        affected_fields=[
                            "recommendation.stance",
                            "overall_assessment.synthesis",
                        ],
                        message=(
                            "Contradiction between recommendation stance 'favorable' "
                            "and overall synthesis describing 'severely deteriorated' "
                            "conditions."
                        ),
                        explanation=(
                            "Recommendation stance must align with overall synthesis."
                        ),
                    )
                )
            elif (
                stance == RecommendationStance.UNFAVORABLE
                and "strong bullish alignment" in synthesis_lower
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_CROSS_STANCE_SYNTHESIS_CONTRADICTION",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="cross_field",
                        affected_fields=[
                            "recommendation.stance",
                            "overall_assessment.synthesis",
                        ],
                        message=(
                            "Contradiction between recommendation stance 'unfavorable' "
                            "and overall synthesis describing 'strong bullish "
                            "alignment'."
                        ),
                        explanation=(
                            "Recommendation stance must align with overall synthesis."
                        ),
                    )
                )

    # -----------------------------------------------------------------------
    # 7. COMPLETENESS VALIDATION (12.3.7)
    # -----------------------------------------------------------------------

    def _validate_completeness(
        self,
        report: FinalReport,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ReportValidationIssue],
    ) -> None:
        """Validate report completeness representation and partial report status."""
        is_complete = report.is_complete

        # Check 7.1: Insufficient evidence report falsely claiming completeness
        if report.insufficient_evidence and is_complete:
            if (
                report.recommendation
                and report.recommendation.stance
                != RecommendationStance.INSUFFICIENT_EVIDENCE
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_COMPLETENESS_INSUFFICIENT_FALSE_COMPLETE",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="completeness",
                        affected_fields=["insufficient_evidence", "is_complete"],
                        message=(
                            "Report claims completeness despite insufficient evidence."
                        ),
                        explanation=(
                            "Insufficient evidence reports cannot issue "
                            "complete recommendations."
                        ),
                    )
                )

        # Check 7.2: Partial specialist availability transparency
        missing_count = max(
            len(report.missing_specialists),
            sum(
                1
                for s, st in report.specialist_statuses.items()
                if st == SpecialistStatus.MISSING
            ),
            sum(
                1
                for s, st in analysis.specialist_statuses.items()
                if st == SpecialistStatus.MISSING
            ),
        )
        failed_count = max(
            len(report.failed_specialists),
            sum(
                1
                for s, st in report.specialist_statuses.items()
                if st == SpecialistStatus.FAILED
            ),
            sum(
                1
                for s, st in analysis.specialist_statuses.items()
                if st == SpecialistStatus.FAILED
            ),
        )
        if (missing_count > 0 or failed_count > 0) and is_complete:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_COMPLETENESS_PARTIAL_REPORT_NOTICE",
                    status=ReportValidationStatus.WARNING,
                    severity=ReportValidationSeverity.LOW,
                    category="completeness",
                    affected_fields=[
                        "missing_specialists",
                        "failed_specialists",
                    ],
                    message=(
                        f"Report is marked complete but operates with partial coverage "
                        f"({missing_count} missing, {failed_count} failed specialists)."
                    ),
                    explanation="Partial report transparency confirmed.",
                )
            )
        else:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_COMPLETENESS_STATUS_VALID",
                    status=ReportValidationStatus.PASSED,
                    severity=ReportValidationSeverity.LOW,
                    category="completeness",
                    affected_fields=["is_complete"],
                    message=(
                        f"Report completeness status "
                        f"(is_complete={is_complete}) is consistent."
                    ),
                    explanation="Completeness state matches field availability.",
                )
            )

    # -----------------------------------------------------------------------
    # 8. REGULATORY SAFETY VALIDATION (12.3.8)
    # -----------------------------------------------------------------------

    def _validate_safety(
        self,
        report: FinalReport,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ReportValidationIssue],
    ) -> None:
        """Verify compliance, disclaimer presence, and absence of advice."""
        # Check 8.1: Disclaimer presence & risk disclosure
        if not report.disclaimer or not report.disclaimer.strip():
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_SAFETY_DISCLAIMER_MISSING",
                    status=ReportValidationStatus.FAILED,
                    severity=ReportValidationSeverity.CRITICAL,
                    category="safety",
                    affected_fields=["disclaimer"],
                    message="Mandatory regulatory disclaimer is missing or blank.",
                    explanation=(
                        "All reports must include a standard regulatory disclaimer."
                    ),
                )
            )
        else:
            disc_lower = report.disclaimer.lower()
            if not any(
                k in disc_lower
                for k in ("informational", "not a registered", "risk of loss")
            ):
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_SAFETY_DISCLAIMER_INVALID",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.HIGH,
                        category="safety",
                        affected_fields=["disclaimer"],
                        message=(
                            "Disclaimer lacks mandatory non-advisory "
                            "or risk disclosures."
                        ),
                        explanation=(
                            "Disclaimer must state research/informational nature."
                        ),
                    )
                )
            else:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_SAFETY_DISCLAIMER_VALID",
                        status=ReportValidationStatus.PASSED,
                        severity=ReportValidationSeverity.LOW,
                        category="safety",
                        affected_fields=["disclaimer"],
                        message="Mandatory regulatory disclaimer is present and valid.",
                        explanation=(
                            "Meets standard non-advisory regulatory requirements."
                        ),
                    )
                )

        # Check 8.2: Prohibited advice patterns across text fields
        texts_to_scan: List[Tuple[str, str]] = [
            (
                "overall_assessment.synthesis",
                report.overall_assessment.synthesis,
            ),
        ]
        if report.recommendation:
            if report.recommendation.rationale:
                texts_to_scan.append(
                    (
                        "recommendation.rationale",
                        report.recommendation.rationale,
                    )
                )
            if report.recommendation.profile_alignment:
                texts_to_scan.append(
                    (
                        "recommendation.profile_alignment",
                        report.recommendation.profile_alignment,
                    )
                )
        for idx, r in enumerate(report.key_reasons):
            texts_to_scan.append((f"key_reasons[{idx}]", r))
        for idx, r in enumerate(report.important_risks):
            texts_to_scan.append((f"important_risks[{idx}]", r))

        detected_prohibited: List[Tuple[str, str, str]] = []
        for field_name, text in texts_to_scan:
            if not text:
                continue
            for pattern in PROHIBITED_ADVICE_PATTERNS:
                if pattern.search(text):
                    detected_prohibited.append((field_name, pattern.pattern, text))

        if detected_prohibited:
            for field_name, pat, _ in detected_prohibited:
                issues.append(
                    ReportValidationIssue(
                        check_id="CHK_SAFETY_PROHIBITED_ADVICE",
                        status=ReportValidationStatus.FAILED,
                        severity=ReportValidationSeverity.CRITICAL,
                        category="safety",
                        affected_fields=[field_name],
                        message=(
                            f"Prohibited non-advisory phrase matching '{pat}' "
                            f"in '{field_name}'."
                        ),
                        explanation=(
                            "FinPilot strictly prohibits buy/sell directives, price "
                            "targets, and guaranteed return claims."
                        ),
                    )
                )
        else:
            issues.append(
                ReportValidationIssue(
                    check_id="CHK_SAFETY_PROHIBITED_ADVICE",
                    status=ReportValidationStatus.PASSED,
                    severity=ReportValidationSeverity.LOW,
                    category="safety",
                    affected_fields=["all_text_fields"],
                    message=(
                        "No prohibited advice, price targets, or "
                        "guarantee claims detected."
                    ),
                    explanation="Strict regulatory non-advisory boundary maintained.",
                )
            )


# ===========================================================================
# 3. PUBLIC VALIDATION API (Phase 12.3.10)
# ===========================================================================


def validate_final_report(
    report: FinalReport,
    source_data: Union[UnifiedSpecialistAnalysis, ReportGeneratorInput],
) -> ReportValidationResult:
    """Validate a FinalReport against its upstream source analysis deterministically.

    Args:
        report: FinalReport instance produced by ReportGeneratorAgent.
        source_data: UnifiedSpecialistAnalysis or ReportGeneratorInput.

    Returns:
        ReportValidationResult: Detailed result containing validity boolean,
        counts, issues list, and executive summary.
    """
    validator = ReportConsistencyValidator()
    return validator.validate(report, source_data)
