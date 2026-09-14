"""Aggregation Consistency Checks for Report Aggregator (Phase 11.3).

Validates the aggregated specialist analysis produced by Phase 11.2 for:
- Evidence provenance and attribution preservation (11.3.1).
- Unanimous signal sanity and contradiction detection (11.3.2).
- Grounded agreement and signal conflict validity.
- Cross-specialist observation groundedness.
- Confidence and data completeness consistency.
- Prohibited advisory, target price, and guaranteed return language.
- Quantitative claim provenance (ensuring values originate in specialist data).
- Correct representation of missing and failed specialists without fabrication.
- Handling empty and malformed aggregation inputs safely.
- Prompt-injection defense: treating specialist content strictly as data.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.agents.aggregator_schema import (
    PROHIBITED_ADVICE_PATTERNS,
    AggregatedEvidenceItem,
    AggregationConsistencyReport,
    ConsistencyCheckSeverity,
    ConsistencyCheckStatus,
    ConsistencyIssue,
    ReportAggregatorInput,
    SpecialistStatus,
    UnifiedSpecialistAnalysis,
)
from app.core.logging import get_logger

logger = get_logger("app.agents.aggregator_consistency")

# Recognized specialist types in FinPilot
VALID_SPECIALIST_TYPES: Set[str] = {
    "technical",
    "fundamental",
    "news",
    "research",
    "risk",
}

# Regex to detect prompt injection attempts in specialist text
PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior)\s+instructions\b"),
    re.compile(r"(?i)\bdisregard\s+(all\s+)?(previous|prior)\b"),
    re.compile(r"(?i)\bsystem\s*:\s*(override|mark\s+all|set\s+is_valid|pass)\b"),
    re.compile(r"(?i)\bmark\s+(all\s+)?checks?\s+(as\s+)?passed\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+an?\s+unfiltered\b"),
    re.compile(r"(?i)\boverride\s+consistency\s+rules\b"),
]

# Regex to detect numbers, currency values, percentages, multiples, and scales
NUMBER_EXTRACTION_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9_])[\$]?[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:%|[xXbBmMkK])?(?![a-zA-Z0-9_])"
)

# Common years and trivial counting integers to exclude from quantitative checks
EXCLUDED_NUMBERS: Set[float] = {
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    2020.0,
    2021.0,
    2022.0,
    2023.0,
    2024.0,
    2025.0,
    2026.0,
    2027.0,
    2028.0,
    2029.0,
    2030.0,
}


def _val(x: Any) -> str:
    """Safely extract string representation from enum or string."""
    if x is None:
        return ""
    if hasattr(x, "value"):
        return str(x.value)
    return str(x)


def extract_numbers_from_text(text: str) -> Set[float]:
    """Extract numeric values from a given text string, filtering out noise."""
    if not text:
        return set()
    matches = NUMBER_EXTRACTION_PATTERN.findall(text)
    numbers: Set[float] = set()
    for m in matches:
        cleaned = (
            m.replace("$", "")
            .replace("%", "")
            .replace(",", "")
            .rstrip("xXbBmMkK")
            .strip()
        )
        try:
            val = float(cleaned)
            if val not in EXCLUDED_NUMBERS:
                numbers.add(val)
        except ValueError:
            continue
    return numbers


def _number_in_set(num: float, number_set: Set[float]) -> bool:
    """Check if a float value exists in a set within tolerance."""
    return any(math.isclose(num, x, rel_tol=1e-2, abs_tol=1e-2) for x in number_set)


def extract_specialist_numbers(
    input_data: ReportAggregatorInput,
) -> Dict[str, Set[float]]:
    """Extract numeric metrics and values partitioned by source specialist."""
    result: Dict[str, Set[float]] = {
        "technical": set(),
        "fundamental": set(),
        "news": set(),
        "research": set(),
        "risk": set(),
    }

    def _collect_from(obj: Any, target_set: Set[float]) -> None:
        if obj is None:
            return
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            val = float(obj)
            if val not in EXCLUDED_NUMBERS:
                target_set.add(val)
        elif isinstance(obj, str):
            target_set.update(extract_numbers_from_text(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect_from(v, target_set)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _collect_from(item, target_set)
        elif hasattr(obj, "model_dump"):
            _collect_from(obj.model_dump(), target_set)
        elif hasattr(obj, "__dict__"):
            _collect_from(obj.__dict__, target_set)

    if input_data.has_technical and input_data.technical:
        _collect_from(input_data.technical, result["technical"])
    if input_data.has_fundamental and input_data.fundamental:
        _collect_from(input_data.fundamental, result["fundamental"])
    if input_data.has_news and input_data.news:
        _collect_from(input_data.news, result["news"])
    if input_data.has_research and input_data.research:
        _collect_from(input_data.research, result["research"])
    if input_data.has_risk and input_data.risk:
        _collect_from(input_data.risk, result["risk"])

    # Also include attributed evidence items by specialist
    for ev in input_data.extract_attributed_evidence():
        s_name = _val(ev.specialist).lower()
        if s_name in result:
            result[s_name].update(extract_numbers_from_text(ev.detail))

    return result


def extract_source_numbers(input_data: ReportAggregatorInput) -> Set[float]:
    """Extract all numeric metrics and values across all specialist data."""
    spec_dict = extract_specialist_numbers(input_data)
    combined: Set[float] = set()
    for s_nums in spec_dict.values():
        combined.update(s_nums)
    return combined


def get_specialist_directional_stance(
    specialist_name: str,
    input_data: ReportAggregatorInput,
) -> Optional[str]:
    """Return deterministic directional stance.

    Allowed values: 'positive', 'negative', 'neutral', or None.
    """
    s_clean = specialist_name.strip().lower()
    if s_clean == "technical":
        if input_data.has_technical and input_data.technical:
            trend = _val(input_data.technical.trend).lower()
            if trend in ("uptrend", "bullish"):
                return "positive"
            if trend in ("downtrend", "bearish"):
                return "negative"
            if trend in ("sideways", "neutral"):
                return "neutral"
        return None
    if s_clean == "fundamental":
        if input_data.has_fundamental and input_data.fundamental:
            assess = _val(input_data.fundamental.overall_assessment).lower()
            if assess in ("favorable", "strong", "positive"):
                return "positive"
            if assess in ("unfavorable", "weak", "negative"):
                return "negative"
            if assess in ("neutral",):
                return "neutral"
        return None
    if s_clean == "news":
        if input_data.has_news and input_data.news:
            sent = _val(input_data.news.overall_sentiment).lower()
            if sent in ("positive",):
                return "positive"
            if sent in ("negative",):
                return "negative"
            if sent in ("neutral",):
                return "neutral"
        return None
    if s_clean == "risk":
        if input_data.has_risk and input_data.risk:
            lvl = _val(input_data.risk.overall_risk_level).lower()
            if lvl in ("low", "minimal"):
                return "positive"
            if lvl in ("high", "critical"):
                return "negative"
            if lvl in ("moderate",):
                return "neutral"
        return None
    return None


class AggregationConsistencyChecker:
    """Deterministic validation engine for Report Aggregator outputs (Phase 11.3)."""

    def validate(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
    ) -> AggregationConsistencyReport:
        """Run all Phase 11.3 consistency checks against the aggregated analysis."""
        issues: List[ConsistencyIssue] = []

        # 1. Payload & Ticker
        self._check_malformed_payload(input_data, analysis, issues)

        # 2. Empty Input & Insufficient Evidence
        self._check_empty_and_insufficient_evidence(input_data, analysis, issues)

        # 3. Specialist Attribution
        self._check_specialist_attribution(input_data, analysis, issues)

        # 4. Evidence Provenance
        self._check_evidence_provenance(input_data, analysis, issues)

        # 5. Agreement Validity
        self._check_agreement_validity(input_data, analysis, issues)

        # 6. Signal Conflicts
        self._check_conflict_validity(input_data, analysis, issues)

        # 7. Cross-Specialist Observations
        self._check_cross_observations(input_data, analysis, issues)

        # 8. Unanimous Signal Sanity
        self._check_unanimous_signal_sanity(input_data, analysis, issues)

        # 9. Confidence vs Completeness
        self._check_confidence_completeness(input_data, analysis, issues)

        # 10. Missing & Failed Specialists
        self._check_missing_failed_specialists(input_data, analysis, issues)

        # 11. Safety & Prohibited Language
        self._check_safety_prohibited_language(input_data, analysis, issues)

        # 12. Quantitative Provenance
        self._check_quantitative_provenance(input_data, analysis, issues)

        # 13. Prompt Injection Defense
        self._check_prompt_injection(input_data, analysis, issues)

        # 14. Duplicate Evidence
        self._check_duplicate_evidence(input_data, analysis, issues)

        passed_count = sum(
            1 for i in issues if i.status == ConsistencyCheckStatus.PASSED
        )
        warning_count = sum(
            1 for i in issues if i.status == ConsistencyCheckStatus.WARNING
        )
        failure_count = sum(
            1 for i in issues if i.status == ConsistencyCheckStatus.FAILED
        )

        is_valid = not any(
            i.status == ConsistencyCheckStatus.FAILED
            and i.severity
            in (ConsistencyCheckSeverity.HIGH, ConsistencyCheckSeverity.CRITICAL)
            for i in issues
        )

        is_empty_or_zero = (
            input_data.is_empty or len(input_data.available_specialists) == 0
        )
        if is_empty_or_zero:
            is_valid = False

        if is_empty_or_zero:
            summary = (
                f"Validation for {input_data.ticker} has insufficient evidence "
                "(0 specialists). Aggregation cannot be certified consistent."
            )
        elif is_valid:
            summary = (
                f"Validation for {input_data.ticker} PASSED with {passed_count} "
                f"passes, {warning_count} warnings, {failure_count} minor fails."
            )
        else:
            summary = (
                f"Validation for {input_data.ticker} FAILED with {failure_count} "
                f"failures, {warning_count} warnings, and {passed_count} passes."
            )

        return AggregationConsistencyReport(
            ticker=input_data.ticker,
            is_valid=is_valid,
            passed_checks_count=passed_count,
            warnings_count=warning_count,
            failures_count=failure_count,
            issues=issues,
            summary=summary,
            insufficient_evidence=(is_empty_or_zero or analysis.insufficient_evidence),
        )

    def _check_malformed_payload(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Validate structural integrity and ticker matching."""
        check_id = "CHK_MALFORMED_PAYLOAD"
        if input_data.ticker != analysis.ticker:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.FAILED,
                    severity=ConsistencyCheckSeverity.HIGH,
                    message=(
                        f"Ticker mismatch: input '{input_data.ticker}', "
                        f"output '{analysis.ticker}'."
                    ),
                    explanation=(
                        f"Input ticker ({input_data.ticker}) does not match "
                        f"output ticker ({analysis.ticker})."
                    ),
                )
            )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="Payload structure and ticker validated.",
                    explanation=(
                        f"Ticker '{analysis.ticker}' matches across input and output."
                    ),
                )
            )

    def _check_empty_and_insufficient_evidence(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Verify zero-specialist inputs yield insufficient evidence."""
        check_id = "CHK_EMPTY_EVIDENCE_INPUT"
        is_empty = input_data.is_empty or len(input_data.available_specialists) == 0

        if is_empty:
            if not analysis.insufficient_evidence:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.CRITICAL,
                        message="0 specialists, but insufficient_evidence is False.",
                        explanation=(
                            "Expected insufficient_evidence=True for zero specialists, "
                            f"found {analysis.insufficient_evidence}."
                        ),
                    )
                )
            elif analysis.confidence > 0.05:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.CRITICAL,
                        message=(
                            f"0 specialists, but confidence "
                            f"{analysis.confidence} > 0.05."
                        ),
                        explanation=(
                            f"Confidence {analysis.confidence} invalid on empty "
                            "input; must be <= 0.05."
                        ),
                    )
                )
            else:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.WARNING,
                        severity=ConsistencyCheckSeverity.MEDIUM,
                        message="0 specialists. Insufficient evidence confirmed.",
                        explanation=(
                            "Analysis correctly records insufficient_evidence=True "
                            "and confidence=0.0."
                        ),
                    )
                )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message=(
                        f"Specialist data present "
                        f"({len(input_data.available_specialists)} available)."
                    ),
                    explanation=(
                        f"Found {len(input_data.available_specialists)} available "
                        "specialists in input."
                    ),
                )
            )

    def _check_specialist_attribution(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Verify all cited specialists are valid and available."""
        check_id = "CHK_ATTR_SPECIALIST_EXISTS"
        invalid_citations: List[Tuple[str, str, str]] = []

        for finding in analysis.areas_of_agreement:
            for s in finding.supporting_specialists:
                s_str = _val(s).lower()
                if s_str not in VALID_SPECIALIST_TYPES:
                    invalid_citations.append(
                        (s_str, "unknown", f"agreement '{finding.topic}'")
                    )
                elif (
                    input_data.specialist_statuses.get(s_str)
                    != SpecialistStatus.AVAILABLE
                ):
                    status_val = _val(input_data.specialist_statuses.get(s_str))
                    invalid_citations.append(
                        (s_str, status_val, f"agreement '{finding.topic}'")
                    )

        for conflict in analysis.signal_conflicts:
            for s in conflict.involved_specialists:
                s_str = _val(s).lower()
                if s_str not in VALID_SPECIALIST_TYPES:
                    invalid_citations.append(
                        (s_str, "unknown", f"conflict '{conflict.topic}'")
                    )
                elif (
                    input_data.specialist_statuses.get(s_str)
                    != SpecialistStatus.AVAILABLE
                ):
                    status_val = _val(input_data.specialist_statuses.get(s_str))
                    invalid_citations.append(
                        (s_str, status_val, f"conflict '{conflict.topic}'")
                    )

        for obs in analysis.cross_specialist_observations:
            for s in obs.connected_specialists:
                s_str = _val(s).lower()
                if s_str not in VALID_SPECIALIST_TYPES:
                    invalid_citations.append((s_str, "unknown", "observation"))
                elif (
                    input_data.specialist_statuses.get(s_str)
                    != SpecialistStatus.AVAILABLE
                ):
                    status_val = _val(input_data.specialist_statuses.get(s_str))
                    invalid_citations.append((s_str, status_val, "observation"))

        for ev in analysis.aggregated_evidence:
            s_str = _val(ev.specialist).lower()
            if s_str not in VALID_SPECIALIST_TYPES:
                invalid_citations.append(
                    (s_str, "unknown", f"evidence ref '{ev.reference_id}'")
                )
            elif (
                input_data.specialist_statuses.get(s_str) != SpecialistStatus.AVAILABLE
            ):
                status_val = _val(input_data.specialist_statuses.get(s_str))
                invalid_citations.append(
                    (s_str, status_val, f"evidence ref '{ev.reference_id}'")
                )

        if invalid_citations:
            for s_str, status_str, ctx in invalid_citations:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.HIGH,
                        affected_specialists=[s_str],
                        message=(
                            f"Specialist '{s_str}' in {ctx} is unavailable "
                            f"(status: {status_str})."
                        ),
                        explanation=(
                            f"Cited specialist '{s_str}' has status '{status_str}' "
                            f"in input, not AVAILABLE."
                        ),
                    )
                )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="All cited specialists are valid and available.",
                    explanation=(
                        "All specialists referenced in findings and evidence "
                        "are recognized and AVAILABLE."
                    ),
                )
            )

    def _check_evidence_provenance(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Validate that all evidence items have IDs and match source data."""
        check_id = "CHK_EVID_PROVENANCE"
        source_evidence = input_data.extract_attributed_evidence()
        source_keys: Set[Tuple[str, str]] = {
            (_val(e.specialist).lower(), e.reference_id.strip())
            for e in source_evidence
        }

        unmatched: List[AggregatedEvidenceItem] = []
        malformed: List[AggregatedEvidenceItem] = []

        all_items: List[AggregatedEvidenceItem] = list(analysis.aggregated_evidence)
        for f in analysis.areas_of_agreement:
            all_items.extend(f.evidence)
        for c in analysis.signal_conflicts:
            all_items.extend(c.evidence)
        for o in analysis.cross_specialist_observations:
            all_items.extend(o.evidence)

        for ev in all_items:
            s_str = _val(ev.specialist).lower()
            ref_id = ev.reference_id.strip() if ev.reference_id else ""
            detail = ev.detail.strip() if ev.detail else ""

            if not ref_id or not detail:
                malformed.append(ev)
                continue

            if not input_data.is_empty and (s_str, ref_id) not in source_keys:
                unmatched.append(ev)

        if malformed:
            for ev in malformed:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.HIGH,
                        affected_specialists=[_val(ev.specialist)],
                        affected_evidence_refs=[ev.reference_id or "UNKNOWN"],
                        message="Evidence item missing reference_id or detail.",
                        explanation=(
                            f"Specialist '{ev.specialist}' evidence lacks "
                            f"ref or detail: ref='{ev.reference_id}'."
                        ),
                    )
                )

        if unmatched:
            for ev in unmatched:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.HIGH,
                        affected_specialists=[_val(ev.specialist)],
                        affected_evidence_refs=[ev.reference_id],
                        message=(
                            f"Ref '{ev.reference_id}' for '{ev.specialist}' "
                            "not found in source evidence."
                        ),
                        explanation=(
                            f"Reference '{ev.reference_id}' does not match "
                            f"any extracted evidence for '{ev.specialist}'."
                        ),
                    )
                )

        if not malformed and not unmatched:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="All evidence items have verified provenance.",
                    explanation=(
                        f"All {len(all_items)} evidence references match "
                        "source specialist evidence."
                    ),
                )
            )

    def _check_agreement_validity(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Validate agreement findings are supported by consistent stances.

        Contradictory structured stances are FAILED.
        Relationships that cannot be deterministically verified are reported
        as explicit WARNINGs rather than falsely certified as proven.
        """
        check_id = "CHK_AGREEMENT_VALIDITY"
        spurious: List[Tuple[str, str, str]] = []
        unverifiable_findings: List[Tuple[str, str]] = []
        verified_count = 0

        for finding in analysis.areas_of_agreement:
            t_lower = finding.topic.lower()
            s_lower = finding.summary.lower()
            is_positive = any(
                w in t_lower or w in s_lower
                for w in (
                    "bullish",
                    "positive",
                    "growth",
                    "favorable",
                    "strong",
                    "upside",
                )
            )
            is_negative = any(
                w in t_lower or w in s_lower
                for w in (
                    "bearish",
                    "negative",
                    "contraction",
                    "unfavorable",
                    "weak",
                    "downside",
                    "headwind",
                )
            )

            concordant_found = False
            finding_unverifiable = True

            for s in finding.supporting_specialists:
                s_str = _val(s).lower()
                stance = get_specialist_directional_stance(s_str, input_data)

                # Format descriptive reason including raw field value
                if (
                    s_str == "fundamental"
                    and input_data.has_fundamental
                    and input_data.fundamental
                ):
                    raw_val = _val(input_data.fundamental.overall_assessment)
                    raw_desc = f"assessment is '{raw_val}'"
                elif (
                    s_str == "technical"
                    and input_data.has_technical
                    and input_data.technical
                ):
                    raw_desc = f"trend is '{_val(input_data.technical.trend)}'"
                elif s_str == "news" and input_data.has_news and input_data.news:
                    raw_desc = (
                        f"sentiment is '{_val(input_data.news.overall_sentiment)}'"
                    )
                elif s_str == "risk" and input_data.has_risk and input_data.risk:
                    raw_desc = (
                        f"risk level is '{_val(input_data.risk.overall_risk_level)}'"
                    )
                else:
                    raw_desc = f"stance is '{stance}'"

                if is_positive:
                    if stance == "negative":
                        spurious.append(
                            (
                                finding.topic,
                                s_str,
                                f"{s_str} {raw_desc}",
                            )
                        )
                    elif stance == "positive":
                        concordant_found = True
                        finding_unverifiable = False
                elif is_negative:
                    if stance == "positive":
                        spurious.append(
                            (
                                finding.topic,
                                s_str,
                                f"{s_str} {raw_desc}",
                            )
                        )
                    elif stance == "negative":
                        concordant_found = True
                        finding_unverifiable = False
                else:
                    if stance in ("positive", "negative", "neutral"):
                        finding_unverifiable = False

            if concordant_found:
                verified_count += 1
            elif finding_unverifiable:
                unverifiable_findings.append(
                    (finding.topic, ", ".join(finding.supporting_specialists))
                )

        if spurious:
            for topic, spec, reason in spurious:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.HIGH,
                        affected_specialists=[spec],
                        message=(
                            f"Agreement '{topic}' claims concordant stance but "
                            f"'{spec}' contradicts ({reason})."
                        ),
                        explanation=(
                            f"Supporting specialist '{spec}' reported a "
                            f"contradictory signal ({reason})."
                        ),
                    )
                )
        else:
            if unverifiable_findings:
                for topic, specs in unverifiable_findings:
                    issues.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.WARNING,
                            severity=ConsistencyCheckSeverity.LOW,
                            affected_specialists=[s.strip() for s in specs.split(",")],
                            message=(
                                f"Agreement '{topic}' is UNVERIFIABLE from "
                                "structured specialist fields."
                            ),
                            explanation=(
                                f"Agreement topic '{topic}' with specialists [{specs}] "
                                "lacks deterministic directional metrics in "
                                "structured fields."
                            ),
                        )
                    )
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message=(
                        f"All {len(analysis.areas_of_agreement)} agreements "
                        "grounded without contradictory specialist stances."
                    ),
                    explanation=(
                        f"Verified {verified_count} concordant agreement findings; "
                        "no opposing structured stances detected."
                    ),
                )
            )

    def _check_conflict_validity(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Validate signal conflicts represent genuine contradictions.

        Artificial conflicts between clearly agreeing structured stances are FAILED.
        Conflicts that cannot be deterministically verified are reported as WARNINGs.
        """
        check_id = "CHK_CONFLICT_VALIDITY"
        spurious: List[Tuple[str, str]] = []
        unverifiable: List[Tuple[str, str]] = []
        verified_conflicts = 0

        if not analysis.signal_conflicts:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message=(
                        "No signal conflicts reported (all specialist signals aligned)."
                    ),
                    explanation="Input signals do not exhibit cross-domain tension.",
                )
            )
            return

        for conflict in analysis.signal_conflicts:
            if len(conflict.involved_specialists) < 2:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.HIGH,
                        affected_specialists=[
                            _val(s) for s in conflict.involved_specialists
                        ],
                        message=f"Conflict '{conflict.topic}' has < 2 specialists.",
                        explanation=(
                            f"Requires >= 2 specialists, found "
                            f"{len(conflict.involved_specialists)}."
                        ),
                    )
                )
                continue

            stances = {
                _val(s).lower(): get_specialist_directional_stance(_val(s), input_data)
                for s in conflict.involved_specialists
            }
            known_stances = [st for st in stances.values() if st is not None]

            if len(known_stances) >= 2:
                if all(st == "positive" for st in known_stances):
                    spurious.append(
                        (
                            conflict.topic,
                            "all involved specialists have concordant positive "
                            f"stances ({stances})",
                        )
                    )
                elif all(st == "negative" for st in known_stances):
                    spurious.append(
                        (
                            conflict.topic,
                            "all involved specialists have concordant negative "
                            f"stances ({stances})",
                        )
                    )
                elif "positive" in known_stances and "negative" in known_stances:
                    verified_conflicts += 1
                else:
                    unverifiable.append(
                        (
                            conflict.topic,
                            (
                                "involved specialists have non-opposing "
                                f"stances ({stances})"
                            ),
                        )
                    )
            else:
                unverifiable.append(
                    (
                        conflict.topic,
                        "fewer than 2 specialists have directional structured "
                        f"stances ({stances})",
                    )
                )

        if spurious:
            for topic, reason in spurious:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.HIGH,
                        message=f"Artificial conflict '{topic}': {reason}.",
                        explanation=(
                            f"Conflict '{topic}' asserts contradiction, but "
                            f"specialists agree ({reason})."
                        ),
                    )
                )
        else:
            if unverifiable:
                for topic, reason in unverifiable:
                    issues.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.WARNING,
                            severity=ConsistencyCheckSeverity.LOW,
                            message=(
                                f"Conflict '{topic}' is UNVERIFIABLE from "
                                "structured fields."
                            ),
                            explanation=(
                                f"Conflict '{topic}' cannot be deterministically "
                                f"verified as opposing: {reason}."
                            ),
                        )
                    )
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message=(
                        f"All {len(analysis.signal_conflicts)} signal conflicts "
                        "validated without artificial consensus contradictions."
                    ),
                    explanation=(
                        f"Validated {verified_conflicts} authentic opposing conflicts; "
                        "no artificial contradictions between agreeing specialists."
                    ),
                )
            )

    def _check_cross_observations(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Verify observations reference genuinely available specialists."""
        check_id = "CHK_CROSS_OBS_VALIDITY"
        invalid_obs: List[Tuple[str, str]] = []

        for obs in analysis.cross_specialist_observations:
            if not obs.connected_specialists:
                invalid_obs.append((obs.observation[:35], "no specialists connected"))
                continue
            for s in obs.connected_specialists:
                s_str = _val(s).lower()
                if (
                    input_data.specialist_statuses.get(s_str)
                    != SpecialistStatus.AVAILABLE
                ):
                    invalid_obs.append(
                        (obs.observation[:35], f"'{s_str}' is unavailable")
                    )

        if invalid_obs:
            for snip, reason in invalid_obs:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.HIGH,
                        message=f"Observation invalid ({reason}): '{snip}...'.",
                        explanation=(
                            f"Cross-specialist observation connects unavailable "
                            f"specialists ({reason})."
                        ),
                    )
                )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message=(
                        f"All {len(analysis.cross_specialist_observations)} "
                        "observations connect available specialists."
                    ),
                    explanation=(
                        "Observations link only available specialists and preserve "
                        "valid connections."
                    ),
                )
            )

    def _check_unanimous_signal_sanity(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Sanity check: stance shouldn't contradict unanimous signals."""
        check_id = "CHK_SANITY_UNANIMOUS_STANCE"
        tech = input_data.technical if input_data.has_technical else None
        fund = input_data.fundamental if input_data.has_fundamental else None
        news = input_data.news if input_data.has_news else None
        risk = input_data.risk if input_data.has_risk else None

        stances: List[Tuple[str, int]] = []
        if tech:
            trend = _val(tech.trend).lower()
            if trend in ("uptrend", "bullish"):
                stances.append(("technical", 1))
            elif trend in ("downtrend", "bearish"):
                stances.append(("technical", -1))
        if fund:
            assess = _val(fund.overall_assessment).lower()
            if assess in ("favorable", "bullish"):
                stances.append(("fundamental", 1))
            elif assess in ("unfavorable", "bearish"):
                stances.append(("fundamental", -1))
        if news:
            sent = _val(news.overall_sentiment).lower()
            if sent in ("positive", "bullish"):
                stances.append(("news", 1))
            elif sent in ("negative", "bearish"):
                stances.append(("news", -1))
        if risk:
            lvl = _val(risk.overall_risk_level).lower()
            if lvl == "low":
                stances.append(("risk", 1))
            elif lvl in ("critical", "high"):
                stances.append(("risk", -1))

        if len(stances) >= 2:
            all_pos = all(d == 1 for _, d in stances)
            all_neg = all(d == -1 for _, d in stances)
            syn_lower = analysis.overall_synthesis.lower()

            if all_pos:
                contradicts = (
                    "severely deteriorating" in syn_lower
                    or "bearish breakdown" in syn_lower
                    or "critical collapse" in syn_lower
                ) and len(analysis.signal_conflicts) == 0
                if contradicts:
                    issues.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.FAILED,
                            severity=ConsistencyCheckSeverity.HIGH,
                            message=(
                                "Synthesis contradicts unanimous positive "
                                "signals without explanation."
                            ),
                            explanation=(
                                f"Specialists ({[s for s, _ in stances]}) are "
                                "positive, but synthesis reported negativity."
                            ),
                        )
                    )
                    return

            if all_neg:
                contradicts = (
                    "exceptional growth" in syn_lower
                    or "unblemished health" in syn_lower
                    or "strong bullish expansion" in syn_lower
                ) and len(analysis.signal_conflicts) == 0
                if contradicts:
                    issues.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.FAILED,
                            severity=ConsistencyCheckSeverity.HIGH,
                            message=(
                                "Synthesis contradicts unanimous negative "
                                "signals without explanation."
                            ),
                            explanation=(
                                f"Specialists ({[s for s, _ in stances]}) are "
                                "negative, but synthesis concluded bullishness."
                            ),
                        )
                    )
                    return

        issues.append(
            ConsistencyIssue(
                check_id=check_id,
                status=ConsistencyCheckStatus.PASSED,
                severity=ConsistencyCheckSeverity.LOW,
                message="Overall stance aligns with specialist signals.",
                explanation=(
                    "No unexplained contradictions against unanimous specialist "
                    "signals were found."
                ),
            )
        )

    def _check_confidence_completeness(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Verify confidence reflects completeness and signal conflicts."""
        check_id = "CHK_CONFIDENCE_CONSISTENCY"
        completeness = input_data.core_completeness_ratio
        conf = analysis.confidence

        if conf < 0.0 or conf > 1.0:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.FAILED,
                    severity=ConsistencyCheckSeverity.CRITICAL,
                    message=f"Confidence {conf} is out of valid range [0.0, 1.0].",
                    explanation=f"Expected confidence in [0.0, 1.0], found {conf}.",
                )
            )
        elif input_data.is_empty and conf > 0.05:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.FAILED,
                    severity=ConsistencyCheckSeverity.CRITICAL,
                    message=f"Confidence {conf} is invalid for empty input.",
                    explanation=(
                        f"Expected confidence <= 0.05 on empty input, found {conf}."
                    ),
                )
            )
        elif completeness < 0.4 and conf > 0.75:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.WARNING,
                    severity=ConsistencyCheckSeverity.MEDIUM,
                    message=(
                        f"Confidence {conf} is elevated for "
                        f"completeness {completeness:.2f}."
                    ),
                    explanation=(
                        f"Completeness is low ({completeness:.2f}); confidence "
                        f"{conf} is elevated (> 0.75)."
                    ),
                )
            )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message=(
                        f"Confidence {conf:.2f} aligns with completeness "
                        f"{completeness:.2f}."
                    ),
                    explanation=(
                        f"Confidence {conf:.2f} matches completeness ratio "
                        f"{completeness:.2f}."
                    ),
                )
            )

    def _check_missing_failed_specialists(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Verify missing and failed specialists are properly recorded."""
        check_id = "CHK_MISSING_FAILED_SPECIALISTS"
        missing_diff = set(input_data.missing_specialists) - set(
            analysis.missing_specialists
        )
        failed_diff = set(input_data.failed_specialists) - set(
            analysis.failed_specialists
        )

        if missing_diff:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.FAILED,
                    severity=ConsistencyCheckSeverity.MEDIUM,
                    affected_specialists=[_val(s) for s in missing_diff],
                    message=f"Missing specialists {list(missing_diff)} omitted.",
                    explanation=(
                        f"Expected missing specialists from input contract, "
                        f"but {list(missing_diff)} was omitted."
                    ),
                )
            )
        elif failed_diff:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.FAILED,
                    severity=ConsistencyCheckSeverity.HIGH,
                    affected_specialists=[_val(s) for s in failed_diff],
                    message=f"Failed specialists {list(failed_diff)} omitted.",
                    explanation=(
                        f"Expected failed specialists from input, but "
                        f"{list(failed_diff)} was omitted."
                    ),
                )
            )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="Missing and failed specialists accurately recorded.",
                    explanation=(
                        f"Recorded {len(analysis.missing_specialists)} missing "
                        f"and {len(analysis.failed_specialists)} failed specialists."
                    ),
                )
            )

    def _check_safety_prohibited_language(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Verify strict adherence to non-advisory boundaries."""
        check_id = "CHK_SAFETY_ADVISORY"
        prohibited: List[Tuple[str, str]] = []

        texts_to_scan = [
            ("synthesis", analysis.overall_synthesis),
        ]
        for f in analysis.areas_of_agreement:
            texts_to_scan.append((f"agreement '{f.topic}'", f.summary))
        for c in analysis.signal_conflicts:
            texts_to_scan.append((f"conflict '{c.topic}'", c.description))
        for o in analysis.cross_specialist_observations:
            texts_to_scan.append(("observation", o.observation))

        for loc, text in texts_to_scan:
            for pattern in PROHIBITED_ADVICE_PATTERNS:
                m = pattern.search(text)
                if m:
                    prohibited.append((loc, m.group(0)))

        if prohibited:
            for loc, match_str in prohibited:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.CRITICAL,
                        message=f"Prohibited language '{match_str}' found in {loc}.",
                        explanation=(
                            f"Checked safety in {loc}. Found prohibited phrase "
                            f"matching '{match_str}'."
                        ),
                    )
                )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="Non-advisory boundaries confirmed.",
                    explanation=(
                        "Scanned all text sections for prohibited advice, price "
                        "targets, and guarantees. No violations found."
                    ),
                )
            )

    def _check_quantitative_provenance(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Validate numeric values cited in statements originate from source data.

        Preserves specialist attribution and evidence reference attribution.
        A quantitative claim must either:
        1. have explicit/traceable evidence attribution that contains the value, or
        2. be deterministically associated with a supplied structured metric where
           the field/value relationship is unambiguous.
        """
        check_id = "CHK_QUANT_PROVENANCE"
        if input_data.is_empty:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="Skipped quantitative provenance for empty input.",
                    explanation=(
                        "Input contract has no specialists to check quantitative "
                        "provenance against."
                    ),
                )
            )
            return

        spec_numbers = extract_specialist_numbers(input_data)
        failures: List[ConsistencyIssue] = []

        # 1. Check Findings in areas_of_agreement
        for finding in analysis.areas_of_agreement:
            claimed_specs = [_val(s).lower() for s in finding.supporting_specialists]
            finding_ev_numbers: Set[float] = set()
            for ev in finding.evidence:
                finding_ev_numbers.update(extract_numbers_from_text(ev.detail))

            for num in extract_numbers_from_text(finding.summary):
                # Check 1: In attached evidence
                if _number_in_set(num, finding_ev_numbers):
                    continue
                # Check 2: In claimed specialists' structured data
                claimed_has_num = any(
                    _number_in_set(num, spec_numbers.get(s, set()))
                    for s in claimed_specs
                )
                if claimed_has_num:
                    continue

                # Not in claimed specialists! Check where it exists
                other_specs = [
                    s
                    for s, nums in spec_numbers.items()
                    if s not in claimed_specs and _number_in_set(num, nums)
                ]
                if other_specs:
                    failures.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.FAILED,
                            severity=ConsistencyCheckSeverity.HIGH,
                            affected_specialists=claimed_specs,
                            message=(
                                f"Value {num} in agreement '{finding.topic}' "
                                f"belongs to {other_specs} but is attributed "
                                f"to {claimed_specs}."
                            ),
                            explanation=(
                                f"Provenance mismatch: numeric claim {num} "
                                "does not appear in claimed source specialists "
                                f"{claimed_specs}. Found only in {other_specs}."
                            ),
                        )
                    )
                else:
                    failures.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.FAILED,
                            severity=ConsistencyCheckSeverity.MEDIUM,
                            affected_specialists=claimed_specs,
                            message=(
                                f"Value {num} in agreement '{finding.topic}' "
                                "lacks source provenance."
                            ),
                            explanation=(
                                f"Numeric value {num} in agreement '{finding.topic}' "
                                "does not appear in any specialist source data."
                            ),
                        )
                    )

        # 2. Check Cross-Specialist Observations
        for obs in analysis.cross_specialist_observations:
            conn_specs = [_val(s).lower() for s in obs.connected_specialists]
            obs_ev_numbers: Set[float] = set()
            for ev in obs.evidence:
                obs_ev_numbers.update(extract_numbers_from_text(ev.detail))

            for num in extract_numbers_from_text(obs.observation):
                if _number_in_set(num, obs_ev_numbers):
                    continue
                conn_has_num = any(
                    _number_in_set(num, spec_numbers.get(s, set())) for s in conn_specs
                )
                if conn_has_num:
                    continue

                other_specs = [
                    s
                    for s, nums in spec_numbers.items()
                    if s not in conn_specs and _number_in_set(num, nums)
                ]
                if other_specs:
                    failures.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.FAILED,
                            severity=ConsistencyCheckSeverity.HIGH,
                            affected_specialists=conn_specs,
                            message=(
                                f"Value {num} in observation belongs to {other_specs} "
                                f"but is attributed to {conn_specs}."
                            ),
                            explanation=(
                                f"Observation references {conn_specs}, but value {num} "
                                f"originates in {other_specs}."
                            ),
                        )
                    )
                else:
                    failures.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.FAILED,
                            severity=ConsistencyCheckSeverity.MEDIUM,
                            affected_specialists=conn_specs,
                            message=(
                                f"Value {num} in observation lacks source provenance."
                            ),
                            explanation=(
                                f"Value {num} does not appear in any specialist data."
                            ),
                        )
                    )

        # 3. Check Signal Conflicts
        for conflict in analysis.signal_conflicts:
            inv_specs = [_val(s).lower() for s in conflict.involved_specialists]
            conf_ev_numbers: Set[float] = set()
            for ev in conflict.evidence:
                conf_ev_numbers.update(extract_numbers_from_text(ev.detail))

            for num in extract_numbers_from_text(conflict.description):
                if _number_in_set(num, conf_ev_numbers):
                    continue
                inv_has_num = any(
                    _number_in_set(num, spec_numbers.get(s, set())) for s in inv_specs
                )
                if not inv_has_num:
                    failures.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.FAILED,
                            severity=ConsistencyCheckSeverity.HIGH,
                            affected_specialists=inv_specs,
                            message=(
                                f"Value {num} in conflict '{conflict.topic}' "
                                f"lacks provenance in {inv_specs}."
                            ),
                            explanation=(
                                f"Value {num} does not appear in involved "
                                f"specialists {inv_specs}."
                            ),
                        )
                    )

            for s_name, pos_text in conflict.specialist_positions.items():
                s_key = s_name.lower()
                for num in extract_numbers_from_text(pos_text):
                    if not _number_in_set(num, spec_numbers.get(s_key, set())):
                        failures.append(
                            ConsistencyIssue(
                                check_id=check_id,
                                status=ConsistencyCheckStatus.FAILED,
                                severity=ConsistencyCheckSeverity.HIGH,
                                affected_specialists=[s_key],
                                message=(
                                    f"Value {num} in position of '{s_name}' "
                                    f"lacks provenance in {s_name}."
                                ),
                                explanation=(
                                    f"Value {num} not found in specialist '{s_name}'."
                                ),
                            )
                        )

        # 4. Check Overall Synthesis (Global narrative)
        # Numbers verified in agreements/observations/conflicts are grounded
        grounded_narrative_numbers: Set[float] = set()
        for finding in analysis.areas_of_agreement:
            grounded_narrative_numbers.update(
                extract_numbers_from_text(finding.summary)
            )
        for obs in analysis.cross_specialist_observations:
            grounded_narrative_numbers.update(
                extract_numbers_from_text(obs.observation)
            )
        for conf in analysis.signal_conflicts:
            grounded_narrative_numbers.update(
                extract_numbers_from_text(conf.description)
            )

        synth_numbers = extract_numbers_from_text(analysis.overall_synthesis)
        for num in synth_numbers:
            if _number_in_set(num, grounded_narrative_numbers):
                continue

            candidates = [
                s for s, nums in spec_numbers.items() if _number_in_set(num, nums)
            ]

            if len(candidates) == 0:
                failures.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.MEDIUM,
                        message=(
                            f"Value {num} in overall synthesis lacks source provenance."
                        ),
                        explanation=(
                            f"Numeric value {num} in overall synthesis does not "
                            "appear in any specialist source data or evidence."
                        ),
                    )
                )
            elif len(candidates) > 1:
                failures.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.FAILED,
                        severity=ConsistencyCheckSeverity.HIGH,
                        affected_specialists=candidates,
                        message=(
                            f"Value {num} in overall synthesis appears across multiple "
                            f"specialists {candidates} without explicit attribution."
                        ),
                        explanation=(
                            f"Ambiguous quantitative provenance: {num} exists in "
                            f"multiple specialists {candidates}. Lacks explicit "
                            "attribution."
                        ),
                    )
                )
            else:
                # Exactly one candidate specialist. Check domain consistency
                cand_spec = candidates[0]
                s_lower = analysis.overall_synthesis.lower()
                fund_keywords = {
                    "revenue",
                    "margin",
                    "cash flow",
                    "fcf",
                    "debt",
                    "ebitda",
                    "pe ratio",
                    "p/e",
                }
                tech_keywords = {
                    "rsi",
                    "sma",
                    "price",
                    "macd",
                    "support",
                    "resistance",
                }
                if (
                    cand_spec == "technical"
                    and any(k in s_lower for k in fund_keywords)
                    and not any(k in s_lower for k in tech_keywords)
                ):
                    failures.append(
                        ConsistencyIssue(
                            check_id=check_id,
                            status=ConsistencyCheckStatus.FAILED,
                            severity=ConsistencyCheckSeverity.HIGH,
                            affected_specialists=[cand_spec],
                            message=(
                                f"Value {num} in overall synthesis belongs to "
                                "technical data but is associated with fundamental "
                                "narrative keywords."
                            ),
                            explanation=(
                                f"Value {num} is found only in technical data, yet the "
                                "narrative discusses fundamental metrics."
                            ),
                        )
                    )

        if failures:
            issues.extend(failures)
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="All cited quantitative values originate in source data.",
                    explanation=(
                        "Verified all quantitative claims against attributed "
                        "specialist metrics and evidence details."
                    ),
                )
            )

    def _check_prompt_injection(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Detect prompt injection in specialist text and verify isolation."""
        check_id = "CHK_PROMPT_INJECTION"
        detected: List[Tuple[str, str]] = []

        texts: List[Tuple[str, str]] = []
        if input_data.has_technical and input_data.technical:
            texts.append(("technical", str(input_data.technical.model_dump())))
        if input_data.has_fundamental and input_data.fundamental:
            texts.append(("fundamental", str(input_data.fundamental.model_dump())))
        if input_data.has_news and input_data.news:
            texts.append(("news", str(input_data.news.model_dump())))
        if input_data.has_research and input_data.research:
            texts.append(("research", str(input_data.research.model_dump())))
        if input_data.has_risk and input_data.risk:
            texts.append(("risk", str(input_data.risk.model_dump())))

        for source_name, text_val in texts:
            for pattern in PROMPT_INJECTION_PATTERNS:
                m = pattern.search(text_val)
                if m:
                    detected.append((source_name, m.group(0)))

        if detected:
            for source_name, match_text in detected:
                issues.append(
                    ConsistencyIssue(
                        check_id=check_id,
                        status=ConsistencyCheckStatus.WARNING,
                        severity=ConsistencyCheckSeverity.HIGH,
                        affected_specialists=[source_name],
                        message=(
                            f"Prompt injection in '{source_name}': '{match_text}'."
                        ),
                        explanation=(
                            f"Injection pattern '{match_text}' in '{source_name}'. "
                            "Isolated strictly as DATA."
                        ),
                    )
                )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="No prompt injection patterns detected in inputs.",
                    explanation=(
                        "Scanned all specialist payloads for injection vectors. "
                        "Inputs are clean and treated as data."
                    ),
                )
            )

    def _check_duplicate_evidence(
        self,
        input_data: ReportAggregatorInput,
        analysis: UnifiedSpecialistAnalysis,
        issues: List[ConsistencyIssue],
    ) -> None:
        """Detect and warn if duplicate items are repeated in evidence.

        Canonical evidence identity is (specialist, reference_id).
        Multiple items with the same specialist and reference_id (even with
        different detail text) are duplicates.
        """
        check_id = "CHK_DUPLICATE_EVIDENCE"
        seen_identities: Set[Tuple[str, str]] = set()
        duplicates: List[Tuple[str, str]] = []

        for ev in analysis.aggregated_evidence:
            s_str = _val(ev.specialist).lower()
            ref_id = ev.reference_id.strip() if ev.reference_id else ""
            identity = (s_str, ref_id)
            if identity in seen_identities:
                duplicates.append(identity)
            else:
                seen_identities.add(identity)

        if duplicates:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.WARNING,
                    severity=ConsistencyCheckSeverity.LOW,
                    affected_evidence_refs=[ref for _, ref in duplicates],
                    message=(
                        f"Detected {len(duplicates)} duplicate evidence references."
                    ),
                    explanation=(
                        "Found duplicate items for canonical identity "
                        f"(specialist, ref_id): {duplicates}."
                    ),
                )
            )
        else:
            issues.append(
                ConsistencyIssue(
                    check_id=check_id,
                    status=ConsistencyCheckStatus.PASSED,
                    severity=ConsistencyCheckSeverity.LOW,
                    message="No duplicate evidence items in aggregated evidence.",
                    explanation=(
                        f"Checked {len(analysis.aggregated_evidence)} evidence items. "
                        "All items have unique canonical "
                        "(specialist, reference_id) identities."
                    ),
                )
            )


def validate_aggregation_consistency(
    input_data: ReportAggregatorInput,
    analysis: UnifiedSpecialistAnalysis,
) -> AggregationConsistencyReport:
    """Convenience functional wrapper to run all consistency checks."""
    checker = AggregationConsistencyChecker()
    return checker.validate(input_data, analysis)
