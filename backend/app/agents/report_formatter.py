"""Report Output Formatting and Rendering Engine (Phase 12.4).

Fulfills Phase 12.4 requirements (plan.md 12.4.1 & 12.4.2):
- 12.4.1 Structured JSON report (for frontend rendering and API consumption).
- 12.4.2 Human-readable text and Markdown rendering of the same report.

Features:
- format_report_markdown: Full publication-grade Markdown rendering with headers,
  tables, callouts, specialist breakdowns, evidence references, and disclaimer.
- format_report_text: Clean plain text ASCII formatted representation.
- format_report_json: Standard round-trippable JSON serialization.
- format_report_frontend_dict: Enriched structured dictionary ready for React /
  frontend card, badge, and tab UI components.
- format_report_frontend_json: JSON string of the frontend UI structure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.agents.aggregator_schema import SpecialistStatus
from app.agents.report_schema import (
    FinalReport,
)


def _val(x: Any) -> str:
    """Safely extract string value from enum or object."""
    if x is None:
        return ""
    if hasattr(x, "value"):
        return str(x.value)
    return str(x)


def _is_report_partial(report: FinalReport) -> bool:
    """Determine whether a report represents partial specialist coverage."""
    return (
        not report.is_complete
        or bool(report.missing_specialists)
        or bool(report.failed_specialists)
        or (
            report.overall_assessment is not None
            and getattr(report.overall_assessment, "data_completeness_ratio", 1.0) < 1.0
        )
    )


# ===========================================================================
# 1. MARKDOWN RENDERING (Phase 12.4.2)
# ===========================================================================


def format_report_markdown(
    report: FinalReport,
    include_disclaimer: bool = True,
    include_table_of_contents: bool = False,
) -> str:
    """Render a FinalReport into a structured, human-readable Markdown document.

    Args:
        report: The validated FinalReport instance.
        include_disclaimer: Whether to include the mandatory regulatory disclaimer.
        include_table_of_contents: Whether to include a Markdown TOC.

    Returns:
        Structured Markdown string.
    """
    lines: List[str] = []

    # 1. Header & Metadata
    company_name = report.company.name or report.company.ticker
    lines.append(
        f"# Investment Research Report: {report.company.ticker} ({company_name})"
    )
    lines.append("")

    gen_time_raw = getattr(report, "generated_at", None)
    gen_time = (
        gen_time_raw.strftime("%Y-%m-%d %H:%M UTC")
        if gen_time_raw
        else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
    meta_parts = [f"**Generated:** {gen_time}"]
    if report.company.currency:
        meta_parts.append(f"**Currency:** {report.company.currency}")
    completeness = (
        report.overall_assessment.data_completeness_ratio
        if report.overall_assessment
        and hasattr(report.overall_assessment, "data_completeness_ratio")
        else getattr(report, "data_completeness_ratio", None)
    )
    if completeness is not None:
        pct = int(round(completeness * 100))
        meta_parts.append(f"**Data Completeness:** {pct}%")
    is_partial = _is_report_partial(report)
    if is_partial:
        meta_parts.append("**Status:** ⚠️ Partial Analysis")
    lines.append(" | ".join(meta_parts))
    lines.append("")

    if is_partial:
        lines.append(
            "> ⚠️ **Notice: Partial Report** — One or more specialist analysis "
            "modules were unavailable or encountered errors. Findings represent "
            "partial coverage."
        )
        lines.append("")

    # Optional Table of Contents
    if include_table_of_contents:
        lines.append("## Table of Contents")
        lines.append("- [1. Investor Context](#1-investor-context)")
        lines.append(
            "- [2. Executive Summary & Recommendation]"
            "(#2-executive-summary--recommendation)"
        )
        lines.append("- [3. Specialist Analysis](#3-specialist-analysis)")
        lines.append("  - [Technical Analysis](#technical-analysis)")
        lines.append("  - [Fundamental Analysis](#fundamental-analysis)")
        lines.append("  - [News & Sentiment](#news--sentiment)")
        lines.append("  - [SEC & Document Research](#sec--document-research)")
        lines.append("  - [Risk Analysis](#risk-analysis)")
        lines.append("- [4. Key Investment Risks](#4-key-investment-risks)")
        lines.append("- [5. Evidence & Provenance](#5-evidence--provenance)")
        if include_disclaimer:
            lines.append("- [6. Regulatory Disclaimer](#6-regulatory-disclaimer)")
        lines.append("")
        lines.append("---")
        lines.append("")

    # 2. Investor Context
    lines.append("## 1. Investor Context")
    lines.append("")
    profile_type = "Standard"
    risk_tol = "Unspecified"
    if report.investor_profile:
        profile_type = (
            _val(getattr(report.investor_profile, "investment_goal", None))
            or _val(getattr(report.investor_profile, "target_profile", None))
            or profile_type
        )
        risk_tol = (
            _val(getattr(report.investor_profile, "risk_tolerance", None)) or risk_tol
        )

    lines.append(f"- **Target Profile:** {profile_type}")
    lines.append(f"- **Risk Tolerance:** {risk_tol}")
    if report.horizon:
        lines.append(f"- **Investment Horizon:** {report.horizon}")
    if report.capital and report.capital.amount is not None:
        cap_str = report.capital.formatted or f"${report.capital.amount:,.2f}"
        lines.append(f"- **Allocated Capital:** {cap_str}")
    if report.company.sector or report.company.industry:
        sector_str = report.company.sector or "N/A"
        ind_str = report.company.industry or "N/A"
        lines.append(f"- **Sector / Industry:** {sector_str} / {ind_str}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. Executive Summary & Recommendation
    lines.append("## 2. Executive Summary & Recommendation")
    lines.append("")

    stance_val = (
        _val(report.recommendation.stance).upper()
        if report.recommendation and report.recommendation.stance
        else "NEUTRAL"
    )

    badge_emoji = {
        "FAVORABLE": "🟢",
        "NEUTRAL": "🟡",
        "CAUTIOUS": "🟠",
        "UNFAVORABLE": "🔴",
        "INSUFFICIENT_EVIDENCE": "⚪",
    }.get(stance_val, "🔵")

    lines.append(f"### Recommendation: {badge_emoji} **{stance_val}**")
    lines.append("")

    if report.recommendation:
        if report.recommendation.rationale:
            lines.append(
                f"> **Strategic Rationale:**  \n> {report.recommendation.rationale}"
            )
            lines.append("")
        if report.recommendation.profile_alignment:
            lines.append(
                f"**Investor Profile Alignment:**  \n"
                f"{report.recommendation.profile_alignment}"
            )
            lines.append("")

    overall_text = (
        report.overall_assessment.synthesis
        if hasattr(report.overall_assessment, "synthesis")
        else str(report.overall_assessment or "")
    )
    if overall_text:
        lines.append(f"**Overall Assessment:**  \n{overall_text}")
        lines.append("")

    cross_synth = getattr(report, "cross_specialist_synthesis", None)
    if cross_synth:
        lines.append(f"**Cross-Specialist Synthesis:**  \n" f"{cross_synth}")
        lines.append("")

    if report.key_reasons:
        lines.append("### Key Supporting Reasons")
        for reason in report.key_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 4. Specialist Analysis Breakdowns
    lines.append("## 3. Specialist Analysis")
    lines.append("")

    # 4.1 Technical Analysis
    lines.append("### Technical Analysis")
    if report.technical is not None:
        trend_str = report.technical.trend or "Unspecified"
        score_str = (
            f"{report.technical.technical_score:.1f}/100"
            if report.technical.technical_score is not None
            else "N/A"
        )
        lines.append(f"- **Trend Direction:** {trend_str} (Score: {score_str})")
        lines.append(f"- **Summary:** {report.technical.summary}")
        if report.technical.momentum:
            lines.append(f"- **Momentum:** {report.technical.momentum}")
        if report.technical.support_levels:
            sup_str = ", ".join(
                f"${s:,.2f}" if isinstance(s, (int, float)) else str(s)
                for s in report.technical.support_levels
            )
            lines.append(f"- **Support Levels:** {sup_str}")
        if report.technical.resistance_levels:
            res_str = ", ".join(
                f"${r:,.2f}" if isinstance(r, (int, float)) else str(r)
                for r in report.technical.resistance_levels
            )
            lines.append(f"- **Resistance Levels:** {res_str}")
    else:
        tech_status = report.specialist_statuses.get(
            "technical", SpecialistStatus.MISSING
        )
        lines.append(
            f"> ⚠️ **Technical Analysis Unavailable** "
            f"(Status: {_val(tech_status).upper()})"
        )
    lines.append("")

    # 4.2 Fundamental Analysis
    lines.append("### Fundamental Analysis")
    if report.fundamental is not None:
        fund_rating = report.fundamental.overall_assessment or "Unspecified"
        lines.append(f"- **Overall Assessment:** {fund_rating.capitalize()}")
        lines.append(f"- **Summary:** {report.fundamental.summary}")
        if report.fundamental.financial_health:
            lines.append(
                f"- **Financial Health:** {report.fundamental.financial_health}"
            )
        if report.fundamental.profitability:
            lines.append(f"- **Profitability:** {report.fundamental.profitability}")
        if report.fundamental.valuation:
            lines.append(f"- **Valuation:** {report.fundamental.valuation}")
        if report.fundamental.growth:
            lines.append(f"- **Growth:** {report.fundamental.growth}")
        if report.fundamental.cash_flow:
            lines.append(f"- **Cash Flow:** {report.fundamental.cash_flow}")
        if report.fundamental.key_strengths:
            strengths_str = ", ".join(report.fundamental.key_strengths)
            lines.append(f"- **Key Strengths:** {strengths_str}")
        if report.fundamental.key_weaknesses:
            weaknesses_str = ", ".join(report.fundamental.key_weaknesses)
            lines.append(f"- **Key Weaknesses:** {weaknesses_str}")
    else:
        fund_status = report.specialist_statuses.get(
            "fundamental", SpecialistStatus.MISSING
        )
        lines.append(
            f"> ⚠️ **Fundamental Analysis Unavailable** "
            f"(Status: {_val(fund_status).upper()})"
        )
    lines.append("")

    # 4.3 News & Sentiment
    lines.append("### News & Sentiment")
    if report.news is not None:
        sent_str = report.news.overall_sentiment or "Neutral"
        score_val = (
            f"{report.news.sentiment_score:+.2f}"
            if report.news.sentiment_score is not None
            else "N/A"
        )
        lines.append(
            f"- **Overall Sentiment:** {sent_str.capitalize()} (Score: {score_val})"
        )
        lines.append(f"- **Summary:** {report.news.summary}")
        if report.news.key_themes:
            lines.append(f"- **Key Themes:** {', '.join(report.news.key_themes)}")
        recent_items = getattr(report.news, "recent_headlines", None) or getattr(
            report.news, "recent_events", []
        )
        if recent_items:
            lines.append("- **Recent Headlines / Events:**")
            for ev in recent_items[:5]:
                lines.append(f"  - {ev}")
    else:
        news_status = report.specialist_statuses.get("news", SpecialistStatus.MISSING)
        lines.append(
            f"> ⚠️ **News & Sentiment Unavailable** "
            f"(Status: {_val(news_status).upper()})"
        )
    lines.append("")

    # 4.4 SEC & Document Research
    lines.append("### SEC & Document Research")
    if report.research is not None:
        lines.append(f"- **Summary:** {report.research.summary}")
        if report.research.key_findings:
            lines.append("- **Key Findings:**")
            for f in report.research.key_findings[:5]:
                lines.append(f"  - {f}")
        citations = getattr(report.research, "document_citations", None) or getattr(
            report.research, "retrieved_documents", []
        )
        if citations:
            docs_str = ", ".join(citations)
            lines.append(f"- **Document Citations:** {docs_str}")
    else:
        res_status = report.specialist_statuses.get(
            "research", SpecialistStatus.MISSING
        )
        lines.append(
            f"> ⚠️ **Document Research Unavailable** "
            f"(Status: {_val(res_status).upper()})"
        )
    lines.append("")

    # 4.5 Risk Analysis
    lines.append("### Risk Analysis")
    if report.risk is not None:
        risk_lvl = report.risk.overall_risk_level or "Unspecified"
        lines.append(f"- **Overall Risk Level:** {risk_lvl.capitalize()}")
        lines.append(f"- **Summary:** {report.risk.summary}")
        if report.risk.quantitative_score is not None:
            lines.append(
                f"- **Quantitative Score:** {report.risk.quantitative_score:.1f}/100"
            )
        if report.risk.top_risk_factors:
            lines.append("- **Top Risk Factors:**")
            for r in report.risk.top_risk_factors[:5]:
                lines.append(f"  - {r}")
    else:
        risk_status = report.specialist_statuses.get("risk", SpecialistStatus.MISSING)
        lines.append(
            f"> ⚠️ **Risk Analysis Unavailable** (Status: {_val(risk_status).upper()})"
        )
    lines.append("")

    lines.append("---")
    lines.append("")

    # 5. Important Risks & Caveats
    lines.append("## 4. Key Investment Risks")
    lines.append("")
    if report.important_risks:
        for r in report.important_risks:
            lines.append(f"- ⚠️ {r}")
    else:
        lines.append("No critical material risks identified.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. Evidence Sources & Provenance
    lines.append("## 5. Evidence & Provenance")
    lines.append("")
    if report.evidence_sources:
        lines.append("| Ref ID | Specialist | Document / Chunk | Detail |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for item in report.evidence_sources:
            ref_id = f"`{item.reference_id}`"
            spec = _val(item.specialist).capitalize()
            doc_part = "-"
            if item.document_id or item.chunk_id:
                d = item.document_id or ""
                c = item.chunk_id or ""
                doc_part = f"{d}:{c}".strip(":")
            detail = item.detail.replace("|", "/")
            lines.append(f"| {ref_id} | {spec} | {doc_part} | {detail} |")
    else:
        lines.append("No explicit evidence sources recorded.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 7. Regulatory Disclaimer
    if include_disclaimer:
        lines.append("## 6. Regulatory Disclaimer")
        lines.append("")
        disclaimer_text = report.disclaimer or (
            "FinPilot provides automated financial research and decision support "
            "for informational purposes only. FinPilot is not a registered investment "
            "advisor, broker-dealer, or financial planner."
        )
        lines.append(f"> **Important Regulatory Notice:**  \n> {disclaimer_text}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


# ===========================================================================
# 2. PLAIN TEXT RENDERING (Phase 12.4.2)
# ===========================================================================


def format_report_text(report: FinalReport, include_disclaimer: bool = True) -> str:
    """Render a FinalReport into a clean ASCII-formatted plain text document."""
    divider_heavy = "=" * 78
    divider_light = "-" * 78

    lines: List[str] = []
    lines.append(divider_heavy)
    company_name = report.company.name or report.company.ticker
    lines.append(
        f"FINPILOT INVESTMENT RESEARCH REPORT: {report.company.ticker} ({company_name})"
    )
    lines.append(divider_heavy)

    gen_time_raw = getattr(report, "generated_at", None)
    gen_time = (
        gen_time_raw.strftime("%Y-%m-%d %H:%M UTC")
        if gen_time_raw
        else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
    lines.append(
        f"Generated: {gen_time} | Currency: {report.company.currency or 'USD'}"
    )
    completeness = (
        report.overall_assessment.data_completeness_ratio
        if report.overall_assessment
        and hasattr(report.overall_assessment, "data_completeness_ratio")
        else getattr(report, "data_completeness_ratio", 1.0)
    )
    pct = int(round((completeness or 1.0) * 100))
    status_lbl = "PARTIAL" if _is_report_partial(report) else "Complete"
    lines.append(f"Completeness: {pct}% | Status: {status_lbl}")
    lines.append(divider_light)

    # Investor Context
    lines.append("1. INVESTOR CONTEXT")
    if report.investor_profile:
        profile_type = (
            _val(getattr(report.investor_profile, "investment_goal", None))
            or _val(getattr(report.investor_profile, "target_profile", None))
            or "Standard"
        )
        lines.append(f"  Target Profile: {profile_type}")
        risk_str = _val(
            getattr(report.investor_profile, "risk_tolerance", "Unspecified")
        )
        lines.append(f"  Risk Tolerance: {risk_str}")
    lines.append(f"  Horizon:        {report.horizon or 'Unspecified'}")
    cap_str = (
        report.capital.formatted
        if report.capital and report.capital.formatted
        else "N/A"
    )
    lines.append(f"  Capital:        {cap_str}")
    lines.append(divider_light)

    # Executive Summary & Recommendation
    lines.append("2. RECOMMENDATION & EXECUTIVE SUMMARY")
    stance_str = (
        _val(report.recommendation.stance).upper()
        if report.recommendation
        else "NEUTRAL"
    )
    lines.append(f"  Stance:    [{stance_str}]")
    if report.recommendation and report.recommendation.rationale:
        lines.append(f"  Rationale: {report.recommendation.rationale}")
    if report.recommendation and report.recommendation.profile_alignment:
        lines.append(f"  Alignment: {report.recommendation.profile_alignment}")
    overall_text = (
        report.overall_assessment.synthesis
        if hasattr(report.overall_assessment, "synthesis")
        else str(report.overall_assessment or "")
    )
    if overall_text:
        lines.append(f"  Overall:   {overall_text}")
    if report.key_reasons:
        lines.append("  Key Reasons:")
        for r in report.key_reasons:
            lines.append(f"    * {r}")
    lines.append(divider_light)

    # Specialist Breakdown
    lines.append("3. SPECIALIST ANALYSIS BREAKDOWN")

    # Technical
    lines.append("  [TECHNICAL ANALYSIS]")
    if report.technical:
        score_val = (
            f"{report.technical.technical_score:.1f}"
            if report.technical.technical_score is not None
            else "N/A"
        )
        lines.append(
            f"    Trend:   {report.technical.trend or 'N/A'} (Score: {score_val}/100)"
        )
        lines.append(f"    Summary: {report.technical.summary}")
    else:
        lines.append("    Status:  UNAVAILABLE")

    # Fundamental
    lines.append("  [FUNDAMENTAL ANALYSIS]")
    if report.fundamental:
        lines.append(f"    Rating:  {report.fundamental.overall_assessment or 'N/A'}")
        lines.append(f"    Summary: {report.fundamental.summary}")
    else:
        lines.append("    Status:  UNAVAILABLE")

    # News
    lines.append("  [NEWS & SENTIMENT]")
    if report.news:
        lines.append(f"    Sentiment: {report.news.overall_sentiment or 'N/A'}")
        lines.append(f"    Summary:   {report.news.summary}")
    else:
        lines.append("    Status:    UNAVAILABLE")

    # Research
    lines.append("  [SEC & RESEARCH]")
    if report.research:
        lines.append(f"    Summary:   {report.research.summary}")
    else:
        lines.append("    Status:    UNAVAILABLE")

    # Risk
    lines.append("  [RISK ASSESSMENT]")
    if report.risk:
        lines.append(f"    Risk Level: {report.risk.overall_risk_level or 'N/A'}")
        lines.append(f"    Summary:    {report.risk.summary}")
    else:
        lines.append("    Status:     UNAVAILABLE")
    lines.append(divider_light)

    # Key Risks
    lines.append("4. KEY INVESTMENT RISKS")
    if report.important_risks:
        for risk in report.important_risks:
            lines.append(f"  ! {risk}")
    else:
        lines.append("  None specified.")
    lines.append(divider_light)

    # Evidence Sources
    lines.append("5. EVIDENCE SOURCES")
    if report.evidence_sources:
        for ev in report.evidence_sources:
            lines.append(f"  [{ev.reference_id}] ({_val(ev.specialist)}): {ev.detail}")
    else:
        lines.append("  None specified.")
    lines.append(divider_heavy)

    # Regulatory Disclaimer
    if include_disclaimer:
        lines.append("REGULATORY DISCLAIMER:")
        lines.append(report.disclaimer)
        lines.append(divider_heavy)

    return "\n".join(lines).strip() + "\n"


# ===========================================================================
# 3. STRUCTURED JSON & FRONTEND DICTIONARY (Phase 12.4.1)
# ===========================================================================


def format_report_json(report: FinalReport, indent: Optional[int] = 2) -> str:
    """Serialize FinalReport to clean, validated JSON string.

    Guarantees round-trip validity with FinalReport.model_validate_json().
    """
    return report.model_dump_json(indent=indent)


def format_report_frontend_dict(report: FinalReport) -> Dict[str, Any]:
    """Format FinalReport into a structured dictionary optimized for frontend UI.

    Organizes data for direct binding to React cards, badges, tabs,
    and accordions, including color indicators, specialist cards, and markdown.
    """
    stance_val = (
        _val(report.recommendation.stance).lower()
        if report.recommendation and report.recommendation.stance
        else "neutral"
    )

    badge_theme = {
        "favorable": {"badge": "FAVORABLE", "color": "green", "variant": "success"},
        "neutral": {"badge": "NEUTRAL", "color": "amber", "variant": "warning"},
        "cautious": {"badge": "CAUTIOUS", "color": "orange", "variant": "warning"},
        "unfavorable": {
            "badge": "UNFAVORABLE",
            "color": "red",
            "variant": "destructive",
        },
        "insufficient_evidence": {
            "badge": "INSUFFICIENT EVIDENCE",
            "color": "gray",
            "variant": "secondary",
        },
    }.get(
        stance_val,
        {"badge": stance_val.upper(), "color": "blue", "variant": "default"},
    )

    specialist_cards: Dict[str, Any] = {}
    for spec_name in ("technical", "fundamental", "news", "research", "risk"):
        status_enum = report.specialist_statuses.get(
            spec_name, SpecialistStatus.MISSING
        )
        sec_obj = getattr(report, spec_name, None)
        specialist_cards[spec_name] = {
            "specialist": spec_name,
            "status": _val(status_enum),
            "is_available": sec_obj is not None,
            "data": sec_obj.model_dump() if sec_obj is not None else None,
        }

    return {
        "meta": {
            "ticker": report.company.ticker,
            "company_name": report.company.name,
            "currency": report.company.currency,
            "sector": report.company.sector,
            "industry": report.company.industry,
            "generated_at": (
                report.generated_at.isoformat()
                if getattr(report, "generated_at", None)
                else datetime.now(timezone.utc).isoformat()
            ),
            "data_completeness_ratio": (
                report.overall_assessment.data_completeness_ratio
                if report.overall_assessment
                and hasattr(report.overall_assessment, "data_completeness_ratio")
                else getattr(report, "data_completeness_ratio", None)
            ),
            "is_complete": report.is_complete and not _is_report_partial(report),
            "is_partial": _is_report_partial(report),
        },
        "investor_context": {
            "target_profile": (
                _val(getattr(report.investor_profile, "investment_goal", None))
                or _val(getattr(report.investor_profile, "target_profile", None))
                if report.investor_profile
                else None
            ),
            "investment_goal": (
                _val(getattr(report.investor_profile, "investment_goal", None))
                if report.investor_profile
                else None
            ),
            "risk_tolerance": (
                _val(getattr(report.investor_profile, "risk_tolerance", None))
                if report.investor_profile
                else None
            ),
            "horizon": (
                report.horizon
                or (
                    report.investor_profile.time_horizon
                    if report.investor_profile
                    else None
                )
            ),
            "capital_amount": (
                report.capital.amount
                if report.capital
                else (
                    getattr(report.investor_profile, "capital_amount", None)
                    if report.investor_profile
                    else None
                )
            ),
            "capital_formatted": (report.capital.formatted if report.capital else None),
        },
        "recommendation": {
            "stance": stance_val,
            "badge": badge_theme["badge"],
            "badge_color": badge_theme["color"],
            "ui_variant": badge_theme["variant"],
            "rationale": (
                report.recommendation.rationale if report.recommendation else None
            ),
            "profile_alignment": (
                report.recommendation.profile_alignment
                if report.recommendation
                else None
            ),
            "monitoring_points": (
                list(getattr(report.recommendation, "monitoring_points", []))
                if report.recommendation
                else []
            ),
            "time_horizon_suitability": (
                getattr(report.recommendation, "time_horizon_suitability", None)
                if report.recommendation
                else None
            ),
            "risk_tolerance_suitability": (
                getattr(report.recommendation, "risk_tolerance_suitability", None)
                if report.recommendation
                else None
            ),
        },
        "executive_summary": {
            "overall_assessment": (
                report.overall_assessment.synthesis
                if hasattr(report.overall_assessment, "synthesis")
                else str(report.overall_assessment or "")
            ),
            "cross_specialist_synthesis": (
                [
                    obs.observation
                    for obs in getattr(
                        report.overall_assessment, "cross_specialist_observations", []
                    )
                ]
                if hasattr(report.overall_assessment, "cross_specialist_observations")
                else getattr(report, "cross_specialist_synthesis", None)
            ),
        },
        "key_reasons": list(report.key_reasons),
        "important_risks": list(report.important_risks),
        "specialist_breakdowns": specialist_cards,
        "evidence_sources": [ev.model_dump() for ev in report.evidence_sources],
        "disclaimer": report.disclaimer,
        "rendered_markdown": format_report_markdown(report),
    }


def format_report_frontend_json(report: FinalReport, indent: Optional[int] = 2) -> str:
    """Serialize the enriched frontend dictionary to a JSON string."""
    data = format_report_frontend_dict(report)
    return json.dumps(data, indent=indent, default=str)
