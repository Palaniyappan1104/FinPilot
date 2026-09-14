"""Prompt design and instructions for Report Generator Agent (Phase 12.2).

Phase 12.2 establishes:
- Report Generator system prompt embedding:
    * Role establishment: Final investment report generator for decision support.
    * Synthesis principles: Explainable narrative, true cross-domain synthesis.
    * Factual grounding: Use ONLY supplied specialist outputs and evidence.
    * Grounded recommendation: Stance must be grounded in specialist findings.
    * Explicit handling of partial inputs: Note missing, failed, or empty specialists.
    * Insufficient evidence disclosure: Communicate limitations honestly.
    * Safety & non-advisory boundaries: No buy/sell imperatives, no price targets,
      no guaranteed returns, mandatory disclaimer.
- Structured prompt formatting utility (`format_report_generator_prompt`) organizing
  aggregated specialist analysis into clean, injection-resistant context blocks.
"""

from typing import Any, List

from app.agents.aggregator_schema import (
    UnifiedSpecialistAnalysis,
)


def _val(x: Any) -> str:
    """Safely extract string representation from enum or object."""
    if x is None:
        return ""
    if hasattr(x, "value"):
        return str(x.value)
    return str(x)


# ===========================================================================
# REPORT GENERATOR SYSTEM PROMPT (Phase 12.2)
# ===========================================================================

REPORT_GENERATOR_SYSTEM_PROMPT = (
    "You are FinPilot's Report Generator Agent.\n"
    "Your responsibility is to synthesize the completed, aggregated specialist "
    "analysis\n"
    "into a comprehensive, structured, and explainable final investment report for "
    "decision support.\n\n"
    "CORE PRINCIPLES & GENERATION BOUNDARIES (PHASE 12.2):\n\n"
    "1. EXPLAINABLE SYNTHESIS (NOT SIMPLE CONCATENATION):\n"
    "   - Formulate a cohesive narrative that synthesizes the entire analytical "
    "picture.\n"
    "   - Explain the 'why' behind conclusions, citing specific specialist evidence.\n"
    "   - Connect findings across technical, fundamental, news, research, and risk "
    "domains.\n"
    "   - Do NOT just copy-paste specialist blocks; synthesize them into key reasons "
    "and risks.\n\n"
    "2. GROUNDED RECOMMENDATION & ANALYTICAL STANCE:\n"
    "   - The recommendation stance MUST be one of:\n"
    "     * 'favorable': Strong positive alignment across fundamentals, technicals, "
    "and acceptable risk.\n"
    "     * 'cautious': Conflicting specialist signals, elevated risk factors, or "
    "notable data gaps.\n"
    "     * 'neutral': Balanced signals where positive and negative factors offset.\n"
    "     * 'unfavorable': Deteriorating fundamentals, persistent downtrend, and high "
    "financial risk.\n"
    "     * 'insufficient_evidence': Missing or failed core specialists preventing a "
    "reliable stance.\n"
    "   - The rationale MUST cite specific factual reasons tied to supplied evidence.\n"
    "   - Explicitly articulate how the analysis aligns with the investor profile "
    "(goals, horizon, risk tolerance).\n"
    "   - When signal conflicts exist, you MUST adopt a cautious stance or explicitly "
    "address the tension.\n"
    "   - If evidence is insufficient, you MUST assign 'insufficient_evidence' and "
    "disclose data limits.\n\n"
    "3. STRICT FACTUAL GROUNDING & ZERO HALLUCINATION:\n"
    "   - Use ONLY the facts, figures, metrics, and evidence items supplied in "
    "context.\n"
    "   - NEVER calculate, invent, or hallucinate prices, financial ratios, valuation "
    "multiples, technical indicators, news events, filings, or evidence references.\n"
    "   - Retain explicit source attribution for every major observation.\n\n"
    "4. PARTIAL COVERAGE & MISSING SPECIALIST ACKNOWLEDGMENT:\n"
    "   - Explicitly acknowledge missing, omitted, or failed specialist domains.\n"
    "   - Clearly distinguish between available evidence, missing data, and "
    "failed analysis.\n"
    "   - Never assume or pretend data exists for an unanalyzed specialist domain.\n\n"
    "5. REGULATORY NON-ADVISORY COMPLIANCE:\n"
    "   - FinPilot is an automated financial research and decision-support engine, NOT "
    "a registered financial advisor, broker-dealer, or fiduciary.\n"
    "   - NEVER issue definitive imperative buy/sell/hold orders (e.g. 'buy this "
    "stock').\n"
    "   - NEVER provide price targets (e.g. 'target price of $250').\n"
    "   - NEVER claim guaranteed returns, risk-free profits, or certainty of "
    "performance.\n"
    "   - Frame all recommendations as an objective decision-support stance.\n"
    "   - Always preserve the mandatory non-advisory disclaimer."
)


# ===========================================================================
# PROMPT FORMATTER (Phase 12.2)
# ===========================================================================


def format_report_generator_prompt(analysis: UnifiedSpecialistAnalysis) -> str:
    """Format a structured, injection-resistant prompt for the Report Generator.

    Organizes the aggregated specialist analysis, evidence items, signal conflicts,
    areas of agreement, and investor parameters into clearly delimited context sections.

    Args:
        analysis: Validated UnifiedSpecialistAnalysis from Phase 11.

    Returns:
        str: Grounded LLM prompt string.
    """
    lines: List[str] = [
        REPORT_GENERATOR_SYSTEM_PROMPT,
        "",
        "===================================================================",
        "AGGREGATED SPECIALIST ANALYSIS CONTEXT (READ-ONLY GROUNDING DATA)",
        "===================================================================",
        f"TARGET TICKER: {analysis.ticker}",
        f"COMPANY NAME: {analysis.target_company or 'Not specified'}",
        f"DATA COMPLETENESS RATIO: {analysis.data_completeness_ratio:.1%}",
        f"OVERALL CONFIDENCE: {analysis.confidence:.2f}",
        f"INSUFFICIENT EVIDENCE FLAG: {analysis.insufficient_evidence}",
    ]

    if analysis.insufficient_evidence_reason:
        lines.append(
            f"INSUFFICIENT EVIDENCE REASON: {analysis.insufficient_evidence_reason}"
        )

    # Investor Profile Context
    lines.extend(
        [
            "",
            "--- INVESTOR PROFILE & CONSTRAINTS ---",
        ]
    )
    if analysis.investor_profile:
        p = analysis.investor_profile
        lines.append(f"Investment Goal: {p.investment_goal or 'Growth / Balanced'}")
        lines.append(f"Time Horizon: {p.time_horizon or '3-5 years'}")
        lines.append(f"Risk Tolerance: {p.risk_tolerance or 'Moderate'}")
        if p.capital_amount is not None:
            lines.append(f"Allocated Capital: ${p.capital_amount:,.2f}")
    else:
        lines.append("Profile: Standard baseline (Moderate risk, 3-5 years horizon).")

    # Specialist Statuses
    lines.extend(
        [
            "",
            "--- SPECIALIST COVERAGE STATUSES ---",
        ]
    )
    for spec, status in analysis.specialist_statuses.items():
        err = analysis.specialist_errors.get(spec)
        err_str = f" (Error: {err})" if err else ""
        lines.append(f"Specialist '{spec}': status={_val(status).upper()}{err_str}")

    if analysis.missing_specialists:
        lines.append(f"Missing Specialists: {', '.join(analysis.missing_specialists)}")
    if analysis.failed_specialists:
        lines.append(f"Failed Specialists: {', '.join(analysis.failed_specialists)}")

    # Technical Analysis Section
    lines.extend(
        [
            "",
            "--- TECHNICAL ANALYST FINDINGS ---",
        ]
    )
    if analysis.technical_assessment:
        t = analysis.technical_assessment
        lines.append(f"Trend: {_val(t.trend)}")
        lines.append(f"Technical Score: {t.technical_score or 'N/A'}")
        if t.interpretation:
            lines.append(f"Summary: {t.interpretation.overall_summary}")
            lines.append(f"Momentum: {t.interpretation.momentum_analysis}")
        if t.support_resistance:
            lines.append(f"Support Levels: {t.support_resistance.support_levels}")
            lines.append(f"Resistance Levels: {t.support_resistance.resistance_levels}")
        lines.append(f"Technical Confidence: {t.confidence:.2f}")
    else:
        lines.append("Technical data: NOT AVAILABLE.")

    # Fundamental Analysis Section
    lines.extend(
        [
            "",
            "--- FUNDAMENTAL ANALYST FINDINGS ---",
        ]
    )
    if analysis.fundamental_assessment:
        f = analysis.fundamental_assessment
        lines.append(f"Overall Assessment: {_val(f.overall_assessment)}")
        lines.append(f"Summary: {f.overall_summary}")
        if f.financial_health:
            lines.append(
                f"Financial Health: {_val(f.financial_health.rating)} - "
                f"{f.financial_health.explanation}"
            )
        if f.profitability_assessment:
            lines.append(
                f"Profitability: {_val(f.profitability_assessment.rating)} - "
                f"{f.profitability_assessment.explanation}"
            )
        if f.valuation_assessment:
            lines.append(
                f"Valuation: {_val(f.valuation_assessment.rating)} - "
                f"{f.valuation_assessment.explanation}"
            )
        if f.growth_assessment:
            lines.append(
                f"Growth: {_val(f.growth_assessment.rating)} - "
                f"{f.growth_assessment.explanation}"
            )
        if f.cash_flow_assessment:
            lines.append(
                f"Cash Flow: {_val(f.cash_flow_assessment.rating)} - "
                f"{f.cash_flow_assessment.explanation}"
            )
        if f.key_strengths:
            lines.append(f"Key Strengths: {'; '.join(f.key_strengths)}")
        if f.key_weaknesses:
            lines.append(f"Key Weaknesses: {'; '.join(f.key_weaknesses)}")
        lines.append(f"Fundamental Confidence: {f.confidence:.2f}")
    else:
        lines.append("Fundamental data: NOT AVAILABLE.")

    # News Analysis Section
    lines.extend(
        [
            "",
            "--- NEWS & SENTIMENT FINDINGS ---",
        ]
    )
    if analysis.news_assessment:
        n = analysis.news_assessment
        lines.append(f"Overall Sentiment: {_val(n.overall_sentiment)}")
        lines.append(f"Summary: {getattr(n, 'summary', '')}")
        if n.recent_news:
            headlines = [item.headline for item in n.recent_news[:3] if item.headline]
            lines.append(f"Recent Headlines: {'; '.join(headlines)}")
        lines.append(f"News Confidence: {n.confidence:.2f}")
    else:
        lines.append("News data: NOT AVAILABLE.")

    # Research Analysis Section
    lines.extend(
        [
            "",
            "--- DOCUMENT & SEC RESEARCH FINDINGS ---",
        ]
    )
    if analysis.research_assessment:
        r = analysis.research_assessment
        lines.append(
            f"Research Answer: {getattr(r, 'answer', getattr(r, 'summary', ''))}"
        )
        if hasattr(r, "key_findings") and r.key_findings:
            findings = [getattr(f, "claim", str(f)) for f in r.key_findings[:3]]
            lines.append(f"Key Findings: {'; '.join(findings)}")
        lines.append(f"Research Confidence: {r.confidence:.2f}")
    else:
        lines.append("Research document data: NOT AVAILABLE.")

    # Risk Analysis Section
    lines.extend(
        [
            "",
            "--- RISK ASSESSMENT FINDINGS ---",
        ]
    )
    if analysis.risk_assessment:
        rk = analysis.risk_assessment
        lines.append(f"Overall Risk Level: {_val(rk.overall_risk_level)}")
        risk_sum = getattr(rk, "summary", "") or getattr(rk, "overall_summary", "")
        lines.append(f"Summary: {risk_sum}")
        all_rf = (
            getattr(rk, "market_risks", [])
            + getattr(rk, "company_risks", [])
            + getattr(rk, "financial_risks", [])
        )
        if all_rf:
            risk_names = [getattr(f, "name", str(f)) for f in all_rf[:5]]
            lines.append(f"Top Risk Factors: {'; '.join(risk_names)}")
        lines.append(f"Risk Confidence: {rk.confidence:.2f}")
    else:
        lines.append("Risk assessment data: NOT AVAILABLE.")

    # Phase 11 Aggregation Synthesis Findings
    lines.extend(
        [
            "",
            "--- PHASE 11 SYNTHESIS & CROSS-REFERENCING ---",
            f"Aggregated Overall Synthesis: {analysis.overall_synthesis}",
        ]
    )

    if analysis.areas_of_agreement:
        lines.append("Areas of Agreement (Consensus):")
        for idx, a in enumerate(analysis.areas_of_agreement, 1):
            specs = ", ".join(a.supporting_specialists)
            lines.append(f"  {idx}. [{a.topic}] ({specs}): {a.summary}")

    if analysis.signal_conflicts:
        lines.append("Signal Conflicts & Analytical Tensions:")
        for idx, c in enumerate(analysis.signal_conflicts, 1):
            specs = ", ".join(c.involved_specialists)
            lines.append(f"  {idx}. [{c.topic}] ({specs}): {c.description}")

    if analysis.cross_specialist_observations:
        lines.append("Cross-Specialist Observations:")
        for idx, o in enumerate(analysis.cross_specialist_observations, 1):
            specs = ", ".join(o.connected_specialists)
            lines.append(f"  {idx}. ({specs}): {o.observation}")

    # Grounding Evidence Items
    lines.extend(
        [
            "",
            "--- ATTRIBUTED EVIDENCE ITEMS (FOR CITATION REFERENCE) ---",
        ]
    )
    if analysis.aggregated_evidence:
        for idx, ev in enumerate(analysis.aggregated_evidence[:15], 1):
            doc_info = f" (Doc: {ev.document_id})" if ev.document_id else ""
            lines.append(
                f"  [{ev.reference_id}] ({ev.specialist}): {ev.detail}{doc_info}"
            )
    else:
        lines.append("No specialist evidence items available.")

    # Instructions for Structured Output
    lines.extend(
        [
            "",
            "===================================================================",
            "GENERATION INSTRUCTIONS:",
            "===================================================================",
            "Produce a structured JSON report matching the schema with:",
            "1. 'recommendation_stance': One of 'favorable', 'cautious', 'neutral', "
            "'unfavorable', 'insufficient_evidence'.",
            "2. 'recommendation_rationale': Grounded narrative explaining the stance "
            "based strictly on specialist findings.",
            "3. 'profile_alignment': Explicit explanation of how findings align with "
            "the investor's goal, horizon, and risk tolerance.",
            "4. 'monitoring_points': List of 2-4 critical upcoming catalysts, earnings "
            "dates, or technical levels to track.",
            "5. 'key_reasons': List of 2-5 specific primary reasons supporting the "
            "stance, derived strictly from specialist data.",
            "6. 'important_risks': List of 2-5 critical risk factors the "
            "investor must monitor.",
            "7. 'executive_synthesis': Cohesive executive overview synthesizing "
            "REMEMBER: NEVER invent numbers or evidence references. "
            "NO buy/sell advice.",
        ]
    )

    return "\n".join(lines)
