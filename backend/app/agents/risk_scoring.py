"""Deterministic quantitative risk scoring engine for the Risk Analyst (Phase 10.3.3).

Implements objective, deterministic baseline risk scoring strictly from genuine
numerical metrics available from upstream specialists.

Principles enforced:
- Only genuine numeric fields (e.g. RSI, technical_score) produce quantitative scores.
- Qualitative LLM ratings ("strong", "weak", "uptrend", "negative") are NOT converted
  to arbitrary numbers.
- List length heuristics (counting bullet points) are NOT used to determine risk.
- Absence of quantitative metrics produces insufficient_data=True, score=None,
  level=None, never an assumed default moderate-risk score.
- Investor profile context may adjust an existing quantitative score, but cannot
  manufacture a score when quantitative evidence is absent.
"""

import math
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agents.risk_schema import (
    RiskAnalystInput,
    RiskSeverity,
)


class DeterministicRiskScore(BaseModel):
    """Deterministic quantitative risk assessment baseline."""

    model_config = ConfigDict(extra="ignore")

    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Overall quantitative risk score (0.0=lowest risk, 1.0=critical). "
            "None if quantitative data is insufficient."
        ),
    )
    level: Optional[RiskSeverity] = Field(
        default=None,
        description=(
            "Composite risk severity mapped from quantitative score. "
            "None if quantitative data is insufficient."
        ),
    )
    rsi_risk: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Risk component derived from Wilder RSI momentum regime.",
    )
    technical_score_risk: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Risk component derived from composite technical score.",
    )
    investor_penalty_multiplier: float = Field(
        default=1.0,
        description="Risk adjustment multiplier based on investor constraints.",
    )
    explanation: str = Field(
        default="",
        description="Deterministic narrative explaining score components.",
    )
    insufficient_data: bool = Field(
        default=False,
        description="True if upstream data lacked genuine quantitative metrics.",
    )
    component_breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed dictionary of active numeric inputs and weights.",
    )


def score_rsi(rsi: Optional[float]) -> Optional[float]:
    """Compute risk score from Wilder RSI indicator reading.

    RSI is bounded [0.0, 100.0]. Extremes indicate heightened momentum exhaustion
    or reversal volatility risk:
    - 40 <= RSI <= 60: Stable / low momentum risk (0.20)
    - 30 <= RSI < 40 or 60 < RSI <= 70: Moderate momentum extension (0.45)
    - 20 <= RSI < 30 or 70 < RSI <= 80: High reversal / overbought / oversold (0.70)
    - RSI < 20 or RSI > 80: Critical momentum extreme (0.85)

    Args:
        rsi: Optional float RSI value from TechnicalMetrics / TechnicalRiskSignals.

    Returns:
        Optional[float]: Risk score in [0.0, 1.0], or None if RSI is not provided.
    """
    if rsi is None or not isinstance(rsi, (int, float)):
        return None

    if math.isnan(rsi) or math.isinf(rsi) or rsi < 0.0 or rsi > 100.0:
        return None

    if 40.0 <= rsi <= 60.0:
        return 0.20
    elif (30.0 <= rsi < 40.0) or (60.0 < rsi <= 70.0):
        return 0.45
    elif (20.0 <= rsi < 30.0) or (70.0 < rsi <= 80.0):
        return 0.70
    else:
        return 0.85


def score_technical_score(technical_score: Optional[float]) -> Optional[float]:
    """Compute risk score from Phase 7.2 rule-based composite technical score.

    Technical score is bounded [0.0, 100.0] where 100 indicates strongest posture
    and 0 indicates severe breakdown. The risk score directly mirrors this:
    risk = (100.0 - technical_score) / 100.0.

    Args:
        technical_score: Optional float from TechnicalAnalysisOutput.

    Returns:
        Optional[float]: Risk score in [0.0, 1.0], or None if score is not provided.
    """
    if technical_score is None or not isinstance(technical_score, (int, float)):
        return None

    if (
        math.isnan(technical_score)
        or math.isinf(technical_score)
        or technical_score < 0.0
        or technical_score > 100.0
    ):
        return None

    return round((100.0 - technical_score) / 100.0, 3)


def calculate_investor_multiplier(
    input_data: RiskAnalystInput,
    base_score: Optional[float],
) -> float:
    """Calculate profile risk multiplier based on user constraints.

    Args:
        input_data: RiskAnalystInput containing optional investor profile.
        base_score: Quantitative base risk score, or None.

    Returns:
        float: Multiplier (defaults to 1.0 if no profile or no base score).
    """
    if (
        not input_data.investor_profile
        or base_score is None
        or math.isnan(base_score)
        or math.isinf(base_score)
    ):
        return 1.0

    prof = input_data.investor_profile
    multiplier = 1.0

    # Risk tolerance adjustment
    if prof.risk_tolerance:
        tol = prof.risk_tolerance.lower()
        if "conservative" in tol or "low" in tol:
            multiplier *= 1.25  # Lower capacity for loss amplifies risk
        elif "aggressive" in tol or "high" in tol:
            multiplier *= 0.80  # High capacity dampens risk perception
        elif "moderate" in tol or "medium" in tol:
            multiplier *= 1.00

    # Time horizon adjustment
    if prof.time_horizon:
        horiz = prof.time_horizon.lower()
        is_short = any(w in horiz for w in ["short", "month", "week", "day", "1 year"])
        is_long = any(w in horiz for w in ["long", "5 year", "10 year", "decade"])

        if is_short and base_score >= 0.50:
            multiplier *= 1.15
        elif is_long:
            multiplier *= 0.90

    return round(multiplier, 2)


def map_score_to_severity(score: float) -> RiskSeverity:
    """Map quantitative composite risk score to standard RiskSeverity enum."""
    if score < 0.30:
        return RiskSeverity.LOW
    elif score < 0.60:
        return RiskSeverity.MODERATE
    elif score < 0.85:
        return RiskSeverity.HIGH
    else:
        return RiskSeverity.CRITICAL


def calculate_deterministic_risk_score(
    input_data: RiskAnalystInput,
) -> DeterministicRiskScore:
    """Calculate quantitative baseline strictly from genuine numeric indicators.

    Evaluates genuine numeric metrics available in input_data:
    - TechnicalRiskSignals.rsi (float)
    - TechnicalRiskSignals.technical_score (float)

    Does NOT convert qualitative strings ("strong", "weak", "uptrend") into numbers,
    does NOT use bullet-point count heuristics, and does NOT default to 0.50.

    If genuine quantitative metrics are unavailable, returns insufficient_data=True
    with score=None and level=None.

    Args:
        input_data: Validated RiskAnalystInput.

    Returns:
        DeterministicRiskScore: Computed quantitative score or insufficient_data state.
    """
    if input_data.is_empty or not input_data.technical_signals:
        return DeterministicRiskScore(
            insufficient_data=True,
            explanation=(
                "No genuine quantitative metrics (RSI, technical score) available "
                "for deterministic baseline scoring."
            ),
        )

    tech = input_data.technical_signals
    rsi_risk = score_rsi(tech.rsi)
    tech_score_risk = score_technical_score(tech.technical_score)

    active_components: Dict[str, tuple[float, float]] = {}
    if rsi_risk is not None:
        active_components["rsi"] = (0.40, rsi_risk)
    if tech_score_risk is not None:
        active_components["technical_score"] = (0.60, tech_score_risk)

    if not active_components:
        return DeterministicRiskScore(
            insufficient_data=True,
            explanation=(
                "Technical signals were provided but contained no genuine numeric "
                "metrics (RSI and technical score are both absent)."
            ),
        )

    total_weight = sum(w for w, _ in active_components.values())
    weighted_sum = sum(w * s for w, s in active_components.values())
    base_score = weighted_sum / total_weight

    inv_multiplier = calculate_investor_multiplier(input_data, base_score)
    final_score = round(min(1.0, max(0.0, base_score * inv_multiplier)), 3)
    severity = map_score_to_severity(final_score)

    comp_parts = []
    if rsi_risk is not None:
        comp_parts.append(f"RSI risk: {rsi_risk:.2f}")
    if tech_score_risk is not None:
        comp_parts.append(f"technical score risk: {tech_score_risk:.2f}")

    comp_str = ", ".join(comp_parts)
    explanation = (
        f"Deterministic quantitative baseline score {final_score:.2f} "
        f"({severity.value}) derived from [{comp_str}] with investor "
        f"multiplier {inv_multiplier:.2f}x."
    )

    return DeterministicRiskScore(
        score=final_score,
        level=severity,
        rsi_risk=rsi_risk,
        technical_score_risk=tech_score_risk,
        investor_penalty_multiplier=inv_multiplier,
        explanation=explanation,
        insufficient_data=False,
        component_breakdown={
            "base_score": round(base_score, 3),
            "rsi_risk": rsi_risk,
            "technical_score_risk": tech_score_risk,
            "investor_multiplier": inv_multiplier,
            "active_metrics": list(active_components.keys()),
        },
    )
