"""Normalized domain and structured output schemas for Technical Analyst Agent.

Phase 7.3 defines the input and structured output models for interpreting
pre-calculated deterministic TechnicalMetrics:
- TechnicalAnalystInput: Carries TechnicalMetrics, investor constraints, and
  CIO guidance.
- TechnicalIndicatorsSummary: Clean representation of deterministic indicators.
- SupportResistanceSummary: Support and resistance levels.
- TechnicalInterpretation: Structured qualitative narrative sections.
- TechnicalAnalysisOutput: Comprehensive structured technical assessment.
- TechnicalAnalysisValidationError: Typed exception for grounding failures.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.technical_metrics import (
    MACDMetrics,
    MovingAverageMetrics,
    RSIMetrics,
    TechnicalMetrics,
    TechnicalScoreBreakdown,
    TrendDirection,
    VolumeMetrics,
)


class TechnicalAnalysisValidationError(ValueError):
    """Raised when Technical Analyst output violates grounding or safety rules."""

    pass


class TechnicalIndicatorsSummary(BaseModel):
    """Summary container of deterministic indicator values from Phase 7.2."""

    model_config = ConfigDict(extra="ignore")

    latest_close: float = Field(
        ..., description="Latest closing price from the market data."
    )
    moving_averages: MovingAverageMetrics = Field(
        default_factory=MovingAverageMetrics,
        description="Simple and exponential moving average values.",
    )
    rsi: RSIMetrics = Field(
        default_factory=RSIMetrics,
        description="Wilder RSI momentum metrics.",
    )
    macd: MACDMetrics = Field(
        default_factory=MACDMetrics,
        description="MACD fast line, signal line, and histogram metrics.",
    )
    volume: VolumeMetrics = Field(
        default_factory=VolumeMetrics,
        description="Latest volume, 20-day average volume, and volume ratio.",
    )


class SupportResistanceSummary(BaseModel):
    """Summary of deterministic support and resistance price levels."""

    model_config = ConfigDict(extra="ignore")

    primary_support: Optional[float] = Field(
        default=None, description="Nearest key support level below current price."
    )
    primary_resistance: Optional[float] = Field(
        default=None,
        description="Nearest key resistance level above current price.",
    )
    support_levels: List[float] = Field(
        default_factory=list,
        description="All detected support levels, sorted descending (nearest first).",
    )
    resistance_levels: List[float] = Field(
        default_factory=list,
        description="All detected resistance levels, sorted ascending (nearest first).",
    )


class TechnicalInterpretation(BaseModel):
    """Qualitative narrative interpretation grounded strictly in supplied metrics."""

    model_config = ConfigDict(extra="ignore")

    overall_summary: str = Field(
        ...,
        description=(
            "Synthesized technical narrative explaining current technical posture."
        ),
    )
    trend_analysis: str = Field(
        ...,
        description=(
            "Explanation of overall trend direction based on price action and MAs."
        ),
    )
    moving_averages_analysis: str = Field(
        ...,
        description="Interpretation of moving average stacking and alignment.",
    )
    momentum_analysis: str = Field(
        ...,
        description="Interpretation of RSI regime and MACD dynamics.",
    )
    volume_analysis: str = Field(
        ...,
        description="Interpretation of volume confirmation and participation.",
    )
    support_resistance_analysis: str = Field(
        ...,
        description=(
            "Context on proximity to nearest key support and resistance boundaries."
        ),
    )


class TechnicalAnalystInput(BaseModel):
    """Input payload for the Technical Analyst Agent."""

    model_config = ConfigDict(extra="ignore")

    metrics: TechnicalMetrics = Field(
        ...,
        description="Deterministic technical metrics computed in Phase 7.2.",
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


class TechnicalAnalysisOutput(BaseModel):
    """Structured analytical assessment produced by Technical Analyst Agent."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Normalized uppercase ticker symbol.")
    trend: Optional[TrendDirection] = Field(
        default=None,
        description="Deterministic trend ('uptrend', 'downtrend', 'sideways').",
    )
    indicators_summary: TechnicalIndicatorsSummary = Field(
        ...,
        description="Pre-calculated deterministic indicator values from Phase 7.2.",
    )
    support_resistance: SupportResistanceSummary = Field(
        ...,
        description="Deterministic support and resistance levels from Phase 7.2.",
    )
    technical_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Deterministic composite technical score from Phase 7.2.",
    )
    score_breakdown: Optional[TechnicalScoreBreakdown] = Field(
        default=None,
        description="Deterministic score component breakdown from Phase 7.2.",
    )
    interpretation: TechnicalInterpretation = Field(
        ...,
        description="Narrative interpretation explaining the deterministic metrics.",
    )
    evidence: List[str] = Field(
        ...,
        min_length=1,
        description="Key factual observations grounded strictly in supplied metrics.",
    )
    risks: List[str] = Field(
        ...,
        min_length=1,
        description="Key technical risk factors identified from indicators.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Interpretation confidence reflecting data completeness and indicator "
            "consistency (NOT an investment-return probability)."
        ),
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return v.strip().upper()
