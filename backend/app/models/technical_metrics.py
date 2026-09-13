"""Normalized domain models for calculated technical indicators and metrics.

Phase 7.2 defines strongly typed Pydantic models for technical analysis outputs:
- Moving averages (SMA 20/50/200, EMA 20/50/200, EMA 12/26).
- Momentum (RSI-14 with Wilder smoothing).
- MACD (MACD line, signal line, histogram).
- Volume analysis (latest volume, 20d average, volume ratio).
- Support and resistance levels (rolling swing extrema).
- Trend classification (uptrend, downtrend, sideways).
- Deterministic technical score (0.0 to 100.0) with component breakdown.

Boundary Rules:
- Preserves full floating-point precision without premature rounding.
- Insufficient historical observations result in None rather than fabricated values.
- Completely isolated from LLM agents, prompts, and recommendations.
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

TrendDirection = Literal["uptrend", "downtrend", "sideways"]


class MovingAverageMetrics(BaseModel):
    """Simple and exponential moving averages computed from historical close prices."""

    model_config = ConfigDict(extra="ignore")

    sma_20: Optional[float] = Field(
        default=None, description="20-period simple moving average."
    )
    sma_50: Optional[float] = Field(
        default=None, description="50-period simple moving average."
    )
    sma_200: Optional[float] = Field(
        default=None, description="200-period simple moving average."
    )
    ema_20: Optional[float] = Field(
        default=None, description="20-period exponential moving average."
    )
    ema_50: Optional[float] = Field(
        default=None, description="50-period exponential moving average."
    )
    ema_200: Optional[float] = Field(
        default=None, description="200-period exponential moving average."
    )
    ema_12: Optional[float] = Field(
        default=None, description="12-period EMA used in MACD."
    )
    ema_26: Optional[float] = Field(
        default=None, description="26-period EMA used in MACD."
    )


class RSIMetrics(BaseModel):
    """Relative Strength Index computed using Wilder's smoothing methodology."""

    model_config = ConfigDict(extra="ignore")

    rsi_14: Optional[float] = Field(
        default=None, description="14-period Wilder Relative Strength Index (0-100)."
    )
    period: int = Field(default=14, description="Lookback period for RSI computation.")


class MACDMetrics(BaseModel):
    """Moving Average Convergence Divergence indicators."""

    model_config = ConfigDict(extra="ignore")

    macd_line: Optional[float] = Field(
        default=None, description="MACD line (EMA12 - EMA26)."
    )
    signal_line: Optional[float] = Field(
        default=None, description="Signal line (9-period EMA of MACD line)."
    )
    histogram: Optional[float] = Field(
        default=None, description="MACD histogram (MACD line - Signal line)."
    )
    fast_period: int = Field(default=12, description="Fast EMA period.")
    slow_period: int = Field(default=26, description="Slow EMA period.")
    signal_period: int = Field(default=9, description="Signal EMA period.")


class VolumeMetrics(BaseModel):
    """Volume analysis metrics and rolling activity comparisons."""

    model_config = ConfigDict(extra="ignore")

    latest_volume: Optional[float] = Field(
        default=None, description="Trading volume of the latest candle."
    )
    average_volume_20d: Optional[float] = Field(
        default=None, description="20-period rolling average volume."
    )
    volume_ratio: Optional[float] = Field(
        default=None, description="Ratio of latest volume to 20-period average volume."
    )


class SupportResistanceMetrics(BaseModel):
    """Deterministic price support and resistance levels from rolling extrema."""

    model_config = ConfigDict(extra="ignore")

    primary_support: Optional[float] = Field(
        default=None, description="Nearest key support level below current price."
    )
    primary_resistance: Optional[float] = Field(
        default=None, description="Nearest key resistance level above current price."
    )
    support_levels: List[float] = Field(
        default_factory=list,
        description="Identified support levels, sorted descending (nearest first).",
    )
    resistance_levels: List[float] = Field(
        default_factory=list,
        description="Identified resistance levels, sorted ascending (nearest first).",
    )


class TechnicalScoreBreakdown(BaseModel):
    """Transparent component scores contributing to the overall technical score."""

    model_config = ConfigDict(extra="ignore")

    ma_alignment_score: Optional[float] = Field(
        default=None, description="Moving average trend alignment component (0-25)."
    )
    rsi_score: Optional[float] = Field(
        default=None, description="RSI momentum regime component (0-25)."
    )
    macd_score: Optional[float] = Field(
        default=None, description="MACD relationship and histogram component (0-25)."
    )
    volume_score: Optional[float] = Field(
        default=None, description="Volume confirmation component (0-25)."
    )


class TechnicalMetrics(BaseModel):
    """Container for calculated and standardized technical indicators for a security."""

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Normalized uppercase ticker symbol.")
    latest_close: float = Field(
        ..., description="Closing price of the most recent candle."
    )
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when technical metrics were computed.",
    )
    candle_count: int = Field(
        ..., description="Total count of historical candles evaluated."
    )

    moving_averages: MovingAverageMetrics = Field(
        default_factory=MovingAverageMetrics,
        description="Simple and exponential moving averages.",
    )
    rsi: RSIMetrics = Field(
        default_factory=RSIMetrics,
        description="Wilder RSI momentum metrics.",
    )
    macd: MACDMetrics = Field(
        default_factory=MACDMetrics,
        description="MACD trend-following momentum metrics.",
    )
    volume: VolumeMetrics = Field(
        default_factory=VolumeMetrics,
        description="Volume activity and expansion metrics.",
    )
    support_resistance: SupportResistanceMetrics = Field(
        default_factory=SupportResistanceMetrics,
        description="Deterministic support and resistance levels.",
    )
    trend: Optional[TrendDirection] = Field(
        default=None,
        description="Classified price trend ('uptrend', 'downtrend', 'sideways').",
    )
    technical_score: Optional[float] = Field(
        default=None,
        description="Composite technical score bounded between 0.0 and 100.0.",
    )
    score_breakdown: Optional[TechnicalScoreBreakdown] = Field(
        default=None,
        description="Detailed point breakdown of technical score components.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return v.strip().upper()
