"""Normalized domain models for historical market data and price series.

Phase 7.1 defines strongly typed Pydantic models for OHLCV candles and historical
market data containers.

Boundary Rules:
- Captures raw/underlying historical price and volume series from data providers.
- Strictly isolated from pandas/yfinance objects.
- Preserves actual numeric values without artificial rounding.
- Preserves missing optional adjusted-close values as None.
- Preserves market gaps (weekends/holidays); NEVER fabricates synthetic candles.
- Rejects invalid OHLC relationships, non-positive prices, negative volume, NaN/Inf.

"""

import math
from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_finite_positive(val: float, field_name: str) -> float:
    if val is None:
        raise ValueError(f"{field_name} cannot be None.")
    try:
        f = float(val)
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid float number, got {val}.")
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{field_name} cannot be NaN or Inf.")
    if f <= 0:
        raise ValueError(f"{field_name} must be strictly positive (> 0), got {f}.")
    return f


def _validate_finite_non_negative(val: float, field_name: str) -> float:
    if val is None:
        raise ValueError(f"{field_name} cannot be None.")
    try:
        f = float(val)
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid float number, got {val}.")
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{field_name} cannot be NaN or Inf.")
    if f < 0:
        raise ValueError(f"{field_name} must be non-negative (>= 0), got {f}.")
    return f


class OHLCVCandle(BaseModel):
    """Normalized domain model representing a single market data candle (OHLCV).

    Attributes:
        timestamp: Timezone-aware UTC timestamp of the candle.
        open: Opening price for the period (strictly positive).
        high: Highest price for the period (strictly positive).
        low: Lowest price for the period (strictly positive).
        close: Closing price for the period (strictly positive).
        volume: Traded volume during the period (non-negative).
        adjusted_close: Optional adjusted close (> 0, if available).
    """

    model_config = ConfigDict(extra="ignore")

    timestamp: datetime = Field(..., description="UTC timestamp of the candle.")
    open: float = Field(..., description="Opening price (> 0).")
    high: float = Field(..., description="Highest price (> 0).")
    low: float = Field(..., description="Lowest price (> 0).")
    close: float = Field(..., description="Closing price (> 0).")
    volume: float = Field(..., description="Traded volume (>= 0).")
    adjusted_close: Optional[float] = Field(
        default=None, description="Adjusted closing price (> 0), if available."
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def normalize_timestamp(cls, v: Any) -> datetime:
        """Ensure timestamp is a timezone-aware UTC datetime."""
        if v is None:
            raise ValueError("Timestamp cannot be None.")
        if isinstance(v, str):
            from datetime import date

            try:
                dt = datetime.fromisoformat(v)
            except ValueError:
                d = date.fromisoformat(v)
                dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        if hasattr(v, "to_pydatetime"):
            pydt = v.to_pydatetime()
            if pydt.tzinfo is None:
                return pydt.replace(tzinfo=timezone.utc)
            return pydt.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if type(v) is not datetime:
                dt = datetime.fromtimestamp(v.timestamp(), tz=timezone.utc)
                return dt
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        raise ValueError(f"Unsupported timestamp format: {type(v)} ({v})")

    @field_validator("open")
    @classmethod
    def validate_open(cls, v: float) -> float:
        return _validate_finite_positive(v, "open")

    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float) -> float:
        return _validate_finite_positive(v, "high")

    @field_validator("low")
    @classmethod
    def validate_low(cls, v: float) -> float:
        return _validate_finite_positive(v, "low")

    @field_validator("close")
    @classmethod
    def validate_close(cls, v: float) -> float:
        return _validate_finite_positive(v, "close")

    @field_validator("volume")
    @classmethod
    def validate_volume(cls, v: float) -> float:
        return _validate_finite_non_negative(v, "volume")

    @field_validator("adjusted_close")
    @classmethod
    def validate_adjusted_close(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        return _validate_finite_positive(v, "adjusted_close")

    @model_validator(mode="after")
    def validate_ohlc_integrity(self) -> "OHLCVCandle":
        """Verify structural relationships between Open, High, Low, and Close."""
        if self.high < self.low:
            raise ValueError(
                f"High price ({self.high}) cannot be less than low price ({self.low})."
            )
        if self.high < self.open:
            raise ValueError(
                f"High price ({self.high}) cannot be less than open ({self.open})."
            )
        if self.high < self.close:
            raise ValueError(
                f"High price ({self.high}) cannot be less than close ({self.close})."
            )
        if self.low > self.open:
            raise ValueError(
                f"Low price ({self.low}) cannot be greater than open ({self.open})."
            )
        if self.low > self.close:
            raise ValueError(
                f"Low price ({self.low}) cannot be greater than close ({self.close})."
            )

        return self


class HistoricalMarketData(BaseModel):
    """Normalized domain container for historical market price and volume data.

    Attributes:
        ticker: Uppercase normalized ticker symbol.
        candles: Chronologically ordered list of validated OHLCVCandles.
        period: Requested lookback period string (e.g. '1y', '6mo', '1mo').
        interval: Requested candle interval string (e.g. '1d', '1wk').
        provider: Source provider identifier (default: 'yahoo').
        retrieved_at: UTC timestamp when data was fetched and normalized.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(..., description="Normalized ticker symbol.")
    candles: List[OHLCVCandle] = Field(
        default_factory=list,
        description="Chronologically sorted OHLCV candle series.",
    )
    period: Optional[str] = Field(
        default=None, description="Requested lookback period string."
    )
    interval: Optional[str] = Field(
        default=None, description="Requested candle interval string."
    )
    provider: str = Field(
        default="yahoo", description="Market data source provider identifier."
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of data retrieval.",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("Ticker symbol cannot be empty or whitespace.")
        return v.strip().upper()

    @model_validator(mode="after")
    def validate_series_integrity(self) -> "HistoricalMarketData":
        """Verify candles are strictly chronological with no duplicate timestamps."""
        if not self.candles:
            return self

        seen_timestamps = set()
        prev_ts: Optional[datetime] = None

        for idx, candle in enumerate(self.candles):
            ts = candle.timestamp
            if ts in seen_timestamps:
                raise ValueError(
                    f"Duplicate timestamp detected at index {idx}: {ts.isoformat()}."
                )
            seen_timestamps.add(ts)

            if prev_ts is not None and ts <= prev_ts:
                raise ValueError(
                    f"Candles not chronological at index {idx}: "
                    f"{ts.isoformat()} <= {prev_ts.isoformat()}."
                )
            prev_ts = ts

        return self
