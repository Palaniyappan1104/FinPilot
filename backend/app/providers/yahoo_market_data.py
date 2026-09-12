"""Yahoo Finance Market Data Provider implementation for FinPilot.

Phase 7.1 implements historical OHLCV data retrieval using yfinance:
- Retrieves complete historical OHLCV candles for requested periods and intervals.
- Normalizes data into FinPilot domain models (OHLCVCandle, HistoricalMarketData).
- Strictly prevents yfinance and pandas objects from leaking outside provider.
- Validates chronological order, OHLC integrity, positive prices, volume >= 0.
- Preserves actual values without arbitrary rounding or truncation.

- Rejects corrupted/malformed rows and duplicate timestamps via typed exceptions.
- Preserves genuine market gaps (weekends/holidays); NEVER fabricates synthetic bars.
"""

import math
from datetime import datetime, timezone
from typing import Any, List, Optional

import pandas as pd
import yfinance
from pydantic import ValidationError

from app.core.logging import get_logger
from app.models.market_data import HistoricalMarketData, OHLCVCandle
from app.providers.exceptions import (
    EmptyMarketDataError,
    MarketDataMalformedError,
    MarketDataProviderRateLimitError,
    MarketDataProviderUnavailableError,
    MarketDataTickerNotFoundError,
)
from app.providers.market_data import MarketDataProvider

logger = get_logger("app.providers.yahoo_market_data")


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a value to float, preserving missing data as None."""
    if val is None:
        return None
    try:
        if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
            if len(val) == 0:
                return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame columns, flattening MultiIndex if present."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df

    if isinstance(df.columns, pd.MultiIndex):
        first_level = [
            str(col).strip().lower() for col in df.columns.get_level_values(0)
        ]
        if any(c in first_level for c in ["open", "high", "low", "close", "volume"]):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(1)

    return df


class YahooFinanceMarketDataProvider(MarketDataProvider):
    """Historical market data provider backed by Yahoo Finance (yfinance)."""

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "yahoo"

    def get_ohlcv(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> HistoricalMarketData:
        """Retrieve and normalize historical OHLCV market data from Yahoo Finance.

        Args:
            ticker: Stock symbol (e.g. 'AAPL', 'MSFT').
            period: Lookback duration (e.g. '1mo', '3mo', '6mo', '1y', 'max').
            interval: Bar aggregation frequency (e.g. '1d', '1wk', '1mo').

        Returns:
            HistoricalMarketData: Strongly typed normalized domain model.

        Raises:
            MarketDataTickerNotFoundError: If ticker is empty, whitespace, or invalid.
            MarketDataProviderUnavailableError: If connection fails or times out.
            MarketDataProviderRateLimitError: If HTTP 429 rate limit encountered.
            MarketDataMalformedError: If data structure is invalid or corrupt.
            EmptyMarketDataError: If historical series is empty.
            MarketDataError: For unexpected operational failures.
        """
        if not ticker or not isinstance(ticker, str) or not ticker.strip():
            raise MarketDataTickerNotFoundError(
                message="Ticker symbol cannot be empty or whitespace.",
                provider=self.provider_name,
            )

        clean_ticker = ticker.strip().upper()
        logger.info(
            "Fetching historical market data for '%s' (period=%s, interval=%s)",
            clean_ticker,
            period,
            interval,
        )

        try:
            yf_ticker = yfinance.Ticker(clean_ticker)
            # auto_adjust=False ensures both unadjusted Close and Adj Close are returned
            df = yf_ticker.history(period=period, interval=interval, auto_adjust=False)
        except Exception as exc:
            err_msg = str(exc)
            err_lower = err_msg.lower()
            if "429" in err_msg or "too many requests" in err_lower:
                logger.error(
                    "Rate limit error from Yahoo Finance for ticker %s: %s",
                    clean_ticker,
                    err_msg,
                )
                raise MarketDataProviderRateLimitError(
                    message=f"Rate limit exceeded for ticker '{clean_ticker}'.",
                    provider=self.provider_name,
                ) from exc

            if any(
                k in err_lower
                for k in (
                    "not found",
                    "404",
                    "delisted",
                    "no price data found",
                    "invalid ticker",
                    "no timezone found",
                )
            ):
                logger.warning(
                    "Invalid or unknown ticker reported by Yahoo: %s (%s)",
                    clean_ticker,
                    err_msg,
                )
                raise MarketDataTickerNotFoundError(
                    message=(f"Ticker '{clean_ticker}' not found or invalid on Yahoo."),
                    provider=self.provider_name,
                ) from exc

            logger.error(
                "Network/connection failure from Yahoo Finance for ticker %s: %s",
                clean_ticker,
                err_msg,
            )
            raise MarketDataProviderUnavailableError(
                message=(
                    f"Failed to connect to Yahoo market data for '{clean_ticker}'."
                ),
                provider=self.provider_name,
            ) from exc

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            meta = getattr(yf_ticker, "history_metadata", None)
            if isinstance(meta, dict):
                error_val = meta.get("error") or ""
                if (
                    error_val
                    or meta.get("instrumentType") in ("UNKNOWN", None)
                    and not meta.get("currency")
                ):
                    logger.warning(
                        "Yahoo metadata confirms invalid ticker: %s (%s)",
                        clean_ticker,
                        meta,
                    )
                    raise MarketDataTickerNotFoundError(
                        message=(
                            f"Ticker '{clean_ticker}' not found or invalid on "
                            "Yahoo Finance."
                        ),
                        provider=self.provider_name,
                    )

            logger.warning(
                "Empty historical data for '%s' (period=%s, interval=%s)",
                clean_ticker,
                period,
                interval,
            )
            raise EmptyMarketDataError(
                message=(
                    f"No historical market data available for ticker '{clean_ticker}' "
                    f"with period='{period}' and interval='{interval}'."
                ),
                provider=self.provider_name,
            )

        df = _normalize_column_names(df)

        col_map = {str(col).strip().lower(): col for col in df.columns}
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in col_map]
        if missing_cols:
            raise MarketDataMalformedError(
                message=(
                    f"Market data for '{clean_ticker}' missing required columns: "
                    f"{missing_cols}."
                ),
                provider=self.provider_name,
            )

        open_col = col_map["open"]
        high_col = col_map["high"]
        low_col = col_map["low"]
        close_col = col_map["close"]
        volume_col = col_map["volume"]

        adj_col = (
            col_map.get("adj close")
            or col_map.get("adjusted_close")
            or col_map.get("adjclose")
        )

        # Chronologically sort the DataFrame index
        df = df.sort_index(ascending=True)

        candles: List[OHLCVCandle] = []
        seen_timestamps = set()

        for raw_ts, row in df.iterrows():
            # Check for duplicate timestamps in index
            if raw_ts in seen_timestamps:
                raise MarketDataMalformedError(
                    message=(
                        f"Duplicate timestamp detected in historical market data for "
                        f"ticker '{clean_ticker}': {raw_ts}."
                    ),
                    provider=self.provider_name,
                )
            seen_timestamps.add(raw_ts)

            # Extract raw values
            o_raw = row[open_col]
            h_raw = row[high_col]
            l_raw = row[low_col]
            c_raw = row[close_col]
            v_raw = row[volume_col]

            # Check for NaN / Inf in required values
            o_val = _safe_float(o_raw)
            h_val = _safe_float(h_raw)
            l_val = _safe_float(l_raw)
            c_val = _safe_float(c_raw)
            v_val = _safe_float(v_raw)

            if (
                o_val is None
                or h_val is None
                or l_val is None
                or c_val is None
                or v_val is None
            ):
                raise MarketDataMalformedError(
                    message=(
                        f"Missing/NaN price or volume in candle at {raw_ts} "
                        f"for ticker '{clean_ticker}'."
                    ),
                    provider=self.provider_name,
                )

            # Validate adjusted_close:
            # - If column absent: None is valid.
            # - If column present: explicit NaN/Inf must raise MarketDataMalformedError.
            adj_val: Optional[float] = None
            if adj_col is not None:
                adj_raw = row[adj_col]
                if adj_raw is None or pd.isna(adj_raw):
                    raise MarketDataMalformedError(
                        message=(
                            f"Missing or NaN adjusted_close in candle at {raw_ts} "
                            f"for ticker '{clean_ticker}'."
                        ),
                        provider=self.provider_name,
                    )
                adj_float = _safe_float(adj_raw)
                if adj_float is None:
                    raise MarketDataMalformedError(
                        message=(
                            f"NaN/Inf or invalid adjusted_close in candle at {raw_ts} "
                            f"for ticker '{clean_ticker}'."
                        ),
                        provider=self.provider_name,
                    )
                adj_val = adj_float

            try:
                candle = OHLCVCandle(
                    timestamp=raw_ts,
                    open=o_val,
                    high=h_val,
                    low=l_val,
                    close=c_val,
                    volume=v_val,
                    adjusted_close=adj_val,
                )
                candles.append(candle)
            except (ValidationError, ValueError, TypeError) as exc:
                logger.error(
                    "Candle validation error at timestamp %s for ticker %s: %s",
                    raw_ts,
                    clean_ticker,
                    exc,
                )
                raise MarketDataMalformedError(
                    message=(
                        f"Malformed candle data at timestamp {raw_ts} for ticker "
                        f"'{clean_ticker}': {exc}"
                    ),
                    provider=self.provider_name,
                ) from exc

        if not candles:
            raise EmptyMarketDataError(
                message=f"No valid candles extracted for ticker '{clean_ticker}'.",
                provider=self.provider_name,
            )

        try:
            return HistoricalMarketData(
                ticker=clean_ticker,
                candles=candles,
                period=period,
                interval=interval,
                provider=self.provider_name,
                retrieved_at=datetime.now(timezone.utc),
            )
        except (ValidationError, ValueError) as exc:
            raise MarketDataMalformedError(
                message=f"Failed to assemble data for '{clean_ticker}': {exc}",
                provider=self.provider_name,
            ) from exc
