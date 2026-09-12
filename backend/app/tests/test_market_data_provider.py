"""Unit tests for Market Data Provider abstraction, models, and Yahoo provider.

Phase 7.1 test suite verifying:
- Successful retrieval and normalization of historical OHLCV candles.
- Complete requested historical range without artificial truncation.
- Strict isolation from pandas and yfinance objects.
- Validation of OHLC integrity, strictly positive prices, and non-negative volume.
- Handling of missing optional adjusted close without fabrication.
- Detection and rejection of duplicate timestamps, NaN/Inf, and malformed records.
- Preservation of weekend/holiday trading gaps without synthetic candle insertion.
- Distinguishing HTTP 429 rate limits from generic connection failures.
- Handling invalid tickers, empty histories, and unsupported factory configs.
- Guarantee of zero live network calls in unit tests.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.models.market_data import HistoricalMarketData, OHLCVCandle
from app.providers.exceptions import (
    EmptyMarketDataError,
    MarketDataError,
    MarketDataMalformedError,
    MarketDataProviderRateLimitError,
    MarketDataProviderUnavailableError,
    MarketDataTickerNotFoundError,
)
from app.providers.market_data import (
    get_market_data_provider,
)
from app.providers.yahoo_market_data import YahooFinanceMarketDataProvider


@pytest.fixture
def mock_daily_ohlcv_df() -> pd.DataFrame:
    """Deterministic sample 5-day daily OHLCV DataFrame mirroring yfinance output."""
    dates = [
        pd.Timestamp("2024-01-08 00:00:00", tz="UTC"),  # Monday
        pd.Timestamp("2024-01-09 00:00:00", tz="UTC"),  # Tuesday
        pd.Timestamp("2024-01-10 00:00:00", tz="UTC"),  # Wednesday
        pd.Timestamp("2024-01-11 00:00:00", tz="UTC"),  # Thursday
        pd.Timestamp("2024-01-12 00:00:00", tz="UTC"),  # Friday
    ]
    data = {
        "Open": [182.15, 183.92, 184.35, 186.12, 185.50],
        "High": [185.60, 186.40, 187.05, 187.50, 186.90],
        "Low": [181.50, 183.00, 183.80, 185.00, 184.20],
        "Close": [185.14, 185.59, 186.19, 185.59, 185.92],
        "Adj Close": [184.50, 184.95, 185.55, 184.95, 185.28],
        "Volume": [59144500.0, 46792900.0, 46792900.0, 49128400.0, 40477800.0],
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates))


# ===========================================================================
# 1. Domain Model Unit Tests: OHLCVCandle & HistoricalMarketData
# ===========================================================================


def test_candle_valid_construction():
    """Verify valid OHLCVCandle construction and property preservation."""
    ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    candle = OHLCVCandle(
        timestamp=ts,
        open=150.0,
        high=155.0,
        low=149.0,
        close=154.0,
        volume=100000.0,
        adjusted_close=153.5,
    )
    assert candle.timestamp == ts
    assert candle.open == 150.0
    assert candle.high == 155.0
    assert candle.low == 149.0
    assert candle.close == 154.0
    assert candle.volume == 100000.0
    assert candle.adjusted_close == 153.5


def test_candle_missing_optional_adjusted_close():
    """Verify adjusted_close defaults to None and is preserved as None."""
    candle = OHLCVCandle(
        timestamp="2024-01-15T00:00:00Z",
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=5000.0,
    )
    assert candle.adjusted_close is None


def test_candle_rejects_negative_volume():
    """Verify negative volume raises validation error."""
    with pytest.raises(ValidationError, match="volume must be non-negative"):
        OHLCVCandle(
            timestamp="2024-01-15T00:00:00Z",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=-100.0,
        )


def test_candle_rejects_zero_or_negative_price():
    """Verify zero or negative price raises validation error."""
    with pytest.raises(ValidationError, match="open must be strictly positive"):
        OHLCVCandle(
            timestamp="2024-01-15T00:00:00Z",
            open=0.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=5000.0,
        )

    with pytest.raises(ValidationError, match="close must be strictly positive"):
        OHLCVCandle(
            timestamp="2024-01-15T00:00:00Z",
            open=100.0,
            high=105.0,
            low=95.0,
            close=-5.0,
            volume=5000.0,
        )


def test_candle_rejects_nan_and_inf():
    """Verify NaN and Inf prices or volume are rejected."""
    with pytest.raises(ValidationError, match="high cannot be NaN or Inf"):
        OHLCVCandle(
            timestamp="2024-01-15T00:00:00Z",
            open=100.0,
            high=float("nan"),
            low=95.0,
            close=102.0,
            volume=5000.0,
        )

    with pytest.raises(ValidationError, match="low cannot be NaN or Inf"):
        OHLCVCandle(
            timestamp="2024-01-15T00:00:00Z",
            open=100.0,
            high=105.0,
            low=float("-inf"),
            close=102.0,
            volume=5000.0,
        )


def test_candle_rejects_invalid_ohlc_relationships():
    """Verify high/low envelope constraints are strictly enforced."""
    # High < Low
    with pytest.raises(ValidationError, match="High price .* cannot be less than low"):
        OHLCVCandle(
            timestamp="2024-01-15T00:00:00Z",
            open=100.0,
            high=90.0,
            low=95.0,
            close=92.0,
            volume=100.0,
        )

    # Open > High
    with pytest.raises(ValidationError, match="High price .* cannot be less than open"):
        OHLCVCandle(
            timestamp="2024-01-15T00:00:00Z",
            open=110.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=100.0,
        )

    # Close < Low
    with pytest.raises(
        ValidationError, match="Low price .* cannot be greater than close"
    ):
        OHLCVCandle(
            timestamp="2024-01-15T00:00:00Z",
            open=100.0,
            high=105.0,
            low=95.0,
            close=90.0,
            volume=100.0,
        )


def test_historical_market_data_chronological_validation():
    """Verify HistoricalMarketData rejects non-chronological candle ordering."""
    c1 = OHLCVCandle(
        timestamp="2024-01-10T00:00:00Z",
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=100.0,
    )
    c2 = OHLCVCandle(
        timestamp="2024-01-09T00:00:00Z",  # Earlier than c1
        open=101.0,
        high=106.0,
        low=96.0,
        close=103.0,
        volume=100.0,
    )

    with pytest.raises(ValidationError, match="Candles not chronological"):
        HistoricalMarketData(ticker="AAPL", candles=[c1, c2])


def test_historical_market_data_rejects_duplicate_timestamps():
    """Verify HistoricalMarketData rejects identical timestamps with clear error."""
    c1 = OHLCVCandle(
        timestamp="2024-01-10T00:00:00Z",
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=100.0,
    )
    c2 = OHLCVCandle(
        timestamp="2024-01-10T00:00:00Z",  # Duplicate timestamp
        open=101.0,
        high=106.0,
        low=96.0,
        close=103.0,
        volume=100.0,
    )

    with pytest.raises(ValidationError, match="Duplicate timestamp detected"):
        HistoricalMarketData(ticker="AAPL", candles=[c1, c2])


# ===========================================================================
# 2. YahooFinanceMarketDataProvider Normalization & Integrity Tests
# ===========================================================================


@patch("yfinance.Ticker")
def test_valid_ohlcv_normalization(mock_ticker_cls, mock_daily_ohlcv_df):
    """Verify valid OHLCV normalization into HistoricalMarketData."""
    instance = MagicMock()
    instance.history.return_value = mock_daily_ohlcv_df
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    result = provider.get_ohlcv("aapl", period="1y", interval="1d")

    assert isinstance(result, HistoricalMarketData)
    assert result.ticker == "AAPL"
    assert result.period == "1y"
    assert result.interval == "1d"
    assert result.provider == "yahoo"
    assert len(result.candles) == 5

    first_candle = result.candles[0]
    assert first_candle.open == 182.15
    assert first_candle.high == 185.60
    assert first_candle.low == 181.50
    assert first_candle.close == 185.14
    assert first_candle.adjusted_close == 184.50
    assert first_candle.volume == 59144500.0

    # Ensure no pandas or yfinance instances leaked
    for candle in result.candles:
        assert isinstance(candle.timestamp, datetime)
        assert isinstance(candle.open, float)
        assert not isinstance(candle.open, pd.Series)
        assert not isinstance(candle.timestamp, pd.Timestamp)


@patch("yfinance.Ticker")
def test_out_of_order_provider_records_sorted_chronologically(
    mock_ticker_cls, mock_daily_ohlcv_df
):
    """Verify candles arriving out of order are sorted chronologically."""
    # Reverse the DataFrame rows
    reversed_df = mock_daily_ohlcv_df.iloc[::-1].copy()
    instance = MagicMock()
    instance.history.return_value = reversed_df
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    result = provider.get_ohlcv("MSFT", period="1mo", interval="1d")

    assert len(result.candles) == 5
    timestamps = [c.timestamp for c in result.candles]
    assert timestamps == sorted(timestamps)
    assert result.candles[0].open == 182.15  # Corresponds to Jan 8


@patch("yfinance.Ticker")
def test_complete_requested_historical_range_preserved(mock_ticker_cls):
    """Verify provider does NOT truncate or limit records (e.g. preserves 300 days)."""
    dates = pd.date_range("2023-01-01", periods=300, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(300)],
            "High": [105.0 + i for i in range(300)],
            "Low": [95.0 + i for i in range(300)],
            "Close": [102.0 + i for i in range(300)],
            "Volume": [1000000.0 for _ in range(300)],
        },
        index=dates,
    )
    instance = MagicMock()
    instance.history.return_value = df
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    result = provider.get_ohlcv("GOOGL", period="2y", interval="1d")

    # Complete 300 candles must be returned, not truncated to 200 or 250
    assert len(result.candles) == 300
    assert result.candles[-1].close == 102.0 + 299


@patch("yfinance.Ticker")
def test_preserves_weekend_and_holiday_gaps_without_fabrication(mock_ticker_cls):
    """Verify genuine market gaps are preserved and no fake bars inserted."""

    dates = [
        pd.Timestamp("2024-01-12 00:00:00", tz="UTC"),  # Friday
        pd.Timestamp(
            "2024-01-16 00:00:00", tz="UTC"
        ),  # Tuesday (MLK Day on Monday Jan 15)
    ]
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [105.0, 107.0],
            "Low": [98.0, 101.0],
            "Close": [103.0, 106.0],
            "Volume": [1000.0, 2000.0],
        },
        index=pd.DatetimeIndex(dates),
    )
    instance = MagicMock()
    instance.history.return_value = df
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    result = provider.get_ohlcv("TSLA")

    # Exactly 2 trading days preserved; no weekend or holiday candles fabricated
    assert len(result.candles) == 2
    assert result.candles[0].timestamp.day == 12
    assert result.candles[1].timestamp.day == 16


@patch("yfinance.Ticker")
def test_missing_optional_adjusted_close_handled(mock_ticker_cls, mock_daily_ohlcv_df):
    """Verify DataFrame lacking 'Adj Close' sets adjusted_close=None for each candle."""
    df_no_adj = mock_daily_ohlcv_df.drop(columns=["Adj Close"])
    instance = MagicMock()
    instance.history.return_value = df_no_adj
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    result = provider.get_ohlcv("AAPL")

    assert len(result.candles) == 5
    for candle in result.candles:
        assert candle.adjusted_close is None


# ===========================================================================
# 3. Error Handling & Exception Tests
# ===========================================================================


def test_invalid_ticker_symbols():
    """Verify invalid or empty tickers raise MarketDataTickerNotFoundError."""
    provider = YahooFinanceMarketDataProvider()

    for invalid in ["", "   ", None, 123]:
        with pytest.raises(
            MarketDataTickerNotFoundError, match="Ticker symbol cannot be empty"
        ):
            provider.get_ohlcv(invalid)


@patch("yfinance.Ticker")
def test_empty_historical_data_raises_empty_error(mock_ticker_cls):
    """Verify empty DataFrame from provider raises EmptyMarketDataError."""
    instance = MagicMock()
    instance.history.return_value = pd.DataFrame()
    instance.history_metadata = None
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(
        EmptyMarketDataError, match="No historical market data available"
    ):
        provider.get_ohlcv("VALID_TICKER")


@patch("yfinance.Ticker")
def test_valid_ticker_empty_history_raises_empty_market_data_error(mock_ticker_cls):
    """Verify valid ticker with empty historical data raises EmptyMarketDataError."""
    instance = MagicMock()
    instance.history.return_value = pd.DataFrame()
    instance.history_metadata = {"currency": "USD", "instrumentType": "EQUITY"}
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(
        EmptyMarketDataError, match="No historical market data available"
    ):
        provider.get_ohlcv("NEW_IPO")


@patch("yfinance.Ticker")
def test_unknown_ticker_exception_raises_ticker_not_found(mock_ticker_cls):
    """Verify Yahoo 404/not found exception raises MarketDataTickerNotFoundError."""
    instance = MagicMock()
    instance.history.side_effect = Exception(
        "404 Client Error: Not Found for symbol: INVALID"
    )
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(MarketDataTickerNotFoundError, match="not found or invalid"):
        provider.get_ohlcv("INVALID")


@patch("yfinance.Ticker")
def test_unknown_ticker_metadata_raises_ticker_not_found(mock_ticker_cls):
    """Verify empty history with error metadata raises MarketDataTickerNotFoundError."""
    instance = MagicMock()
    instance.history.return_value = pd.DataFrame()
    instance.history_metadata = {"error": "Not Found", "validRanges": []}
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(MarketDataTickerNotFoundError, match="not found or invalid"):
        provider.get_ohlcv("UNKNOWN_TICKER")


@patch("yfinance.Ticker")
def test_nan_adjusted_close_raises_malformed_error(
    mock_ticker_cls, mock_daily_ohlcv_df
):
    """Verify row with NaN in adjusted_close raises MarketDataMalformedError."""
    df_bad = mock_daily_ohlcv_df.copy()
    df_bad.loc[df_bad.index[0], "Adj Close"] = np.nan
    instance = MagicMock()
    instance.history.return_value = df_bad
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(MarketDataMalformedError, match="Missing or NaN adjusted_close"):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_inf_adjusted_close_raises_malformed_error(
    mock_ticker_cls, mock_daily_ohlcv_df
):
    """Verify row with Inf in adjusted_close raises MarketDataMalformedError."""
    df_bad = mock_daily_ohlcv_df.copy()
    df_bad.loc[df_bad.index[0], "Adj Close"] = np.inf
    instance = MagicMock()
    instance.history.return_value = df_bad
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(
        MarketDataMalformedError, match="NaN/Inf or invalid adjusted_close"
    ):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_rate_limit_error_mapping(mock_ticker_cls):
    """Verify HTTP 429 raises MarketDataProviderRateLimitError."""
    instance = MagicMock()
    instance.history.side_effect = Exception("HTTP 429 Too Many Requests")
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(MarketDataProviderRateLimitError, match="Rate limit exceeded"):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_provider_unavailable_error_mapping(mock_ticker_cls):
    """Verify connection failure raises MarketDataProviderUnavailableError."""
    instance = MagicMock()
    instance.history.side_effect = Exception(
        "Failed to establish a new connection: Timeout"
    )
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(
        MarketDataProviderUnavailableError, match="Failed to connect to Yahoo"
    ):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_missing_required_column_raises_malformed_error(
    mock_ticker_cls, mock_daily_ohlcv_df
):
    """Verify missing required column (e.g. 'Close') raises MarketDataMalformedError."""
    df_missing_close = mock_daily_ohlcv_df.drop(columns=["Close"])
    instance = MagicMock()
    instance.history.return_value = df_missing_close
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(
        MarketDataMalformedError, match="missing required columns.*close"
    ):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_nan_price_row_raises_malformed_error(mock_ticker_cls, mock_daily_ohlcv_df):
    """Verify row containing NaN raises MarketDataMalformedError."""
    df_nan = mock_daily_ohlcv_df.copy()
    df_nan.loc[df_nan.index[2], "Open"] = np.nan
    instance = MagicMock()
    instance.history.return_value = df_nan
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(MarketDataMalformedError, match="Missing.*NaN"):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_inf_price_row_raises_malformed_error(mock_ticker_cls, mock_daily_ohlcv_df):
    """Verify row containing Inf raises MarketDataMalformedError."""
    df_inf = mock_daily_ohlcv_df.copy()
    df_inf.loc[df_inf.index[1], "High"] = np.inf
    instance = MagicMock()
    instance.history.return_value = df_inf
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(MarketDataMalformedError, match="Missing/NaN price or volume"):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_negative_volume_raises_malformed_error(mock_ticker_cls, mock_daily_ohlcv_df):
    """Verify candle with negative volume raises MarketDataMalformedError."""
    df_bad = mock_daily_ohlcv_df.copy()
    df_bad.loc[df_bad.index[0], "Volume"] = -500.0
    instance = MagicMock()
    instance.history.return_value = df_bad
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(MarketDataMalformedError, match="Malformed candle data"):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_invalid_ohlc_relationship_raises_malformed_error(
    mock_ticker_cls, mock_daily_ohlcv_df
):
    """Verify candle where high < low raises MarketDataMalformedError."""
    df_bad = mock_daily_ohlcv_df.copy()
    df_bad.loc[df_bad.index[0], "High"] = 150.0  # Open is 182, Low is 181.5
    instance = MagicMock()
    instance.history.return_value = df_bad
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(
        MarketDataMalformedError, match="High price .* cannot be less than low price"
    ):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_open_greater_than_high_raises_malformed_error(
    mock_ticker_cls, mock_daily_ohlcv_df
):
    """Verify candle where open > high raises MarketDataMalformedError."""
    df_bad = mock_daily_ohlcv_df.copy()
    df_bad.loc[df_bad.index[0], "Open"] = 200.0  # High is 185.60, Low is 181.50
    instance = MagicMock()
    instance.history.return_value = df_bad
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(
        MarketDataMalformedError, match="High price .* cannot be less than open"
    ):
        provider.get_ohlcv("AAPL")


@patch("yfinance.Ticker")
def test_duplicate_timestamps_rejected_with_malformed_error(
    mock_ticker_cls, mock_daily_ohlcv_df
):
    """Verify duplicate timestamps raise MarketDataMalformedError."""
    df_dup = pd.concat([mock_daily_ohlcv_df.iloc[[0]], mock_daily_ohlcv_df])
    instance = MagicMock()
    instance.history.return_value = df_dup
    mock_ticker_cls.return_value = instance

    provider = YahooFinanceMarketDataProvider()
    with pytest.raises(MarketDataMalformedError, match="Duplicate timestamp detected"):
        provider.get_ohlcv("AAPL")


# ===========================================================================
# 4. Factory & Configuration Tests
# ===========================================================================


def test_factory_returns_yahoo_provider_by_default():
    """Verify factory returns YahooFinanceMarketDataProvider with default settings."""
    provider = get_market_data_provider()
    assert isinstance(provider, YahooFinanceMarketDataProvider)
    assert provider.provider_name == "yahoo"


def test_factory_unsupported_provider_raises_error():
    """Verify factory raises MarketDataError when unsupported provider is configured."""
    custom_settings = Settings(MARKET_DATA_PROVIDER="bloomberg")
    with pytest.raises(
        MarketDataError, match="Unsupported MARKET_DATA_PROVIDER 'bloomberg'"
    ):
        get_market_data_provider(settings=custom_settings)


def test_factory_empty_provider_raises_error():
    """Verify factory raises MarketDataError when provider setting is empty."""
    custom_settings = Settings(MARKET_DATA_PROVIDER="")
    with pytest.raises(
        MarketDataError, match="MARKET_DATA_PROVIDER configuration cannot be empty"
    ):
        get_market_data_provider(settings=custom_settings)
