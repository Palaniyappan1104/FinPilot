"""Unit tests for Phase 7.2 Technical Metrics calculation engine.

Verifies:
- Simple Moving Average (SMA 20, 50, 200) series and latest values.
- Exponential Moving Average (EMA 20, 50, 200, 12, 26) with SMA seeding.
- Relative Strength Index (RSI 14) with Wilder's smoothing and edge cases
  (100.0 for zero losses, 0.0 for zero gains, 50.0 for flat price series).
- MACD (12/26/9 line, signal line, histogram) and minimum history requirements.
- Volume metrics (latest volume, 20-period average, volume ratio, zero division).
- Support and resistance detection via local swing extrema.
- Categorical trend detection ('uptrend', 'downtrend', 'sideways').
- Deterministic 0-100 technical scoring with bounded breakdown components.
- Complete TechnicalMetrics model creation and input immutability.
"""

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from app.models.market_data import HistoricalMarketData, OHLCVCandle
from app.models.technical_metrics import (
    MACDMetrics,
    MovingAverageMetrics,
    RSIMetrics,
    VolumeMetrics,
)
from app.services.technical_metrics import (
    calculate_ema,
    calculate_ema_series,
    calculate_macd,
    calculate_rsi,
    calculate_rsi_series,
    calculate_sma,
    calculate_sma_series,
    calculate_support_resistance,
    calculate_technical_metrics,
    calculate_technical_score,
    calculate_volume_metrics,
    detect_trend,
    safe_divide,
)


def make_candle(
    index: int,
    close: float,
    open_: float = None,
    high: float = None,
    low: float = None,
    volume: float = 1_000_000.0,
) -> OHLCVCandle:
    """Helper to create a valid OHLCVCandle with monotonically increasing timestamps."""
    if open_ is None:
        open_ = close
    if high is None:
        high = max(open_, close) + 1.0
    if low is None:
        low = min(open_, close) - 1.0
    return OHLCVCandle(
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        adjusted_close=close,
    )


def make_market_data(
    candles: List[OHLCVCandle], ticker: str = "AAPL"
) -> HistoricalMarketData:
    """Helper to wrap candles in HistoricalMarketData."""
    return HistoricalMarketData(
        ticker=ticker,
        candles=candles,
        period="1y",
        interval="1d",
        total_candles=len(candles),
    )


# ---------------------------------------------------------------------------
# 1. safe_divide tests
# ---------------------------------------------------------------------------


def test_safe_divide_normal():
    assert safe_divide(10.0, 2.0) == 5.0


def test_safe_divide_zero_denominator():
    assert safe_divide(10.0, 0.0) is None


def test_safe_divide_none_inputs():
    assert safe_divide(None, 2.0) is None
    assert safe_divide(10.0, None) is None
    assert safe_divide(None, None) is None


def test_safe_divide_nan_inf():
    assert safe_divide(float("nan"), 2.0) is None
    assert safe_divide(10.0, float("nan")) is None
    assert safe_divide(float("inf"), 2.0) is None


# ---------------------------------------------------------------------------
# 2. SMA Tests
# ---------------------------------------------------------------------------


def test_calculate_sma_series_known_values():
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    series = calculate_sma_series(prices, 3)
    assert len(series) == 5
    assert series[0] is None
    assert series[1] is None
    assert series[2] == pytest.approx(11.0)  # (10+11+12)/3
    assert series[3] == pytest.approx(12.0)  # (11+12+13)/3
    assert series[4] == pytest.approx(13.0)  # (12+13+14)/3


def test_calculate_sma_insufficient_history():
    prices = [10.0, 11.0, 12.0]
    assert calculate_sma(prices, 5) is None
    series = calculate_sma_series(prices, 5)
    assert series == [None, None, None]


def test_calculate_sma_invalid_period():
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_sma_series([10.0, 11.0], 0)
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_sma_series([10.0, 11.0], -5)


def test_calculate_sma_latest_value():
    prices = [10.0, 20.0, 30.0, 40.0]
    assert calculate_sma(prices, 2) == pytest.approx(35.0)  # (30+40)/2
    assert calculate_sma(prices, 4) == pytest.approx(25.0)  # (10+20+30+40)/4


# ---------------------------------------------------------------------------
# 3. EMA Tests (with SMA Seeding)
# ---------------------------------------------------------------------------


def test_calculate_ema_series_known_values_sma_seeding():
    """Verify EMA recurrence and SMA seeding.

    prices = [10.0, 11.0, 12.0, 13.0, 14.0], period = 3
    alpha = 2 / (3 + 1) = 0.5
    Seed at index 2 = SMA(3) = (10 + 11 + 12) / 3 = 11.0
    Index 3 = 13.0 * 0.5 + 11.0 * 0.5 = 12.0
    Index 4 = 14.0 * 0.5 + 12.0 * 0.5 = 13.0
    """
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    series = calculate_ema_series(prices, 3)
    assert series[0] is None
    assert series[1] is None
    assert series[2] == pytest.approx(11.0)
    assert series[3] == pytest.approx(12.0)
    assert series[4] == pytest.approx(13.0)


def test_calculate_ema_latest_value():
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    ema_val = calculate_ema(prices, 3)
    assert ema_val == pytest.approx(13.0)


def test_calculate_ema_insufficient_history():
    prices = [10.0, 11.0]
    assert calculate_ema(prices, 3) is None
    series = calculate_ema_series(prices, 3)
    assert series == [None, None]


def test_calculate_ema_invalid_period():
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_ema_series([10.0, 11.0], 0)
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_ema_series([10.0, 11.0], -1)


# ---------------------------------------------------------------------------
# 4. RSI Tests (Wilder's Smoothing & Edge Cases)
# ---------------------------------------------------------------------------


def test_calculate_rsi_all_gains_returns_100():
    """Strictly increasing prices with zero losses must produce RSI 100.0."""
    prices = [float(i) for i in range(1, 20)]  # 19 prices, all gains
    rsi_val = calculate_rsi(prices, 14)
    assert rsi_val == 100.0


def test_calculate_rsi_all_losses_returns_0():
    """Strictly decreasing prices with zero gains must produce RSI 0.0."""
    prices = [float(20 - i) for i in range(20)]  # 20 prices, all losses
    rsi_val = calculate_rsi(prices, 14)
    assert rsi_val == 0.0


def test_calculate_rsi_flat_series_returns_50():
    """Flat prices with zero gains and zero losses must produce RSI 50.0."""
    prices = [100.0] * 20
    rsi_val = calculate_rsi(prices, 14)
    assert rsi_val == 50.0


def test_calculate_rsi_insufficient_history():
    """RSI requires at least period + 1 prices (15 for 14-period RSI)."""
    prices = [100.0 + i for i in range(14)]  # only 14 prices
    assert calculate_rsi(prices, 14) is None
    series = calculate_rsi_series(prices, 14)
    assert all(v is None for v in series)


def test_calculate_rsi_known_hand_calculated_series():
    """Verify exact Wilder's smoothing formula against manual math.

    Let period = 3, need 4 prices.
    prices = [10.0, 12.0, 11.0, 14.0]
    changes:
    p1-p0 = +2.0 (gain=2, loss=0)
    p2-p1 = -1.0 (gain=0, loss=1)
    p3-p2 = +3.0 (gain=3, loss=0)
    First window (first 3 changes):
    avg_gain_0 = (2 + 0 + 3) / 3 = 5/3
    avg_loss_0 = (0 + 1 + 0) / 3 = 1/3
    RS_0 = (5/3) / (1/3) = 5.0
    RSI_0 = 100 - (100 / (1 + 5)) = 100 - 100/6 = 83.33333333333334
    """
    prices = [10.0, 12.0, 11.0, 14.0]
    series = calculate_rsi_series(prices, 3)
    assert len(series) == 4
    assert series[0] is None
    assert series[1] is None
    assert series[2] is None
    expected_rsi = 100.0 - (100.0 / (1.0 + 5.0))
    assert series[3] == pytest.approx(expected_rsi)


def test_calculate_rsi_wilder_continuation():
    """Verify Wilder recurrence step on 5th price.

    prices = [10.0, 12.0, 11.0, 14.0, 12.0]
    p4-p3 = -2.0 (gain=0, loss=2)
    prev_avg_gain = 5/3, prev_avg_loss = 1/3
    new_avg_gain = (5/3 * 2 + 0) / 3 = 10/9
    new_avg_loss = (1/3 * 2 + 2) / 3 = (2/3 + 2)/3 = (8/3)/3 = 8/9
    RS = (10/9) / (8/9) = 10/8 = 1.25
    RSI = 100 - (100 / (1 + 1.25)) = 100 - (100 / 2.25) = 100 - 44.4444... = 55.5555...
    """
    prices = [10.0, 12.0, 11.0, 14.0, 12.0]
    series = calculate_rsi_series(prices, 3)
    expected_rsi = 100.0 - (100.0 / 2.25)
    assert series[4] == pytest.approx(expected_rsi)


# ---------------------------------------------------------------------------
# 5. MACD Tests (12 / 26 / 9)
# ---------------------------------------------------------------------------


def test_calculate_macd_insufficient_data():
    """MACD line requires 26 prices; signal line requires 26 + 9 - 1 = 34."""
    prices_20 = [100.0 + i for i in range(20)]
    macd_res = calculate_macd(prices_20)
    assert macd_res.macd_line is None
    assert macd_res.signal_line is None
    assert macd_res.histogram is None

    prices_30 = [100.0 + i for i in range(30)]
    macd_res_30 = calculate_macd(prices_30)
    assert macd_res_30.macd_line is not None
    assert macd_res_30.signal_line is None
    assert macd_res_30.histogram is None


def test_calculate_macd_sufficient_data():
    """With >= 34 prices, all three MACD components must be floats."""
    prices_40 = [100.0 + (i * 0.5) for i in range(40)]
    macd_res = calculate_macd(prices_40)
    assert isinstance(macd_res.macd_line, float)
    assert isinstance(macd_res.signal_line, float)
    assert isinstance(macd_res.histogram, float)
    # Histogram = macd_line - signal_line
    assert macd_res.histogram == pytest.approx(
        macd_res.macd_line - macd_res.signal_line
    )


def test_calculate_macd_flat_series():
    """For a perfectly flat series, MACD, Signal, and Histogram should be 0.0."""
    prices_50 = [100.0] * 50
    macd_res = calculate_macd(prices_50)
    assert macd_res.macd_line == pytest.approx(0.0)
    assert macd_res.signal_line == pytest.approx(0.0)
    assert macd_res.histogram == pytest.approx(0.0)


def test_calculate_macd_reference_values_linear():
    """Verify MACD for linear ramp prices P_i = 10 + i (35 prices).

    EMA12 lag = 5.5, EMA26 lag = 12.5 -> difference is exactly 7.0.
    Signal line of constant 7.0 is 7.0, and Histogram is 0.0.
    """
    prices = [10.0 + float(i) for i in range(35)]
    macd = calculate_macd(prices)
    assert macd.macd_line == pytest.approx(7.0)
    assert macd.signal_line == pytest.approx(7.0)
    assert macd.histogram == pytest.approx(0.0, abs=1e-12)


def test_calculate_macd_reference_values_nonlinear():
    """Verify MACD on a non-linear dataset with acceleration (36 prices).

    25 flat prices at 100.0 followed by 11 prices stepping by +2.0.
    Valid MACD starts at index 25 (first difference is 0.0).
    Seed for signal line at index 8 of valid MACD = SMA of first 9 values.
    """
    prices = [100.0] * 25 + [100.0 + (i * 2.0) for i in range(11)]
    macd = calculate_macd(prices)
    assert macd.macd_line == pytest.approx(4.489760882007474)
    assert macd.signal_line == pytest.approx(2.4481740619130132)
    assert macd.histogram == pytest.approx(2.0415868200944605)
    assert macd.histogram == pytest.approx(macd.macd_line - macd.signal_line)


# ---------------------------------------------------------------------------
# 6. Volume Metrics Tests
# ---------------------------------------------------------------------------


def test_calculate_volume_metrics_normal():
    volumes = [1_000_000.0] * 19 + [2_000_000.0]
    vol_metrics = calculate_volume_metrics(volumes, 20)
    assert vol_metrics.latest_volume == 2_000_000.0
    # Average = (19 * 1M + 2M) / 20 = 21M / 20 = 1,050,000.0
    assert vol_metrics.average_volume_20d == pytest.approx(1_050_000.0)
    assert vol_metrics.volume_ratio == pytest.approx(2_000_000.0 / 1_050_000.0)


def test_calculate_volume_metrics_insufficient_history():
    volumes = [1_000_000.0] * 10
    vol_metrics = calculate_volume_metrics(volumes, 20)
    assert vol_metrics.latest_volume == 1_000_000.0
    assert vol_metrics.average_volume_20d is None
    assert vol_metrics.volume_ratio is None


def test_calculate_volume_metrics_empty():
    vol_metrics = calculate_volume_metrics([], 20)
    assert vol_metrics.latest_volume is None
    assert vol_metrics.average_volume_20d is None
    assert vol_metrics.volume_ratio is None


def test_calculate_volume_metrics_zero_average():
    volumes = [0.0] * 20
    vol_metrics = calculate_volume_metrics(volumes, 20)
    assert vol_metrics.latest_volume == 0.0
    assert vol_metrics.average_volume_20d == 0.0
    assert vol_metrics.volume_ratio is None


# ---------------------------------------------------------------------------
# 7. Support & Resistance Tests
# ---------------------------------------------------------------------------


def test_calculate_support_resistance_known_extrema():
    """Test swing extrema detection.

    Candles with distinct peaks and valleys:
    highs: [100, 105, 115, 108, 102, 106, 120, 110, 105, 112]
    lows:  [ 90,  95, 100,  92,  88,  94, 101,  98,  85,  95]
    closes: [95, 100, 110, 100,  95, 100, 115, 105,  90, 105]

    Current price = 105.0.
    Resistance peaks:
    - index 2: high=115 (115 >= 105 and 115 >= 108)
    - index 6: high=120 (120 >= 106 and 120 >= 110)
    Support valleys:
    - index 4: low=88 (88 <= 92 and 88 <= 94)
    - index 8: low=85 (85 <= 98 and 85 <= 95)
    """
    highs = [100, 105, 115, 108, 102, 106, 120, 110, 105, 112]
    lows = [90, 95, 100, 92, 88, 94, 101, 98, 85, 95]
    closes = [95, 100, 110, 100, 95, 100, 115, 105, 90, 105]

    candles = [
        make_candle(i, close=c, high=h, low=low_val)
        for i, (h, low_val, c) in enumerate(zip(highs, lows, closes))
    ]

    sr = calculate_support_resistance(candles, current_price=105.0)

    # Resistances should include 115 and 120, sorted ascending (nearest first)
    assert 115.0 in sr.resistance_levels
    assert 120.0 in sr.resistance_levels
    assert sr.primary_resistance == 115.0

    # Supports should include 88 and 85, sorted descending (nearest first)
    assert 88.0 in sr.support_levels
    assert 85.0 in sr.support_levels
    assert sr.primary_support == 88.0


def test_calculate_support_resistance_insufficient_candles():
    candles = [make_candle(i, close=100.0) for i in range(2)]
    sr = calculate_support_resistance(candles, current_price=100.0)
    assert sr.support_levels == []
    assert sr.resistance_levels == []
    assert sr.primary_support is None
    assert sr.primary_resistance is None


def test_calculate_support_resistance_flat_candles():
    candles = [make_candle(i, close=100.0, high=101.0, low=99.0) for i in range(10)]
    sr = calculate_support_resistance(candles, current_price=100.0)
    assert sr.primary_support is not None
    assert sr.primary_resistance is not None


# ---------------------------------------------------------------------------
# 8. Trend Detection Tests
# ---------------------------------------------------------------------------


def test_detect_trend_uptrend():
    trend = detect_trend(close=110.0, sma_20=105.0, sma_50=100.0)
    assert trend == "uptrend"


def test_detect_trend_downtrend():
    trend = detect_trend(close=90.0, sma_20=95.0, sma_50=100.0)
    assert trend == "downtrend"


def test_detect_trend_sideways_conditions():
    # Price between SMA20 and SMA50
    assert detect_trend(close=102.0, sma_20=105.0, sma_50=100.0) == "sideways"
    # Price below SMA20 but SMA20 > SMA50
    assert detect_trend(close=95.0, sma_20=105.0, sma_50=100.0) == "sideways"
    # Price above SMA20 but SMA20 < SMA50
    assert detect_trend(close=105.0, sma_20=95.0, sma_50=100.0) == "sideways"


def test_detect_trend_insufficient_data():
    # When SMA20 is None
    assert detect_trend(close=100.0, sma_20=None, sma_50=90.0) is None
    # When close is None
    assert detect_trend(close=None, sma_20=100.0, sma_50=90.0) is None


# ---------------------------------------------------------------------------
# 9. Technical Scoring Tests
# ---------------------------------------------------------------------------


def test_calculate_technical_score_bullish():
    """Test score calculation in strongly bullish conditions."""
    mas = MovingAverageMetrics(
        sma_20=115.0,
        sma_50=110.0,
        sma_200=100.0,
        ema_20=116.0,
        ema_50=111.0,
        ema_200=101.0,
    )
    rsi = RSIMetrics(rsi_14=60.0)
    macd = MACDMetrics(macd_line=2.0, signal_line=1.0, histogram=1.0)
    vol = VolumeMetrics(
        latest_volume=1_500_000.0, average_volume_20d=1_000_000.0, volume_ratio=1.5
    )

    score, breakdown = calculate_technical_score(
        close=120.0,
        mas=mas,
        rsi=rsi,
        macd=macd,
        vol=vol,
    )
    assert score is not None
    assert 0.0 <= score <= 100.0
    assert score > 80.0
    assert breakdown.ma_alignment_score == pytest.approx(25.0)
    assert breakdown.rsi_score == pytest.approx(25.0)
    assert breakdown.macd_score == pytest.approx(25.0)
    assert breakdown.volume_score == pytest.approx(25.0)


def test_calculate_technical_score_bearish():
    """Test score calculation in strongly bearish conditions."""
    mas = MovingAverageMetrics(
        sma_20=85.0,
        sma_50=90.0,
        sma_200=100.0,
        ema_20=84.0,
        ema_50=89.0,
        ema_200=99.0,
    )
    rsi = RSIMetrics(rsi_14=25.0)
    macd = MACDMetrics(macd_line=-2.0, signal_line=-1.0, histogram=-1.0)
    vol = VolumeMetrics(
        latest_volume=500_000.0, average_volume_20d=1_000_000.0, volume_ratio=0.5
    )

    score, breakdown = calculate_technical_score(
        close=80.0,
        mas=mas,
        rsi=rsi,
        macd=macd,
        vol=vol,
    )
    assert score is not None
    assert 0.0 <= score <= 100.0
    assert score < 25.0
    assert breakdown.ma_alignment_score == pytest.approx(0.0)
    assert breakdown.rsi_score == pytest.approx(0.0)
    assert breakdown.macd_score == pytest.approx(0.0)
    assert breakdown.volume_score == pytest.approx(10.0)


def test_calculate_technical_score_insufficient_data():
    """When all inputs are empty/None, score should return None."""
    score, breakdown = calculate_technical_score(
        close=100.0,
        mas=MovingAverageMetrics(),
        rsi=RSIMetrics(),
        macd=MACDMetrics(),
        vol=VolumeMetrics(),
    )
    assert score is None
    assert breakdown is None


def test_calculate_technical_score_data_sufficiency_policy():
    """Enforce data-sufficiency safety: require at least 3 pillars (75 pts).

    Prevents misleading 0-100 scores from isolated single or dual indicators.
    """
    # 1. Only RSI available (25 pts available) -> None
    score_1, b_1 = calculate_technical_score(
        close=100.0,
        mas=MovingAverageMetrics(),
        rsi=RSIMetrics(rsi_14=60.0),
        macd=MACDMetrics(),
        vol=VolumeMetrics(),
    )
    assert score_1 is None
    assert b_1 is None

    # 2. Only RSI + Volume available (50 pts available) -> None
    score_2, b_2 = calculate_technical_score(
        close=100.0,
        mas=MovingAverageMetrics(),
        rsi=RSIMetrics(rsi_14=60.0),
        macd=MACDMetrics(),
        vol=VolumeMetrics(volume_ratio=1.5),
    )
    assert score_2 is None
    assert b_2 is None

    # 3. 3 pillars available: MA + RSI + MACD without volume (75 pts) -> Valid score
    score_3, b_3 = calculate_technical_score(
        close=120.0,
        mas=MovingAverageMetrics(sma_20=115.0, sma_50=110.0),
        rsi=RSIMetrics(rsi_14=60.0),
        macd=MACDMetrics(macd_line=2.0, signal_line=1.0, histogram=1.0),
        vol=VolumeMetrics(),
    )
    assert score_3 is not None
    assert 0.0 <= score_3 <= 100.0
    assert b_3 is not None
    assert b_3.ma_alignment_score is not None
    assert b_3.rsi_score is not None
    assert b_3.macd_score is not None
    assert b_3.volume_score is None

    # 4. All 4 pillars available (100 pts) -> Valid score
    score_4, b_4 = calculate_technical_score(
        close=120.0,
        mas=MovingAverageMetrics(sma_20=115.0, sma_50=110.0),
        rsi=RSIMetrics(rsi_14=60.0),
        macd=MACDMetrics(macd_line=2.0, signal_line=1.0, histogram=1.0),
        vol=VolumeMetrics(volume_ratio=1.5),
    )
    assert score_4 is not None
    assert 0.0 <= score_4 <= 100.0
    assert b_4 is not None
    assert b_4.volume_score is not None


def test_calculate_technical_score_deterministic_reproducibility():
    """Score must be bitwise reproducible across multiple invocations."""
    mas = MovingAverageMetrics(sma_20=115.0, sma_50=110.0, sma_200=100.0)
    rsi = RSIMetrics(rsi_14=58.5)
    macd = MACDMetrics(macd_line=1.5, signal_line=0.8, histogram=0.7)
    vol = VolumeMetrics(volume_ratio=1.25)

    score_1, b_1 = calculate_technical_score(122.0, mas, rsi, macd, vol)
    score_2, b_2 = calculate_technical_score(122.0, mas, rsi, macd, vol)

    assert score_1 == score_2
    assert b_1.model_dump() == b_2.model_dump()


def test_calculate_technical_score_strictly_bounded():
    """Score must strictly stay in [0.0, 100.0] under all configurations."""
    for rsi_val in [0.0, 20.0, 50.0, 70.0, 95.0, 100.0]:
        for macd_diff in [-5.0, 0.0, 5.0]:
            mas = MovingAverageMetrics(sma_20=100.0, sma_50=100.0)
            rsi = RSIMetrics(rsi_14=rsi_val)
            macd = MACDMetrics(
                macd_line=macd_diff, signal_line=0.0, histogram=macd_diff
            )
            vol = VolumeMetrics(volume_ratio=1.0)
            score, _ = calculate_technical_score(100.0, mas, rsi, macd, vol)
            if score is not None:
                assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# 10. Complete Pipeline & HistoricalMarketData Integration
# ---------------------------------------------------------------------------


def test_calculate_technical_metrics_full_dataset():
    """Full 250-day dataset generating all indicators."""
    candles = [
        make_candle(
            i,
            close=100.0 + (i * 0.4),
            high=102.0 + (i * 0.4),
            low=99.0 + (i * 0.4),
            volume=1_000_000.0 + (i * 1000),
        )
        for i in range(250)
    ]
    market_data = make_market_data(candles, ticker="NVDA")

    metrics = calculate_technical_metrics(market_data)

    assert metrics.ticker == "NVDA"
    assert metrics.candle_count == 250
    assert metrics.latest_close == pytest.approx(candles[-1].close)

    # Moving averages
    assert metrics.moving_averages.sma_20 is not None
    assert metrics.moving_averages.sma_50 is not None
    assert metrics.moving_averages.sma_200 is not None
    assert metrics.moving_averages.ema_20 is not None
    assert metrics.moving_averages.ema_50 is not None
    assert metrics.moving_averages.ema_200 is not None
    assert metrics.moving_averages.ema_12 is not None
    assert metrics.moving_averages.ema_26 is not None

    # RSI
    assert metrics.rsi.rsi_14 is not None
    assert 0.0 <= metrics.rsi.rsi_14 <= 100.0

    # MACD
    assert metrics.macd.macd_line is not None
    assert metrics.macd.signal_line is not None
    assert metrics.macd.histogram is not None

    # Volume
    assert metrics.volume.latest_volume == candles[-1].volume
    assert metrics.volume.average_volume_20d is not None
    assert metrics.volume.volume_ratio is not None

    # Trend & Score
    assert metrics.trend == "uptrend"
    assert 0.0 <= metrics.technical_score <= 100.0
    assert metrics.score_breakdown is not None


def test_calculate_technical_metrics_short_history():
    """Short 10-day dataset: 20/50/200 period indicators must be None."""
    candles = [make_candle(i, close=100.0 + i) for i in range(10)]
    market_data = make_market_data(candles, ticker="TEST")

    metrics = calculate_technical_metrics(market_data)

    assert metrics.candle_count == 10
    assert metrics.moving_averages.sma_20 is None
    assert metrics.moving_averages.sma_50 is None
    assert metrics.moving_averages.sma_200 is None
    assert metrics.moving_averages.ema_20 is None
    assert metrics.rsi.rsi_14 is None  # requires 15 prices
    assert metrics.macd.macd_line is None  # requires 26 prices
    assert metrics.trend is None  # requires >= 20 candles
    assert metrics.technical_score is None  # insufficient indicator pillars


def test_calculate_technical_metrics_input_immutability():
    """Verify market_data is not mutated during indicator computation."""
    candles = [make_candle(i, close=100.0 + i) for i in range(30)]
    market_data = make_market_data(candles, ticker="IMMUTABLE")

    original_candles_repr = [c.model_dump() for c in market_data.candles]

    _ = calculate_technical_metrics(market_data)

    after_candles_repr = [c.model_dump() for c in market_data.candles]
    assert original_candles_repr == after_candles_repr
